"""Real pipeline runner using Datapizza agents and Gemini intake."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import re
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError

from tripplanner.cache import MemoryCache, RateLimiter
from tripplanner.contracts import PlanTask, StandardAgentResult
from tripplanner.executor import OrchestratorExecutor
from tripplanner.geo_tool import GeoTool, GeoToolError, NominatimClient
from tripplanner.itinerary_synth import ItinerarySynthesizer
from tripplanner.orchestrator_intake import OrchestratorIntake
from tripplanner.planner import OrchestratorPlanner
from tripplanner.poi_tool import OverpassClient, POITool
from tripplanner.search_tool import DuckDuckGoClient, SearchTool
from tripplanner.specialist_agents import GeoAgent, POIAgent, TransportAgent, WeatherAgent
from tripplanner.telemetry import start_span
from tripplanner.weather_tool import OpenMeteoClient, WeatherTool

T = TypeVar("T")


def load_env_file(path: str = ".env") -> None:
    """Best-effort .env loader without external dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


class PipelineRunner:
    """End-to-end runner for real Datapizza+Gemini execution."""

    def __init__(self) -> None:
        cache = MemoryCache(max_size=512)
        self._intake = OrchestratorIntake()
        self._planner = OrchestratorPlanner()
        self._synth = ItinerarySynthesizer()
        self._geo_agent = GeoAgent(
            geo_tool=GeoTool(
                client=NominatimClient(),
                cache=cache,
                rate_limiter=RateLimiter(rate_per_second=1.0, capacity=1.0),
            )
        )
        self._weather_agent = WeatherAgent(
            weather_tool=WeatherTool(
                client=OpenMeteoClient(),
                cache=cache,
            )
        )
        search_tool = SearchTool(client=DuckDuckGoClient(), cache=cache)
        self._poi_agent = POIAgent(
            poi_tool=POITool(
                client=OverpassClient(),
                cache=cache,
                rate_limiter=RateLimiter(rate_per_second=0.5, capacity=1.0),
            ),
            search_tool=search_tool,
        )
        self._transport_agent = TransportAgent(search_tool=search_tool)

    def run(
        self,
        query: str,
        *,
        now_ts: datetime | None = None,
        timezone_name: str = "UTC",
        output_language: str | None = None,
    ) -> dict[str, Any]:
        request_now = now_ts or datetime.now(timezone.utc)
        runtime_results: dict[str, StandardAgentResult] = {}

        with start_span("orchestrator.extract_tripspec"):
            intake = self._intake.process(query, now_ts=request_now, timezone_name=timezone_name)
            if intake.status == "clarification_needed":
                return {
                    "status": "clarification_needed",
                    "clarifying_question": " ".join(intake.clarifying_question.questions),
                    "reason": intake.clarifying_question.reason,
                }
            tripspec = intake.tripspec
            if output_language in {"en", "it"}:
                tripspec.request_context.output_language = output_language

        with start_span("orchestrator.plan"):
            plan = self._planner.generate(tripspec)

        def leg_index_for(task: PlanTask) -> int:
            match = re.search(r"legs\[(\d+)\]", task.input_ref)
            if not match:
                raise ValueError(f"Cannot resolve leg index from input_ref={task.input_ref}")
            return int(match.group(1))

        def geo_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.geo"):
                leg_idx = leg_index_for(task)
                leg = tripspec.legs[leg_idx]
                result = _invoke_with_geo_rate_limit_retry(
                    lambda: self._geo_agent.invoke(
                        {
                            "query": leg.destination_text,
                            "locale": tripspec.request_context.output_language,
                            "limit": 5,
                        }
                    )
                )
                runtime_results[task.task_id] = result
                return result

        def weather_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.weather"):
                leg_idx = leg_index_for(task)
                leg = tripspec.legs[leg_idx]
                lat = leg.geo.lat if leg.geo else None
                lon = leg.geo.lon if leg.geo else None
                if lat is None or lon is None:
                    geo = runtime_results.get(f"geo_leg_{leg_idx}")
                    selected = (geo.data if geo else {}).get("selected", {})
                    if isinstance(selected, dict):
                        raw_lat = selected.get("lat")
                        raw_lon = selected.get("lon")
                        if raw_lat is not None and raw_lon is not None:
                            lat = float(raw_lat)
                            lon = float(raw_lon)
                if lat is None or lon is None:
                    raise RuntimeError("Missing coordinates for weather execution.")
                result = _invoke_with_transient_retry(
                    lambda: self._weather_agent.invoke(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "start_date": leg.date_range.start_date.isoformat(),
                            "end_date": leg.date_range.end_date.isoformat(),
                            "timezone_name": tripspec.request_context.timezone,
                        }
                    )
                )
                runtime_results[task.task_id] = result
                return result

        def poi_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.poi"):
                leg_idx = leg_index_for(task)
                leg = tripspec.legs[leg_idx]
                bbox = leg.geo.bbox if leg.geo else None
                if bbox is None:
                    geo = runtime_results.get(f"geo_leg_{leg_idx}")
                    selected = (geo.data if geo else {}).get("selected", {})
                    candidate_bbox = selected.get("bbox") if isinstance(selected, dict) else None
                    if isinstance(candidate_bbox, list) and len(candidate_bbox) == 4:
                        bbox = [float(v) for v in candidate_bbox]
                if bbox is None:
                    raise RuntimeError("Missing bbox for POI execution.")
                result = _invoke_with_transient_retry(
                    lambda: self._poi_agent.invoke(
                        {
                            "bbox": bbox,
                            "tags": tripspec.preferences.tags or ["tourism=attraction", "tourism=museum"],
                            "limit": 8,
                            "locale": tripspec.request_context.output_language,
                            "enrichment_query": f"best things to do in {leg.destination_text}",
                        }
                    ),
                    max_attempts=3,
                )
                runtime_results[task.task_id] = result
                return result

        def transport_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.transport"):
                match = re.search(r"legs\[(\d+)\]->legs\[(\d+)\]", task.input_ref)
                if not match:
                    raise ValueError(f"Invalid transport input_ref={task.input_ref}")
                from_idx = int(match.group(1))
                to_idx = int(match.group(2))
                result = _invoke_with_transient_retry(
                    lambda: self._transport_agent.invoke(
                        {
                            "origin": tripspec.legs[from_idx].destination_text,
                            "destination": tripspec.legs[to_idx].destination_text,
                            "departure_date": tripspec.legs[to_idx].date_range.start_date.isoformat(),
                            "locale": tripspec.request_context.output_language,
                            "limit": 4,
                            "mode_preferences": ["train", "bus", "flight"],
                        }
                    )
                )
                runtime_results[task.task_id] = result
                return result

        def synth_handler(task: PlanTask) -> StandardAgentResult:
            marker = _synth_marker_result(task.task_id)
            runtime_results[task.task_id] = marker
            return marker

        outcome = OrchestratorExecutor(
            handlers={
                "geo": geo_handler,
                "weather": weather_handler,
                "poi": poi_handler,
                "transport": transport_handler,
                "synth": synth_handler,
            }
        ).execute(plan)

        with start_span("orchestrator.evaluate"):
            if outcome.status == "clarification_needed":
                return {
                    "status": "clarification_needed",
                    "clarifying_question": outcome.clarifying_question,
                    "stages": outcome.stages,
                }

        with start_span("orchestrator.synthesize"):
            itinerary = self._synth.synthesize(tripspec=tripspec, results=outcome.results)

        return {
            "status": "completed",
            "language": itinerary.language,
            "title": itinerary.title,
            "days": [day.model_dump(mode="json") for day in itinerary.days],
            "warnings": itinerary.warnings,
            "stages": outcome.stages,
            "itinerary_text": _render_day_by_day(itinerary.days),
        }


def run_pipeline(
    query: str,
    *,
    now_ts: datetime | None = None,
    timezone_name: str = "UTC",
    output_language: str | None = None,
) -> dict[str, Any]:
    load_env_file(".env")
    return PipelineRunner().run(
        query,
        now_ts=now_ts,
        timezone_name=timezone_name,
        output_language=output_language,
    )


def _synth_marker_result(task_id: str) -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": {"task_id": task_id, "status": "synthesized"},
            "evidence": [
                {
                    "source": "synth",
                    "title": "Synthesis stage marker",
                    "snippet": "Synthesis stage completed.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 1.0,
            "warnings": [],
            "cache_key": f"synth:{task_id}",
        }
    )


def _invoke_with_geo_rate_limit_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except GeoToolError as exc:
            last_error = exc
            wait_s = _extract_wait_seconds(str(exc))
            if wait_s is None or attempt >= max_attempts:
                raise
            sleep_fn(wait_s + 0.05)
    assert last_error is not None
    raise last_error


def _extract_wait_seconds(message: str) -> float | None:
    match = re.search(r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s+seconds", message, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _invoke_with_transient_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = 2,
    base_delay_seconds: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if not _is_retryable_transient_error(exc) or attempt >= max_attempts:
                raise
            sleep_fn(base_delay_seconds * (2 ** (attempt - 1)))
    raise RuntimeError("Unreachable retry state.")


def _is_retryable_transient_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {429, 500, 502, 503, 504}
    if isinstance(exc, URLError):
        return True
    return isinstance(exc, TimeoutError)


def _render_day_by_day(days: list[Any]) -> str:
    lines: list[str] = []
    for day in days:
        lines.append(f"Day {day.day_index} ({day.date}) - {day.destination}")
        for activity in day.activities:
            lines.append(f"  - {activity.period}: {activity.name}")
    return "\n".join(lines)
