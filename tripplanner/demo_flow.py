"""End-to-end demo orchestration flow for CLI usage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
import re
from typing import Any

from tripplanner.cache import MemoryCache, RateLimiter
from tripplanner.contracts import PlanTask, StandardAgentResult, TripSpec
from tripplanner.executor import OrchestratorExecutor
from tripplanner.geo_tool import GeoTool, NominatimClient
from tripplanner.guardrails import parse_date_expression, resolve_weekend_range
from tripplanner.itinerary_synth import ItinerarySynthesizer
from tripplanner.planner import OrchestratorPlanner
from tripplanner.poi_tool import OverpassClient, POITool
from tripplanner.search_tool import DuckDuckGoClient, SearchTool
from tripplanner.telemetry import start_span
from tripplanner.weather_tool import OpenMeteoClient, WeatherTool


AMBIGUOUS_DESTINATIONS = {
    "spain",
    "italy",
    "france",
    "germany",
    "portugal",
    "greece",
    "japan",
    "usa",
    "united states",
}


@dataclass
class DemoFlow:
    """Composable orchestrator for demo CLI."""

    offline: bool = False

    def run(
        self,
        query: str,
        *,
        now_ts: datetime | None = None,
        timezone_name: str = "UTC",
        output_language: str | None = None,
    ) -> dict[str, Any]:
        request_now = now_ts or datetime.now(timezone.utc)

        with start_span("orchestrator.extract_tripspec") as span:
            if span is not None:
                span.set_attribute("demo.query_length", len(query))
            extracted = self._extract_tripspec(
                query=query,
                now_ts=request_now,
                timezone_name=timezone_name,
                output_language=output_language,
            )
            if extracted["status"] == "clarification_needed":
                if span is not None:
                    span.set_attribute("demo.status", "clarification_needed")
                return extracted
            tripspec = TripSpec.model_validate(extracted["tripspec"])
            if span is not None:
                span.set_attribute("demo.legs", len(tripspec.legs))

        with start_span("orchestrator.plan"):
            plan = OrchestratorPlanner().generate(tripspec)

        handlers = self._build_handlers(tripspec)
        executor = OrchestratorExecutor(handlers=handlers)
        outcome = executor.execute(plan)

        with start_span("orchestrator.evaluate") as span:
            if span is not None:
                span.set_attribute("demo.execution_status", outcome.status)
                span.set_attribute("demo.completed_tasks", len(outcome.results))
            if outcome.status == "clarification_needed":
                return {
                    "status": "clarification_needed",
                    "clarifying_question": outcome.clarifying_question,
                    "stages": outcome.stages,
                }

        with start_span("orchestrator.synthesize"):
            itinerary = ItinerarySynthesizer().synthesize(
                tripspec=tripspec,
                results=outcome.results,
            )

        return {
            "status": "completed",
            "language": itinerary.language,
            "title": itinerary.title,
            "days": [day.model_dump(mode="json") for day in itinerary.days],
            "warnings": itinerary.warnings,
            "stages": outcome.stages,
            "itinerary_text": _render_day_by_day(itinerary.days),
        }

    def _extract_tripspec(
        self,
        *,
        query: str,
        now_ts: datetime,
        timezone_name: str,
        output_language: str | None,
    ) -> dict[str, Any]:
        destination_candidates = _extract_destinations(query)
        if not destination_candidates:
            return {
                "status": "clarification_needed",
                "clarifying_question": "Which city or region should I plan for?",
            }

        for destination in destination_candidates:
            if destination.lower() in AMBIGUOUS_DESTINATIONS:
                return {
                    "status": "clarification_needed",
                    "clarifying_question": (
                        "Please specify a city or region in that country so I can plan the itinerary."
                    ),
                }

        date_range = _extract_date_range(query, now_ts=now_ts, timezone_name=timezone_name)
        if date_range is None:
            return {
                "status": "clarification_needed",
                "clarifying_question": "What are your exact travel dates?",
            }
        start_date, end_date = date_range
        legs = _split_dates_into_legs(start_date, end_date, destination_candidates)
        if not legs:
            return {
                "status": "clarification_needed",
                "clarifying_question": "I could not map dates to destinations. Please clarify your itinerary.",
            }

        lang = _infer_language(query, output_language)
        tripspec = {
            "request_context": {
                "now_ts": now_ts.isoformat(),
                "timezone": timezone_name,
                "input_language": lang,
                "output_language": lang,
            },
            "budget": {
                "amount": _extract_budget_amount(query),
                "currency": _extract_budget_currency(query),
                "scope": "total",
                "num_travelers": _extract_num_travelers(query),
            },
            "legs": legs,
            "preferences": {"tags": _extract_tags(query)},
            "constraints": {"pace": "standard", "mobility": "public_transport"},
        }
        return {"status": "ready", "tripspec": tripspec}

    def _build_handlers(self, tripspec: TripSpec) -> dict[str, Any]:
        cache = MemoryCache(max_size=512)
        geo_tool = GeoTool(
            client=NominatimClient(),
            cache=cache,
            rate_limiter=RateLimiter(rate_per_second=1.0, capacity=1.0),
        )
        weather_tool = WeatherTool(client=OpenMeteoClient(), cache=cache)
        poi_tool = POITool(
            client=OverpassClient(),
            cache=cache,
            rate_limiter=RateLimiter(rate_per_second=0.5, capacity=1.0),
        )
        search_tool = SearchTool(client=DuckDuckGoClient(), cache=cache)

        runtime_results: dict[str, StandardAgentResult] = {}

        def leg_index_for(task: PlanTask) -> int:
            match = re.search(r"legs\[(\d+)\]", task.input_ref)
            if not match:
                raise ValueError(f"Cannot resolve leg index from input_ref={task.input_ref}")
            return int(match.group(1))

        def geo_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.geo") as span:
                leg_index = leg_index_for(task)
                leg = tripspec.legs[leg_index]
                try:
                    result = (
                        _fallback_geo_result(leg.destination_text)
                        if self.offline
                        else geo_tool.run(query=leg.destination_text, locale=tripspec.request_context.output_language)
                    )
                except Exception:
                    result = _fallback_geo_result(leg.destination_text)
                if span is not None:
                    span.set_attribute("demo.cache_key", result.cache_key)
                runtime_results[task.task_id] = result
                return result

        def weather_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.weather") as span:
                leg_index = leg_index_for(task)
                leg = tripspec.legs[leg_index]
                lat = leg.geo.lat if leg.geo is not None else None
                lon = leg.geo.lon if leg.geo is not None else None
                if lat is None or lon is None:
                    geo_result = runtime_results.get(f"geo_leg_{leg_index}")
                    selected = (geo_result.data if geo_result else {}).get("selected", {})
                    if isinstance(selected, dict) and selected.get("lat") is not None and selected.get("lon") is not None:
                        lat = float(selected["lat"])
                        lon = float(selected["lon"])
                try:
                    if self.offline or lat is None or lon is None:
                        result = _fallback_weather_result(leg.date_range.start_date, leg.date_range.end_date)
                    else:
                        result = weather_tool.run(
                            latitude=lat,
                            longitude=lon,
                            start_date=leg.date_range.start_date.isoformat(),
                            end_date=leg.date_range.end_date.isoformat(),
                            timezone_name=tripspec.request_context.timezone,
                        )
                except Exception:
                    result = _fallback_weather_result(leg.date_range.start_date, leg.date_range.end_date)
                if span is not None:
                    span.set_attribute("demo.cache_key", result.cache_key)
                runtime_results[task.task_id] = result
                return result

        def poi_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.poi") as span:
                leg_index = leg_index_for(task)
                leg = tripspec.legs[leg_index]
                geo = leg.geo
                bbox = geo.bbox if geo is not None else None
                if bbox is None:
                    geo_result = runtime_results.get(f"geo_leg_{leg_index}")
                    selected = (geo_result.data if geo_result else {}).get("selected", {})
                    candidate_bbox = selected.get("bbox") if isinstance(selected, dict) else None
                    if isinstance(candidate_bbox, list) and len(candidate_bbox) == 4:
                        bbox = [float(v) for v in candidate_bbox]
                try:
                    if self.offline or bbox is None:
                        result = _fallback_poi_result(leg.destination_text)
                    else:
                        result = poi_tool.run(
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            tags=["tourism=attraction", "tourism=museum"],
                            limit=8,
                            locale=tripspec.request_context.output_language,
                        )
                except Exception:
                    result = _fallback_poi_result(leg.destination_text)
                if span is not None:
                    span.set_attribute("demo.cache_key", result.cache_key)
                runtime_results[task.task_id] = result
                return result

        def transport_handler(task: PlanTask) -> StandardAgentResult:
            with start_span("agent.transport") as span:
                leg_match = re.search(r"legs\[(\d+)\]->legs\[(\d+)\]", task.input_ref)
                if not leg_match:
                    result = _fallback_transport_result("origin", "destination")
                else:
                    from_idx = int(leg_match.group(1))
                    to_idx = int(leg_match.group(2))
                    origin = tripspec.legs[from_idx].destination_text
                    destination = tripspec.legs[to_idx].destination_text
                    departure_date = tripspec.legs[to_idx].date_range.start_date.isoformat()
                    try:
                        if self.offline:
                            result = _fallback_transport_result(origin, destination)
                        else:
                            result = search_tool.run(
                                query=f"{origin} to {destination} transport options {departure_date}",
                                locale=tripspec.request_context.output_language,
                                limit=3,
                            )
                    except Exception:
                        result = _fallback_transport_result(origin, destination)
                if span is not None:
                    span.set_attribute("demo.cache_key", result.cache_key)
                runtime_results[task.task_id] = result
                return result

        def synth_handler(task: PlanTask) -> StandardAgentResult:
            result = _synth_marker_result(task.task_id)
            runtime_results[task.task_id] = result
            return result

        return {
            "geo": geo_handler,
            "weather": weather_handler,
            "poi": poi_handler,
            "transport": transport_handler,
            "synth": synth_handler,
        }


def run_demo_flow(
    query: str,
    *,
    now_ts: datetime | None = None,
    timezone_name: str = "UTC",
    output_language: str | None = None,
) -> dict[str, Any]:
    offline = os.getenv("TRIPPLANNER_DEMO_OFFLINE", "0").strip().lower() in {"1", "true", "yes", "on"}
    flow = DemoFlow(offline=offline)
    return flow.run(
        query,
        now_ts=now_ts,
        timezone_name=timezone_name,
        output_language=output_language,
    )


def _extract_destinations(query: str) -> list[str]:
    lowered = query.lower()
    match = re.search(r"\bto\s+(.+)", lowered)
    if match:
        segment = query[match.start(1):]
    else:
        segment = query
    segment = re.split(r"\b(next weekend|this weekend|from|on|for \d+ day)", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"\b(plan|trip|travel|visit|days?|day|a|an|the|please)\b", " ", segment, flags=re.IGNORECASE)
    parts = re.split(r",| and | then | -> ", cleaned, flags=re.IGNORECASE)
    destinations = []
    for part in parts:
        text = " ".join(part.strip().split())
        if not text:
            continue
        if len(text) <= 2:
            continue
        destinations.append(text.title())
    seen: set[str] = set()
    unique = []
    for destination in destinations:
        key = destination.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(destination)
    return unique


def _extract_date_range(query: str, *, now_ts: datetime, timezone_name: str) -> tuple[date, date] | None:
    normalized = " ".join(query.lower().split())
    if "next weekend" in normalized:
        start, end = resolve_weekend_range("next weekend", now_ts, timezone_name)
    elif "this weekend" in normalized:
        start, end = resolve_weekend_range("this weekend", now_ts, timezone_name)
    else:
        range_match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(to|-)\s*(\d{4}-\d{2}-\d{2})", query)
        if range_match:
            start = parse_date_expression(range_match.group(1), now_ts, timezone_name)
            end = parse_date_expression(range_match.group(3), now_ts, timezone_name)
        else:
            single_match = re.search(
                r"\b(on\s+)?([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})\b",
                query,
            )
            if single_match:
                start = parse_date_expression(single_match.group(2), now_ts, timezone_name)
                end = start
            else:
                return None

    duration_match = re.search(r"\b(\d+)\s*-\s*day\b|\b(\d+)\s+day\b", normalized)
    if duration_match:
        days_raw = duration_match.group(1) or duration_match.group(2)
        trip_days = max(1, int(days_raw))
        end = start + timedelta(days=trip_days - 1)
    return start, end


def _split_dates_into_legs(start_date: date, end_date: date, destinations: list[str]) -> list[dict[str, Any]]:
    total_days = (end_date - start_date).days + 1
    if total_days < len(destinations):
        return []

    base_days = total_days // len(destinations)
    remainder = total_days % len(destinations)
    legs: list[dict[str, Any]] = []
    cursor = start_date
    for idx, destination in enumerate(destinations):
        days_for_leg = base_days + (1 if idx < remainder else 0)
        leg_end = cursor + timedelta(days=days_for_leg - 1)
        legs.append(
            {
                "destination_text": destination,
                "date_range": {
                    "start_date": cursor.isoformat(),
                    "end_date": leg_end.isoformat(),
                },
            }
        )
        cursor = leg_end + timedelta(days=1)
    return legs


def _extract_budget_amount(query: str) -> float:
    money = re.search(r"([€$])\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(eur|usd)", query, flags=re.IGNORECASE)
    if not money:
        return 1500.0
    if money.group(2):
        return float(money.group(2))
    return float(money.group(3))


def _extract_budget_currency(query: str) -> str:
    lowered = query.lower()
    if "€" in query or "eur" in lowered:
        return "EUR"
    if "$" in query or "usd" in lowered:
        return "USD"
    return "EUR"


def _extract_num_travelers(query: str) -> int | None:
    match = re.search(r"\bfor\s+(\d+)\s+(people|travelers|travellers|persons)\b", query, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _extract_tags(query: str) -> list[str]:
    lowered = query.lower()
    tags = []
    for token in ("museum", "food", "hiking", "beach", "nightlife"):
        if token in lowered:
            tags.append(token)
    return tags


def _infer_language(query: str, forced: str | None) -> str:
    if forced in {"en", "it"}:
        return forced
    italian_markers = {"ciao", "viaggio", "giorni", "itinerario", "musei"}
    lowered = query.lower()
    return "it" if any(marker in lowered for marker in italian_markers) else "en"


def _fallback_geo_result(destination: str) -> StandardAgentResult:
    lat = round(10 + (abs(hash(destination)) % 7000) / 100, 4)
    lon = round(5 + (abs(hash(destination[::-1])) % 7000) / 100, 4)
    return StandardAgentResult.model_validate(
        {
            "data": {
                "query": destination,
                "candidates": [],
                "selected": {
                    "lat": lat,
                    "lon": lon,
                    "bbox": [lat - 0.1, lon - 0.1, lat + 0.1, lon + 0.1],
                    "place_name": destination,
                    "country_code": "xx",
                },
            },
            "evidence": [
                {
                    "source": "demo",
                    "title": f"Fallback geocoding for {destination}",
                    "snippet": "Generated fallback coordinates because online geocoder was unavailable.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 0.65,
            "warnings": ["Using fallback geocoding; verify exact location manually."],
            "cache_key": f"demo:geo:{destination.lower()}",
        }
    )


def _fallback_weather_result(start_date: date, end_date: date) -> StandardAgentResult:
    days = []
    cursor = start_date
    while cursor <= end_date:
        days.append(
            {
                "date": cursor.isoformat(),
                "temp_min_c": 12,
                "temp_max_c": 20,
                "precipitation_probability_max": 35,
                "weather_code": 2,
            }
        )
        cursor += timedelta(days=1)
    return StandardAgentResult.model_validate(
        {
            "data": {"daily": days},
            "evidence": [
                {
                    "source": "demo",
                    "title": "Fallback weather data",
                    "snippet": "Generated heuristic weather due to unavailable weather service.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 0.6,
            "warnings": ["Using fallback weather estimates; check local forecasts closer to departure."],
            "cache_key": f"demo:weather:{start_date.isoformat()}:{end_date.isoformat()}",
        }
    )


def _fallback_poi_result(destination: str) -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": {
                "pois": [
                    {"name": f"{destination} Old Town Walk", "tags": {"tourism": "attraction"}},
                    {"name": f"{destination} City Museum", "tags": {"tourism": "museum"}},
                ]
            },
            "evidence": [
                {
                    "source": "demo",
                    "title": f"Fallback POIs for {destination}",
                    "snippet": "Generated fallback activity candidates due to unavailable POI service.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 0.62,
            "warnings": ["Using fallback POIs; validate opening hours independently."],
            "cache_key": f"demo:poi:{destination.lower()}",
        }
    )


def _fallback_transport_result(origin: str, destination: str) -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": {
                "options": [
                    {"title": f"Train {origin} to {destination} (approx. 2h 30m)"},
                    {"title": f"Bus {origin} to {destination} (approx. 3h 45m)"},
                ]
            },
            "evidence": [
                {
                    "source": "demo",
                    "title": f"Fallback transport options {origin} -> {destination}",
                    "snippet": "Generated fallback transport guidance due to unavailable search service.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 0.61,
            "warnings": ["Transport times are indicative and may change."],
            "cache_key": f"demo:transport:{origin.lower()}:{destination.lower()}",
        }
    )


def _synth_marker_result(task_id: str) -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": {"task_id": task_id, "status": "synthesized"},
            "evidence": [
                {
                    "source": "demo",
                    "title": "Synthesis stage marker",
                    "snippet": "Synthesis stage completed.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 1.0,
            "warnings": [],
            "cache_key": f"demo:synth:{task_id}",
        }
    )


def _render_day_by_day(days: list[Any]) -> str:
    lines: list[str] = []
    for day in days:
        lines.append(f"Day {day.day_index} ({day.date}) - {day.destination}")
        for activity in day.activities:
            lines.append(f"  - {activity.period}: {activity.name}")
    return "\n".join(lines)
