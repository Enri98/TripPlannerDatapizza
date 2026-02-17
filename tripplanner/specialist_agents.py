"""Datapizza specialist agents wrapping project tools with JSON-only outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from datapizza.agents import Agent
from datapizza.agents.logger import AgentLogger
from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.tools import Tool
from datapizza.type import FunctionCallBlock, FunctionCallResultBlock
from pydantic import BaseModel, Field

from tripplanner.contracts import StandardAgentResult
from tripplanner.geo_tool import GeoTool
from tripplanner.poi_tool import POITool
from tripplanner.search_tool import SearchTool
from tripplanner.weather_tool import WeatherTool


class _DeterministicToolClient(MockClient):
    """Mock client that always issues a deterministic call to the first tool."""

    def __init__(self) -> None:
        super().__init__(model_name="deterministic_tool_client", system_prompt="")
        self._next_arguments: dict[str, Any] | None = None

    def set_next_arguments(self, arguments: dict[str, Any]) -> None:
        self._next_arguments = arguments

    def _invoke(  # type: ignore[override]
        self,
        input: list[Any],
        tools: list[Tool] | None = None,
        memory=None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ClientResponse:
        if memory and isinstance(memory[-1].blocks[-1], FunctionCallResultBlock):
            return super()._invoke(
                input=input,
                tools=tools,
                memory=memory,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                **kwargs,
            )

        if not tools:
            return super()._invoke(
                input=input,
                tools=tools,
                memory=memory,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                **kwargs,
            )

        arguments = self._next_arguments or {}
        self._next_arguments = None
        return ClientResponse(
            content=[
                FunctionCallBlock(
                    id="specialist_call_1",
                    arguments=arguments,
                    name=tools[0].name,
                    tool=tools[0],
                )
            ]
        )


class _BaseSpecialistAgent:
    """Shared Datapizza-agent wrapper for deterministic tool invocation."""

    def __init__(self, name: str, system_prompt: str) -> None:
        self._client = _DeterministicToolClient()
        tool = Tool(
            func=self._invoke_tool,
            name=f"{name.lower()}_tool",
            description=f"Execute {name} specialist tool call.",
        )
        self.agent = Agent(
            name=name,
            client=self._client,
            system_prompt=system_prompt,
            tools=[tool],
            logger=_QuietAgentLogger(name),
            max_steps=2,
            terminate_on_text=True,
            stateless=True,
        )

    def _invoke_tool(self, payload_json: str) -> str:
        payload = json.loads(payload_json)
        result = self._execute(payload)
        return result.model_dump_json(ensure_ascii=True)

    def invoke(self, payload: dict[str, Any] | BaseModel) -> StandardAgentResult:
        payload_dict = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        self._client.set_next_arguments({"payload_json": json.dumps(payload_dict, ensure_ascii=True)})
        step = self.agent.run("function", tool_choice="required_first")
        if step is None or not step.text:
            raise RuntimeError(f"{self.agent.name} produced no output.")
        result = StandardAgentResult.model_validate_json(step.text)
        _assert_no_question_marks(result)
        return result

    def _execute(self, payload: dict[str, Any]) -> StandardAgentResult:
        raise NotImplementedError


class GeoAgentInput(BaseModel):
    query: str
    locale: str = "en"
    limit: int = Field(default=5, ge=1, le=10)


class WeatherAgentInput(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    timezone_name: str


class POIAgentInput(BaseModel):
    bbox: tuple[float, float, float, float]
    tags: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    locale: str = "en"
    enrichment_query: str | None = None


class TransportAgentInput(BaseModel):
    origin: str
    destination: str
    departure_date: str | None = None
    locale: str = "en"
    limit: int = Field(default=5, ge=1, le=10)
    mode_preferences: list[str] = Field(default_factory=list)


class GeoAgent(_BaseSpecialistAgent):
    def __init__(self, geo_tool: GeoTool) -> None:
        self._geo_tool = geo_tool
        super().__init__(
            name="GeoAgent",
            system_prompt="You are a geocoding specialist. Return JSON-only outputs.",
        )

    def _execute(self, payload: dict[str, Any]) -> StandardAgentResult:
        data = GeoAgentInput.model_validate(payload)
        return self._geo_tool.run(query=data.query, locale=data.locale, limit=data.limit)


class WeatherAgent(_BaseSpecialistAgent):
    def __init__(self, weather_tool: WeatherTool) -> None:
        self._weather_tool = weather_tool
        super().__init__(
            name="WeatherAgent",
            system_prompt="You are a weather specialist. Return JSON-only outputs.",
        )

    def _execute(self, payload: dict[str, Any]) -> StandardAgentResult:
        data = WeatherAgentInput.model_validate(payload)
        return self._weather_tool.run(
            latitude=data.latitude,
            longitude=data.longitude,
            start_date=data.start_date,
            end_date=data.end_date,
            timezone_name=data.timezone_name,
        )


class POIAgent(_BaseSpecialistAgent):
    def __init__(self, poi_tool: POITool, search_tool: SearchTool | None = None) -> None:
        self._poi_tool = poi_tool
        self._search_tool = search_tool
        super().__init__(
            name="POIAgent",
            system_prompt="You are a points-of-interest specialist. Return JSON-only outputs.",
        )

    def _execute(self, payload: dict[str, Any]) -> StandardAgentResult:
        data = POIAgentInput.model_validate(payload)
        poi_result = self._poi_tool.run(
            bbox=data.bbox,
            tags=data.tags,
            limit=data.limit,
            locale=data.locale,
        )

        if self._search_tool is None or not data.enrichment_query:
            return poi_result

        enrichment = self._search_tool.run(
            query=data.enrichment_query,
            locale=data.locale,
            limit=min(3, data.limit),
        )
        return StandardAgentResult.model_validate(
            {
                "data": {
                    **poi_result.data,
                    "enrichment": {"web_results": enrichment.data.get("results", [])},
                },
                "evidence": [
                    *[item.model_dump(mode="json") for item in poi_result.evidence],
                    *[item.model_dump(mode="json") for item in enrichment.evidence],
                ],
                "confidence": min(1.0, round((poi_result.confidence + enrichment.confidence) / 2, 3)),
                "warnings": [*poi_result.warnings, *enrichment.warnings],
                "cache_key": poi_result.cache_key,
            }
        )


class TransportAgent(_BaseSpecialistAgent):
    def __init__(self, search_tool: SearchTool) -> None:
        self._search_tool = search_tool
        super().__init__(
            name="TransportAgent",
            system_prompt="You are a transport specialist. Return JSON-only outputs. Never offer booking.",
        )

    def _execute(self, payload: dict[str, Any]) -> StandardAgentResult:
        data = TransportAgentInput.model_validate(payload)
        date_part = f" on {data.departure_date}" if data.departure_date else ""
        modes = ", ".join(data.mode_preferences) if data.mode_preferences else "train bus flight"
        query = (
            f"{data.origin} to {data.destination}{date_part} transport options {modes} "
            "duration frequency no booking"
        )
        search_result = self._search_tool.run(query=query, locale=data.locale, limit=data.limit)
        return StandardAgentResult.model_validate(
            {
                "data": {
                    "origin": data.origin,
                    "destination": data.destination,
                    "departure_date": data.departure_date,
                    "mode_preferences": data.mode_preferences,
                    "options": search_result.data.get("results", []),
                    "heuristics": [
                        "Compare train/bus/flight by duration and transfer complexity.",
                        "Treat timings and prices as indicative, not real-time availability.",
                    ],
                },
                "evidence": [item.model_dump(mode="json") for item in search_result.evidence],
                "confidence": min(0.85, search_result.confidence),
                "warnings": [
                    *search_result.warnings,
                    "No booking or live-price guarantees are provided.",
                ],
                "cache_key": search_result.cache_key,
            }
        )


def _assert_no_question_marks(result: StandardAgentResult) -> None:
    for warning in result.warnings:
        if "?" in warning:
            raise RuntimeError("Specialist agents must not ask user questions.")


class _QuietAgentLogger(AgentLogger):
    """Suppress rich panel output to avoid terminal encoding crashes."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(agent_name)

    def _colored_log(self, log_text: str, *args, **kwargs) -> None:  # type: ignore[override]
        return

    def _log(self, log_text: int, *args, **kwargs) -> None:  # type: ignore[override]
        return

    def log_panel(self, *args, **kwargs) -> None:  # type: ignore[override]
        return
