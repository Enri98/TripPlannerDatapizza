"""Integration-style tests for specialist Datapizza agents with mocked tools."""

from __future__ import annotations

from datetime import datetime, timezone

from datapizza.agents import Agent

from tripplanner.contracts import StandardAgentResult
from tripplanner.specialist_agents import GeoAgent, POIAgent, TransportAgent, WeatherAgent


def _result(source: str, data: dict, cache_key: str, confidence: float = 0.8) -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": data,
            "evidence": [
                {
                    "source": source,
                    "title": "mock evidence",
                    "snippet": "mock snippet",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "url": "https://example.com",
                }
            ],
            "confidence": confidence,
            "warnings": [],
            "cache_key": cache_key,
        }
    )


class MockGeoTool:
    def __init__(self) -> None:
        self.calls = []

    def run(self, *, query: str, locale: str, limit: int) -> StandardAgentResult:
        self.calls.append((query, locale, limit))
        return _result(
            "osm",
            {"candidates": [{"place_name": "Rome", "lat": 41.9, "lon": 12.4}]},
            "geo:rome",
        )


class MockWeatherTool:
    def __init__(self) -> None:
        self.calls = []

    def run(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone_name: str,
    ) -> StandardAgentResult:
        self.calls.append((latitude, longitude, start_date, end_date, timezone_name))
        return _result(
            "open-meteo",
            {"daily": [{"date": start_date, "weather_code": 1}]},
            "weather:key",
        )


class MockPOITool:
    def __init__(self) -> None:
        self.calls = []

    def run(self, *, bbox: tuple[float, float, float, float], tags, limit: int, locale: str) -> StandardAgentResult:
        self.calls.append((bbox, tags, limit, locale))
        return _result(
            "osm",
            {"pois": [{"name": "Colosseum", "lat": 41.89, "lon": 12.49}]},
            "poi:key",
        )


class MockSearchTool:
    def __init__(self) -> None:
        self.calls = []

    def run(self, *, query: str, locale: str, limit: int) -> StandardAgentResult:
        self.calls.append((query, locale, limit))
        return _result(
            "web",
            {"results": [{"title": "Result", "snippet": "Snippet", "url": "https://example.com"}]},
            "search:key",
            confidence=0.7,
        )


def test_geo_agent_invocation_returns_standard_result() -> None:
    tool = MockGeoTool()
    agent = GeoAgent(geo_tool=tool)

    assert isinstance(agent.agent, Agent)
    result = agent.invoke({"query": "Rome", "locale": "en", "limit": 3})

    assert isinstance(result, StandardAgentResult)
    assert tool.calls == [("Rome", "en", 3)]
    assert result.evidence[0].source == "osm"


def test_weather_agent_invocation_returns_standard_result() -> None:
    tool = MockWeatherTool()
    agent = WeatherAgent(weather_tool=tool)
    result = agent.invoke(
        {
            "latitude": 41.9,
            "longitude": 12.4,
            "start_date": "2026-03-10",
            "end_date": "2026-03-11",
            "timezone_name": "Europe/Rome",
        }
    )

    assert isinstance(result, StandardAgentResult)
    assert len(tool.calls) == 1
    assert result.evidence[0].source == "open-meteo"


def test_poi_agent_supports_optional_search_enrichment() -> None:
    poi_tool = MockPOITool()
    search_tool = MockSearchTool()
    agent = POIAgent(poi_tool=poi_tool, search_tool=search_tool)
    result = agent.invoke(
        {
            "bbox": [41.8, 12.3, 42.0, 12.7],
            "tags": ["tourism=attraction"],
            "limit": 5,
            "locale": "en",
            "enrichment_query": "best attractions in rome",
        }
    )

    assert isinstance(result, StandardAgentResult)
    assert len(poi_tool.calls) == 1
    assert len(search_tool.calls) == 1
    assert "enrichment" in result.data
    assert "?" not in " ".join(result.warnings)


def test_transport_agent_uses_search_and_no_booking_warnings() -> None:
    search_tool = MockSearchTool()
    agent = TransportAgent(search_tool=search_tool)
    result = agent.invoke(
        {
            "origin": "Rome",
            "destination": "Florence",
            "departure_date": "2026-03-12",
            "locale": "en",
            "limit": 3,
            "mode_preferences": ["train"],
        }
    )

    assert isinstance(result, StandardAgentResult)
    assert len(search_tool.calls) == 1
    assert any("booking" in warning.lower() for warning in result.warnings)
    assert "?" not in " ".join(result.warnings)
