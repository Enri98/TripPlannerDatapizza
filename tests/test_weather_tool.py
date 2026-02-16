"""Unit tests for Open-Meteo client and WeatherTool wrapper."""

from __future__ import annotations

from tripplanner.cache import MemoryCache
from tripplanner.weather_tool import OpenMeteoClient, WeatherTool


def _sample_payload() -> dict:
    return {
        "daily": {
            "time": ["2026-03-10", "2026-03-11"],
            "temperature_2m_min": [12.0, 11.5],
            "temperature_2m_max": [20.0, 19.0],
            "precipitation_probability_max": [10, 25],
            "weather_code": [1, 3],
        }
    }


def test_weather_tool_returns_structured_result() -> None:
    def fetcher(url: str) -> dict:
        assert "api.open-meteo.com" in url
        return _sample_payload()

    tool = WeatherTool(client=OpenMeteoClient(fetcher=fetcher), cache=MemoryCache(max_size=8))
    result = tool.run(
        latitude=41.9028,
        longitude=12.4964,
        start_date="2026-03-10",
        end_date="2026-03-11",
        timezone_name="Europe/Rome",
    )

    assert result.evidence[0].source == "open-meteo"
    assert result.data["summary"]["provider"] == "open-meteo"
    assert len(result.data["daily"]) == 2
    assert result.data["daily"][0]["date"] == "2026-03-10"


def test_weather_tool_uses_cache_on_repeated_calls() -> None:
    calls = {"count": 0}

    def fetcher(url: str) -> dict:
        calls["count"] += 1
        return _sample_payload()

    cache = MemoryCache(max_size=8)
    tool = WeatherTool(client=OpenMeteoClient(fetcher=fetcher), cache=cache)
    kwargs = {
        "latitude": 41.9028,
        "longitude": 12.4964,
        "start_date": "2026-03-10",
        "end_date": "2026-03-11",
        "timezone_name": "Europe/Rome",
    }
    first = tool.run(**kwargs)
    second = tool.run(**kwargs)

    assert calls["count"] == 1
    assert first.cache_key == second.cache_key
    assert first.data == second.data


def test_weather_tool_accepts_legacy_weathercode_key() -> None:
    def fetcher(url: str) -> dict:
        return {
            "daily": {
                "time": ["2026-03-10"],
                "temperature_2m_min": [10.0],
                "temperature_2m_max": [16.0],
                "precipitation_probability_max": [30],
                "weathercode": [61],
            }
        }

    tool = WeatherTool(client=OpenMeteoClient(fetcher=fetcher), cache=MemoryCache(max_size=8))
    result = tool.run(
        latitude=45.4642,
        longitude=9.19,
        start_date="2026-03-10",
        end_date="2026-03-10",
        timezone_name="Europe/Rome",
    )

    assert result.data["daily"][0]["weather_code"] == 61
