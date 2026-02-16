"""Open-Meteo client and WeatherTool wrapper."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

from tripplanner.cache import MemoryCache, make_cache_key
from tripplanner.contracts import StandardAgentResult


Fetcher = Callable[[str], dict[str, Any]]


def _default_fetcher(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:  # nosec B310 - fixed trusted Open-Meteo endpoint
        return json.loads(response.read().decode("utf-8"))


class OpenMeteoClient:
    """Minimal Open-Meteo HTTP client for daily weather forecasts."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher or _default_fetcher

    def get_daily_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone_name: str,
    ) -> dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone_name,
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "weather_code",
                ]
            ),
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"
        return self._fetcher(url)


class WeatherTool:
    """Weather wrapper that returns StandardAgentResult with caching."""

    def __init__(
        self,
        client: OpenMeteoClient | None = None,
        cache: MemoryCache | None = None,
        *,
        ttl_seconds: int = 21600,
    ) -> None:
        self._client = client or OpenMeteoClient()
        self._cache = cache or MemoryCache(max_size=256)
        self._ttl_seconds = ttl_seconds

    def run(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone_name: str,
    ) -> StandardAgentResult:
        cache_key = make_cache_key(
            "weather",
            latitude,
            longitude,
            start_date,
            end_date,
            timezone_name,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return StandardAgentResult.model_validate(cached)

        payload = self._client.get_daily_forecast(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            timezone_name=timezone_name,
        )
        normalized = self._normalize_payload(
            payload=payload,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            timezone_name=timezone_name,
            cache_key=cache_key,
        )
        self._cache.set(cache_key, normalized.model_dump(mode="json"), ttl_seconds=self._ttl_seconds)
        return normalized

    def _normalize_payload(
        self,
        *,
        payload: dict[str, Any],
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone_name: str,
        cache_key: str,
    ) -> StandardAgentResult:
        daily = payload.get("daily", {})
        dates: list[str] = daily.get("time", []) or []
        t_min: list[float] = daily.get("temperature_2m_min", []) or []
        t_max: list[float] = daily.get("temperature_2m_max", []) or []
        precip: list[float | None] = daily.get("precipitation_probability_max", []) or []
        weather_code: list[int | None] = daily.get("weather_code", daily.get("weathercode", [])) or []

        rows: list[dict[str, Any]] = []
        for idx, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "temp_min_c": t_min[idx] if idx < len(t_min) else None,
                    "temp_max_c": t_max[idx] if idx < len(t_max) else None,
                    "precipitation_probability_max": precip[idx] if idx < len(precip) else None,
                    "weather_code": weather_code[idx] if idx < len(weather_code) else None,
                }
            )

        summary = {
            "provider": "open-meteo",
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
            "start_date": start_date,
            "end_date": end_date,
            "days": len(rows),
        }
        confidence = 0.85 if rows else 0.25
        warnings = [] if rows else ["No daily weather rows returned by Open-Meteo."]

        return StandardAgentResult.model_validate(
            {
                "data": {
                    "summary": summary,
                    "daily": rows,
                },
                "evidence": [
                    {
                        "source": "open-meteo",
                        "title": f"Open-Meteo daily forecast for {start_date} to {end_date}",
                        "snippet": f"Forecast retrieved for ({latitude}, {longitude}) in timezone {timezone_name}.",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "url": "https://open-meteo.com/",
                    }
                ],
                "confidence": confidence,
                "warnings": warnings,
                "cache_key": cache_key,
            }
        )
