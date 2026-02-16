"""Nominatim geocoding client and GeoTool wrapper."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tripplanner.cache import MemoryCache, RateLimiter, make_cache_key
from tripplanner.contracts import StandardAgentResult


Fetcher = Callable[[Request], list[dict[str, Any]]]


class GeoToolError(RuntimeError):
    """Raised when GeoTool cannot execute safely."""


def _default_fetcher(request: Request) -> list[dict[str, Any]]:
    with urlopen(request, timeout=15) as response:  # nosec B310 - fixed trusted Nominatim endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise GeoToolError("Unexpected Nominatim response shape.")
    return payload


class NominatimClient:
    """Minimal Nominatim HTTP client."""

    BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        user_agent: str = "TripPlanner/0.1 (+https://github.com/datapizzaTripPlanner)",
    ) -> None:
        self._fetcher = fetcher or _default_fetcher
        self._user_agent = user_agent

    def search(self, *, query: str, locale: str = "en", limit: int = 5) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": max(1, min(limit, 10)),
            "accept-language": locale,
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": self._user_agent})  # noqa: S310
        return self._fetcher(request)


class GeoTool:
    """Geocoding wrapper returning StandardAgentResult payloads."""

    def __init__(
        self,
        client: NominatimClient | None = None,
        cache: MemoryCache | None = None,
        rate_limiter: RateLimiter | None = None,
        *,
        ttl_seconds: int = 30 * 24 * 60 * 60,
    ) -> None:
        self._client = client or NominatimClient()
        self._cache = cache or MemoryCache(max_size=256)
        self._rate_limiter = rate_limiter or RateLimiter(rate_per_second=1.0, capacity=1.0)
        self._ttl_seconds = ttl_seconds

    def run(self, *, query: str, locale: str = "en", limit: int = 5) -> StandardAgentResult:
        cache_key = make_cache_key("geocode", query, locale, limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return StandardAgentResult.model_validate(cached)

        if not self._rate_limiter.allow():
            wait_seconds = self._rate_limiter.wait_time()
            raise GeoToolError(
                f"Nominatim rate limit exceeded. Retry after {wait_seconds:.2f} seconds."
            )

        rows = self._client.search(query=query, locale=locale, limit=limit)
        normalized = self._normalize(rows=rows, query=query, locale=locale, cache_key=cache_key)
        self._cache.set(cache_key, normalized.model_dump(mode="json"), ttl_seconds=self._ttl_seconds)
        return normalized

    def _normalize(
        self,
        *,
        rows: list[dict[str, Any]],
        query: str,
        locale: str,
        cache_key: str,
    ) -> StandardAgentResult:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            lat = _to_float(row.get("lat"))
            lon = _to_float(row.get("lon"))
            bbox = _normalize_bbox(row.get("boundingbox"))
            if lat is None or lon is None or bbox is None:
                continue

            address = row.get("address") or {}
            country_code = str(address.get("country_code") or "").lower()
            place_name = str(row.get("display_name") or "")
            confidence = _confidence_from_row(row, query)
            candidates.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "bbox": bbox,
                    "place_name": place_name,
                    "country_code": country_code,
                    "confidence": confidence,
                }
            )

        candidates.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        selected = candidates[0] if candidates else None
        warnings = [] if candidates else ["No geocoding candidates returned by Nominatim."]
        confidence = float(selected["confidence"]) if selected else 0.0

        return StandardAgentResult.model_validate(
            {
                "data": {
                    "query": query,
                    "locale": locale,
                    "candidates": candidates,
                    "selected": selected,
                },
                "evidence": [
                    {
                        "source": "osm",
                        "title": f"Nominatim geocoding for '{query}'",
                        "snippet": f"Returned {len(candidates)} candidate(s) for locale '{locale}'.",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "url": "https://nominatim.openstreetmap.org/",
                    }
                ],
                "confidence": confidence,
                "warnings": warnings,
                "cache_key": cache_key,
            }
        )


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_bbox(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    # Nominatim returns [south, north, west, east] as strings.
    south = _to_float(raw_bbox[0])
    north = _to_float(raw_bbox[1])
    west = _to_float(raw_bbox[2])
    east = _to_float(raw_bbox[3])
    if None in {south, north, west, east}:
        return None
    return [south, west, north, east]  # [south, west, north, east]


def _confidence_from_row(row: dict[str, Any], query: str) -> float:
    importance = _to_float(row.get("importance")) or 0.0
    score = max(0.0, min(1.0, importance))
    display_name = str(row.get("display_name") or "").lower()
    normalized_query = " ".join(query.lower().split())
    if normalized_query and normalized_query in display_name:
        score = min(1.0, score + 0.1)
    return round(score, 3)
