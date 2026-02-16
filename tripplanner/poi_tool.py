"""Overpass POI client and POITool wrapper."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable
from urllib.request import Request, urlopen

from tripplanner.cache import MemoryCache, RateLimiter, hash_bbox, make_cache_key, normalize_text
from tripplanner.contracts import StandardAgentResult


Fetcher = Callable[[str], dict[str, Any]]


class POIToolError(RuntimeError):
    """Raised when POI tool requests cannot be executed safely."""


def _default_fetcher(query: str) -> dict[str, Any]:
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=query.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:  # nosec B310 - fixed trusted Overpass endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise POIToolError("Unexpected Overpass response shape.")
    return payload


class OverpassClient:
    """Minimal Overpass HTTP client."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher or _default_fetcher

    def search_pois(self, query: str) -> dict[str, Any]:
        return self._fetcher(query)


class POITool:
    """POI wrapper returning StandardAgentResult payloads."""

    _DEFAULT_TAGS = ["tourism=attraction", "tourism=museum", "amenity=restaurant"]

    def __init__(
        self,
        client: OverpassClient | None = None,
        cache: MemoryCache | None = None,
        rate_limiter: RateLimiter | None = None,
        *,
        ttl_seconds: int = 24 * 60 * 60,
    ) -> None:
        self._client = client or OverpassClient()
        self._cache = cache or MemoryCache(max_size=256)
        self._rate_limiter = rate_limiter or RateLimiter(rate_per_second=0.5, capacity=1.0)
        self._ttl_seconds = ttl_seconds

    def run(
        self,
        *,
        bbox: tuple[float, float, float, float],
        tags: list[str] | None = None,
        limit: int = 20,
        locale: str = "en",
    ) -> StandardAgentResult:
        normalized_tags = [normalize_text(t) for t in (tags or self._DEFAULT_TAGS)]
        capped_limit = max(1, min(100, limit))
        bbox_token = hash_bbox(bbox)
        cache_key = make_cache_key("poi", bbox_token, normalized_tags, capped_limit, locale)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return StandardAgentResult.model_validate(cached)

        if not self._rate_limiter.allow():
            wait_seconds = self._rate_limiter.wait_time()
            raise POIToolError(f"Overpass throttled. Retry after {wait_seconds:.2f} seconds.")

        query = _build_overpass_query(bbox=bbox, tags=normalized_tags, limit=capped_limit)
        payload = self._client.search_pois(query)
        normalized = self._normalize(payload=payload, tags=normalized_tags, limit=capped_limit, cache_key=cache_key)
        self._cache.set(cache_key, normalized.model_dump(mode="json"), ttl_seconds=self._ttl_seconds)
        return normalized

    def _normalize(
        self,
        *,
        payload: dict[str, Any],
        tags: list[str],
        limit: int,
        cache_key: str,
    ) -> StandardAgentResult:
        elements = payload.get("elements", [])
        pois: list[dict[str, Any]] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            coords = _extract_coords(element)
            if coords is None:
                continue
            name = str((element.get("tags") or {}).get("name") or "")
            element_type = str(element.get("type") or "unknown")
            poi_tags = dict(element.get("tags") or {})
            pois.append(
                {
                    "name": name,
                    "type": element_type,
                    "lat": coords[0],
                    "lon": coords[1],
                    "tags": poi_tags,
                }
            )
            if len(pois) >= limit:
                break

        warnings = [] if pois else ["No POIs returned by Overpass for requested area/tags."]
        confidence = 0.8 if pois else 0.2

        return StandardAgentResult.model_validate(
            {
                "data": {
                    "pois": pois,
                    "requested_tags": tags,
                    "returned_count": len(pois),
                },
                "evidence": [
                    {
                        "source": "osm",
                        "title": "Overpass POI search",
                        "snippet": f"Returned {len(pois)} POI(s) for tags: {', '.join(tags)}",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "url": "https://overpass-api.de/",
                    }
                ],
                "confidence": confidence,
                "warnings": warnings,
                "cache_key": cache_key,
            }
        )


def _build_overpass_query(
    *,
    bbox: tuple[float, float, float, float],
    tags: list[str],
    limit: int,
) -> str:
    south, west, north, east = bbox
    selectors: list[str] = []
    for tag in tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
            key = normalize_text(key)
            value = normalize_text(value)
        else:
            key, value = "tourism", normalize_text(tag)
        if not key or not value:
            continue
        selectors.extend(
            [
                f'node["{key}"="{value}"]({south},{west},{north},{east});',
                f'way["{key}"="{value}"]({south},{west},{north},{east});',
                f'relation["{key}"="{value}"]({south},{west},{north},{east});',
            ]
        )
    if not selectors:
        raise POIToolError("No valid tags provided for Overpass query.")

    return (
        "[out:json][timeout:25];\n"
        "(\n"
        + "\n".join(selectors)
        + "\n);\n"
        f"out center {limit};"
    )


def _extract_coords(element: dict[str, Any]) -> tuple[float, float] | None:
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    center = element.get("center")
    if isinstance(center, dict):
        c_lat = center.get("lat")
        c_lon = center.get("lon")
        if c_lat is not None and c_lon is not None:
            try:
                return float(c_lat), float(c_lon)
            except (TypeError, ValueError):
                return None
    return None
