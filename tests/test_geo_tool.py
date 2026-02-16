"""Unit tests for Nominatim client and GeoTool wrapper."""

from __future__ import annotations

from tripplanner.cache import MemoryCache, RateLimiter
from tripplanner.geo_tool import GeoTool, GeoToolError, NominatimClient


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _sample_rows() -> list[dict]:
    return [
        {
            "lat": "41.8933",
            "lon": "12.4829",
            "boundingbox": ["41.80", "42.00", "12.30", "12.70"],
            "display_name": "Rome, Roma Capitale, Lazio, Italia",
            "importance": 0.78,
            "address": {"country_code": "it"},
        },
        {
            "lat": "40.4168",
            "lon": "-3.7038",
            "boundingbox": ["40.31", "40.58", "-3.89", "-3.51"],
            "display_name": "Madrid, Comunidad de Madrid, España",
            "importance": 0.71,
            "address": {"country_code": "es"},
        },
    ]


def test_geo_tool_parses_candidates_and_evidence() -> None:
    def fetcher(request):  # type: ignore[no-untyped-def]
        return _sample_rows()

    tool = GeoTool(client=NominatimClient(fetcher=fetcher), cache=MemoryCache(max_size=8))
    result = tool.run(query="Rome", locale="en", limit=5)

    assert result.evidence[0].source == "osm"
    assert result.data["query"] == "Rome"
    assert len(result.data["candidates"]) == 2
    first = result.data["candidates"][0]
    assert "lat" in first and "lon" in first and "bbox" in first
    assert first["country_code"] in {"it", "es"}


def test_geo_tool_uses_cache_on_repeated_query() -> None:
    calls = {"count": 0}

    def fetcher(request):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return _sample_rows()

    cache = MemoryCache(max_size=8)
    tool = GeoTool(client=NominatimClient(fetcher=fetcher), cache=cache)

    first = tool.run(query="Rome", locale="en", limit=5)
    second = tool.run(query="Rome", locale="en", limit=5)

    assert calls["count"] == 1
    assert first.cache_key == second.cache_key
    assert first.data == second.data


def test_geo_tool_rate_limit_blocks_burst_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate_per_second=1.0, capacity=1.0, clock=clock)

    def fetcher(request):  # type: ignore[no-untyped-def]
        return _sample_rows()

    tool = GeoTool(
        client=NominatimClient(fetcher=fetcher),
        cache=MemoryCache(max_size=8),
        rate_limiter=limiter,
    )

    first = tool.run(query="Rome", locale="en", limit=5)
    assert first.data["selected"] is not None

    try:
        tool.run(query="Milan", locale="en", limit=5)
        assert False, "Expected GeoToolError due to rate limiting"
    except GeoToolError as exc:
        assert "rate limit" in str(exc).lower()

    clock.advance(1.0)
    third = tool.run(query="Milan", locale="en", limit=5)
    assert third.data["selected"] is not None
