"""Unit tests for Overpass POI client and POITool wrapper."""

from __future__ import annotations

from tripplanner.cache import MemoryCache, RateLimiter
from tripplanner.poi_tool import OverpassClient, POITool, POIToolError


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _sample_payload() -> dict:
    return {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 41.9005,
                "lon": 12.4833,
                "tags": {"name": "Trevi Fountain", "tourism": "attraction"},
            },
            {
                "type": "way",
                "id": 2,
                "center": {"lat": 41.8986, "lon": 12.4768},
                "tags": {"name": "Pantheon", "historic": "monument"},
            },
        ]
    }


def test_poi_tool_builds_query_and_parses_payload() -> None:
    captured = {"query": ""}

    def fetcher(query: str) -> dict:
        captured["query"] = query
        return _sample_payload()

    tool = POITool(client=OverpassClient(fetcher=fetcher), cache=MemoryCache(max_size=8))
    result = tool.run(
        bbox=(41.80, 12.30, 42.00, 12.70),
        tags=["tourism=attraction", "historic=monument"],
        limit=10,
        locale="en",
    )

    assert 'node["tourism"="attraction"]' in captured["query"]
    assert "out center 10;" in captured["query"]
    assert result.evidence[0].source == "osm"
    assert result.data["returned_count"] == 2
    assert result.data["pois"][0]["name"] == "Trevi Fountain"


def test_poi_tool_uses_cache_on_repeated_calls() -> None:
    calls = {"count": 0}

    def fetcher(query: str) -> dict:
        calls["count"] += 1
        return _sample_payload()

    tool = POITool(client=OverpassClient(fetcher=fetcher), cache=MemoryCache(max_size=8))
    kwargs = {
        "bbox": (41.80, 12.30, 42.00, 12.70),
        "tags": ["tourism=attraction"],
        "limit": 10,
        "locale": "en",
    }
    first = tool.run(**kwargs)
    second = tool.run(**kwargs)

    assert calls["count"] == 1
    assert first.cache_key == second.cache_key
    assert first.data == second.data


def test_poi_tool_rate_limiter_blocks_burst_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate_per_second=0.5, capacity=1.0, clock=clock)

    def fetcher(query: str) -> dict:
        return _sample_payload()

    tool = POITool(
        client=OverpassClient(fetcher=fetcher),
        cache=MemoryCache(max_size=8),
        rate_limiter=limiter,
    )

    first = tool.run(bbox=(41.80, 12.30, 42.00, 12.70), tags=["tourism=attraction"])
    assert first.data["returned_count"] > 0

    try:
        tool.run(bbox=(41.80, 12.30, 42.00, 12.70), tags=["amenity=restaurant"])
        assert False, "Expected POIToolError due to throttling"
    except POIToolError as exc:
        assert "throttled" in str(exc).lower()

    clock.advance(2.0)
    third = tool.run(bbox=(41.80, 12.30, 42.00, 12.70), tags=["amenity=restaurant"])
    assert third.data["returned_count"] > 0
