"""Unit tests for SearchTool wrapper."""

from __future__ import annotations

from tripplanner.cache import MemoryCache
from tripplanner.search_tool import DuckDuckGoClient, SearchTool


def _sample_rows() -> list[dict]:
    return [
        {
            "title": "Rome Travel Guide",
            "href": "https://example.com/rome",
            "body": "Top sights, food, and neighborhoods in Rome.",
        },
        {
            "title": "Best Time To Visit Rome",
            "href": "https://example.com/rome-weather",
            "body": "Seasonality, weather patterns, and monthly tips.",
        },
    ]


def test_search_tool_normalizes_results_and_evidence() -> None:
    def provider(query: str) -> list[dict]:
        assert query == "rome itinerary"
        return _sample_rows()

    tool = SearchTool(client=DuckDuckGoClient(search_provider=provider), cache=MemoryCache(max_size=8))
    result = tool.run(query="  Rome   itinerary ", locale="en", limit=5)

    assert result.data["query"] == "rome itinerary"
    assert len(result.data["results"]) == 2
    assert result.data["results"][0]["title"] == "Rome Travel Guide"
    assert result.evidence[0].source == "web"


def test_search_tool_uses_cache_for_repeated_queries() -> None:
    calls = {"count": 0}

    def provider(query: str) -> list[dict]:
        calls["count"] += 1
        return _sample_rows()

    tool = SearchTool(client=DuckDuckGoClient(search_provider=provider), cache=MemoryCache(max_size=8))
    kwargs = {"query": "rome itinerary", "locale": "en", "limit": 5}
    first = tool.run(**kwargs)
    second = tool.run(**kwargs)

    assert calls["count"] == 1
    assert first.cache_key == second.cache_key
    assert first.data == second.data


def test_search_tool_handles_empty_provider_output() -> None:
    def provider(query: str) -> list[dict]:
        return []

    tool = SearchTool(client=DuckDuckGoClient(search_provider=provider), cache=MemoryCache(max_size=8))
    result = tool.run(query="unknown query", locale="en", limit=3)

    assert result.data["results"] == []
    assert result.warnings
    assert result.confidence < 0.5
