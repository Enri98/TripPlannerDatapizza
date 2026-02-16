"""DuckDuckGo web search integration and SearchTool wrapper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from tripplanner.cache import MemoryCache, make_cache_key, normalize_text
from tripplanner.contracts import StandardAgentResult


SearchProvider = Callable[[str], list[dict[str, Any]]]


class SearchToolError(RuntimeError):
    """Raised when search provider cannot be used."""


class DuckDuckGoClient:
    """Adapter for Datapizza DuckDuckGo search tool."""

    def __init__(self, search_provider: SearchProvider | None = None) -> None:
        if search_provider is not None:
            self._search_provider = search_provider
            return

        try:
            from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise SearchToolError(
                "DuckDuckGo search tool is unavailable. Install datapizza-ai-tools-duckduckgo."
            ) from exc

        tool = DuckDuckGoSearchTool()
        self._search_provider = tool.search

    def search(self, query: str) -> list[dict[str, Any]]:
        results = self._search_provider(query)
        if not isinstance(results, list):
            raise SearchToolError("DuckDuckGo provider returned invalid response type.")
        return results


class SearchTool:
    """Web search wrapper returning StandardAgentResult payloads."""

    def __init__(
        self,
        client: DuckDuckGoClient | None = None,
        cache: MemoryCache | None = None,
        *,
        ttl_seconds: int = 12 * 60 * 60,
    ) -> None:
        self._client = client or DuckDuckGoClient()
        self._cache = cache or MemoryCache(max_size=256)
        self._ttl_seconds = ttl_seconds

    def run(self, *, query: str, locale: str = "en", limit: int = 5) -> StandardAgentResult:
        normalized_query = normalize_text(query)
        capped_limit = max(1, min(10, limit))
        cache_key = make_cache_key("search", normalized_query, locale, capped_limit)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return StandardAgentResult.model_validate(cached)

        rows = self._client.search(normalized_query)
        normalized_results = _normalize_search_results(rows, capped_limit)
        warnings = [] if normalized_results else ["No search results returned."]
        confidence = 0.75 if normalized_results else 0.2

        evidence = [
            {
                "source": "web",
                "title": item["title"],
                "snippet": item["snippet"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "url": item["url"] or None,
            }
            for item in normalized_results
        ]

        if not evidence:
            evidence = [
                {
                    "source": "web",
                    "title": f"Web search for '{normalized_query}'",
                    "snippet": "No results returned by provider.",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "url": None,
                }
            ]

        result = StandardAgentResult.model_validate(
            {
                "data": {
                    "query": normalized_query,
                    "locale": locale,
                    "results": normalized_results,
                },
                "evidence": evidence,
                "confidence": confidence,
                "warnings": warnings,
                "cache_key": cache_key,
            }
        )
        self._cache.set(cache_key, result.model_dump(mode="json"), ttl_seconds=self._ttl_seconds)
        return result


def _normalize_search_results(rows: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        url = str(row.get("href") or row.get("url") or "").strip()
        snippet = str(row.get("body") or row.get("snippet") or "").strip()
        if not (title or url or snippet):
            continue
        normalized.append(
            {
                "title": title or "Untitled result",
                "snippet": snippet or "",
                "url": url or "",
            }
        )
        if len(normalized) >= limit:
            break
    return normalized
