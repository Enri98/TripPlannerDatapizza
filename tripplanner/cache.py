"""In-memory cache and rate limiter utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
import time
from typing import Any, Callable, Iterable


Clock = Callable[[], float]


@dataclass
class _CacheItem:
    value: Any
    expires_at: float


class MemoryCache:
    """LRU + TTL in-memory cache."""

    def __init__(self, max_size: int = 256, clock: Clock | None = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self._clock = clock or time.monotonic
        self._items: OrderedDict[str, _CacheItem] = OrderedDict()

    def get(self, key: str, default: Any = None) -> Any:
        item = self._items.get(key)
        if item is None:
            return default
        if item.expires_at <= self._clock():
            self._items.pop(key, None)
            return default
        self._items.move_to_end(key)
        return item.value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            self._items.pop(key, None)
            return

        expires_at = self._clock() + ttl_seconds
        self._items.pop(key, None)
        self._items[key] = _CacheItem(value=value, expires_at=expires_at)
        self._items.move_to_end(key)
        self._evict_if_needed()

    def delete(self, key: str) -> None:
        self._items.pop(key, None)

    def cleanup_expired(self) -> int:
        now = self._clock()
        expired_keys = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired_keys:
            self._items.pop(key, None)
        return len(expired_keys)

    def _evict_if_needed(self) -> None:
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate_per_second: float, capacity: float, clock: Clock | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.rate_per_second = rate_per_second
        self.capacity = capacity
        self._clock = clock or time.monotonic
        self._tokens = capacity
        self._last_refill = self._clock()

    def allow(self, tokens: float = 1.0) -> bool:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: float = 1.0) -> float:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        self._refill()
        if self._tokens >= tokens:
            return 0.0
        missing = tokens - self._tokens
        return missing / self.rate_per_second

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def hash_bbox(bbox: Iterable[float], precision: int = 6) -> str:
    formatted = ",".join(f"{float(v):.{precision}f}" for v in bbox)
    return sha256(formatted.encode("utf-8")).hexdigest()[:16]


def stable_json_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(serialized.encode("utf-8")).hexdigest()[:16]


def make_cache_key(prefix: str, *parts: Any) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            normalized_parts.append(normalize_text(part))
        elif isinstance(part, float):
            normalized_parts.append(str(round(part, 6)))
        elif isinstance(part, (list, tuple, dict)):
            normalized_parts.append(stable_json_hash(part))
        elif part is None:
            normalized_parts.append("none")
        else:
            normalized_parts.append(str(part))
    return ":".join([normalize_text(prefix), *normalized_parts])
