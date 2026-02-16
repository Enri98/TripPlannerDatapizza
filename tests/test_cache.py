"""Unit tests for cache and rate limiter utilities."""

from __future__ import annotations

from tripplanner.cache import MemoryCache, RateLimiter, hash_bbox, make_cache_key, normalize_text


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_memory_cache_ttl_expiry() -> None:
    clock = FakeClock()
    cache = MemoryCache(max_size=4, clock=clock)
    cache.set("weather:rome", {"temp": 20}, ttl_seconds=10)

    assert cache.get("weather:rome") == {"temp": 20}
    clock.advance(10.1)
    assert cache.get("weather:rome") is None


def test_memory_cache_lru_eviction_order() -> None:
    clock = FakeClock()
    cache = MemoryCache(max_size=2, clock=clock)
    cache.set("a", "A", ttl_seconds=100)
    cache.set("b", "B", ttl_seconds=100)

    assert cache.get("a") == "A"
    cache.set("c", "C", ttl_seconds=100)

    assert cache.get("a") == "A"
    assert cache.get("b") is None
    assert cache.get("c") == "C"


def test_rate_limiter_blocks_burst_and_refills() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate_per_second=1.0, capacity=2.0, clock=clock)

    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False
    assert limiter.wait_time() == 1.0

    clock.advance(1.0)
    assert limiter.allow() is True


def test_cache_key_helpers_are_stable() -> None:
    assert normalize_text("  Rome   City ") == "rome city"
    bbox_hash_1 = hash_bbox((41.9028, 12.4964, 41.95, 12.55))
    bbox_hash_2 = hash_bbox([41.9028, 12.4964, 41.95, 12.55])
    assert bbox_hash_1 == bbox_hash_2

    key = make_cache_key("Geocode", "  Rome ", {"lang": "it", "limit": 3})
    assert key.startswith("geocode:rome:")
