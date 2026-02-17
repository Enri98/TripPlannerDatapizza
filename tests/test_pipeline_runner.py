from __future__ import annotations

import os
from urllib.error import HTTPError, URLError

import pytest

from tripplanner.geo_tool import GeoToolError
from tripplanner.pipeline_runner import (
    _extract_wait_seconds,
    _invoke_with_transient_retry,
    _invoke_with_geo_rate_limit_retry,
    _is_retryable_transient_error,
    load_env_file,
    render_itinerary_text,
    run_pipeline,
)


def test_load_env_file_sets_only_missing_keys(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=new\nBAR=bar\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "existing")
    monkeypatch.delenv("BAR", raising=False)

    load_env_file(str(env_file))

    assert os.environ["FOO"] == "existing"
    assert os.environ["BAR"] == "bar"


def test_run_pipeline_uses_runner_and_returns_payload(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=dummy\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    captured = {}

    class DummyRunner:
        def run(self, query, *, now_ts=None, timezone_name="UTC", output_language=None):  # type: ignore[no-untyped-def]
            captured["query"] = query
            captured["timezone_name"] = timezone_name
            captured["output_language"] = output_language
            return {"status": "completed", "query": query}

    monkeypatch.setattr("tripplanner.pipeline_runner.PipelineRunner", DummyRunner)
    payload = run_pipeline("Trip to Rome", timezone_name="Europe/Rome", output_language="en")

    assert payload["status"] == "completed"
    assert captured == {
        "query": "Trip to Rome",
        "timezone_name": "Europe/Rome",
        "output_language": "en",
    }


def test_extract_wait_seconds_parses_nominatim_message() -> None:
    assert _extract_wait_seconds("Nominatim rate limit exceeded. Retry after 0.67 seconds.") == 0.67
    assert _extract_wait_seconds("some other error") is None


def test_invoke_with_geo_rate_limit_retry_waits_and_recovers() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def flaky_call() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise GeoToolError("Nominatim rate limit exceeded. Retry after 0.50 seconds.")
        return "ok"

    result = _invoke_with_geo_rate_limit_retry(
        flaky_call,
        sleep_fn=lambda value: sleeps.append(value),
    )

    assert result == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.55]


def test_invoke_with_geo_rate_limit_retry_raises_after_max_attempts() -> None:
    def always_fails() -> str:
        raise GeoToolError("Nominatim rate limit exceeded. Retry after 0.10 seconds.")

    with pytest.raises(GeoToolError):
        _invoke_with_geo_rate_limit_retry(always_fails, max_attempts=1, sleep_fn=lambda _: None)


def test_invoke_with_transient_retry_retries_http_504_and_recovers() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def flaky_call() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError("https://example.com", 504, "Gateway Timeout", {}, None)
        return "ok"

    result = _invoke_with_transient_retry(
        flaky_call,
        max_attempts=2,
        base_delay_seconds=0.25,
        sleep_fn=lambda value: sleeps.append(value),
    )

    assert result == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.25]


def test_is_retryable_transient_error_classifies_errors() -> None:
    assert _is_retryable_transient_error(HTTPError("https://example.com", 504, "x", {}, None))
    assert _is_retryable_transient_error(URLError("temporary unavailable"))
    assert _is_retryable_transient_error(TimeoutError("timed out"))
    assert not _is_retryable_transient_error(ValueError("bad input"))


def test_render_itinerary_text_includes_blocks_caveats_and_summary() -> None:
    text = render_itinerary_text(
        days=[
            {
                "day_index": 1,
                "date": "2026-03-10",
                "destination": "Rome",
                "weather_note": "Weather looks manageable for outdoor plans.",
                "activities": [
                    {"name": "Colosseum", "period": "morning", "indoor": False},
                    {"name": "Vatican Museums", "period": "afternoon", "indoor": True},
                ],
                "transport_notes": ["Fast train suggestion"],
                "alternatives": ["Indoor backup: museum"],
            }
        ],
        title="Day-by-day itinerary",
        warnings=["Some places may require advance booking."],
    )
    assert "Day-by-day itinerary" in text
    assert "Day 1 (2026-03-10) - Rome" in text
    assert "Transfer:" in text
    assert "Caveats:" in text
    assert "Summary: 1 day(s) across 1 destination(s): Rome." in text
