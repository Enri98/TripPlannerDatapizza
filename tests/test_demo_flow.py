"""Integration-style tests for end-to-end demo flow."""

from __future__ import annotations

from datetime import datetime, timezone

from tripplanner.demo_flow import DemoFlow


def test_demo_flow_returns_clarification_for_ambiguous_destination() -> None:
    flow = DemoFlow(offline=True)
    payload = flow.run(
        "Plan a 5-day trip to Spain next weekend",
        now_ts=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Rome",
    )

    assert payload["status"] == "clarification_needed"
    assert "city or region" in str(payload["clarifying_question"]).lower()


def test_demo_flow_builds_multi_destination_day_by_day_itinerary() -> None:
    flow = DemoFlow(offline=True)
    payload = flow.run(
        "Plan a 4-day trip to Rome and Florence next weekend",
        now_ts=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Rome",
    )

    assert payload["status"] == "completed"
    days = payload["days"]
    assert isinstance(days, list)
    assert len(days) == 4
    assert days[0]["destination"] == "Rome"
    assert days[-1]["destination"] == "Florence"
    assert payload["stages"]


def test_demo_flow_resolves_relative_date_with_request_timestamp() -> None:
    flow = DemoFlow(offline=True)
    payload = flow.run(
        "Plan a trip to Rome next weekend",
        now_ts=datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Rome",
    )
    assert payload["status"] == "completed"
    days = payload["days"]
    assert days[0]["date"] == "2026-02-28"
