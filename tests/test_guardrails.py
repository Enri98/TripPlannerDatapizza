"""Unit tests for date guardrails."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tripplanner.guardrails import (
    DateGuardrailError,
    parse_date_expression,
    parse_date_range,
    resolve_weekend_range,
    validate_date_range,
    validate_non_overlapping_legs,
)


FIXED_NOW_TS = datetime.fromisoformat("2026-02-16T10:30:00+00:00")
TIMEZONE = "Europe/Rome"


def test_parse_absolute_date_expression() -> None:
    parsed = parse_date_expression("2026-03-10", FIXED_NOW_TS, TIMEZONE)
    assert parsed == date(2026, 3, 10)


def test_parse_relative_in_two_weeks_is_deterministic() -> None:
    parsed = parse_date_expression("in two weeks", FIXED_NOW_TS, TIMEZONE)
    assert parsed == date(2026, 3, 2)


def test_resolve_next_weekend_range_is_deterministic() -> None:
    saturday, sunday = resolve_weekend_range("next weekend", FIXED_NOW_TS, TIMEZONE)
    assert saturday == date(2026, 2, 28)
    assert sunday == date(2026, 3, 1)


def test_validate_date_range_rejects_end_before_start() -> None:
    with pytest.raises(DateGuardrailError, match="before"):
        validate_date_range(date(2026, 3, 5), date(2026, 3, 4))


def test_parse_date_range_rejects_excessive_trip_length() -> None:
    with pytest.raises(DateGuardrailError, match="exceeds maximum"):
        parse_date_range(
            "2026-03-01",
            "2026-04-15",
            FIXED_NOW_TS,
            TIMEZONE,
            max_trip_days=20,
        )


def test_validate_non_overlapping_legs_rejects_overlap() -> None:
    with pytest.raises(DateGuardrailError, match="overlap"):
        validate_non_overlapping_legs(
            [
                (date(2026, 3, 10), date(2026, 3, 12)),
                (date(2026, 3, 12), date(2026, 3, 15)),
            ]
        )
