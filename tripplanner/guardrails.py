"""Deterministic date guardrails for trip planning."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import dateparser


class DateGuardrailError(ValueError):
    """Raised when date parsing/validation fails."""


def _to_local_now(now_ts: datetime, timezone: str) -> datetime:
    tz = ZoneInfo(timezone)
    if now_ts.tzinfo is None:
        return now_ts.replace(tzinfo=tz)
    return now_ts.astimezone(tz)


def parse_date_expression(expression: str, now_ts: datetime, timezone: str) -> date:
    text = expression.strip()
    if not text:
        raise DateGuardrailError("Date expression cannot be empty.")

    normalized = " ".join(text.lower().split())
    local_now = _to_local_now(now_ts, timezone)

    if normalized in {"next weekend", "this weekend"}:
        return resolve_weekend_range(normalized, now_ts, timezone)[0]

    parsed = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": local_now,
            "TIMEZONE": timezone,
            "TO_TIMEZONE": timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DAY_OF_MONTH": "first",
        },
    )
    if parsed is None:
        raise DateGuardrailError(f"Could not parse date expression: '{expression}'.")

    return parsed.astimezone(ZoneInfo(timezone)).date()


def resolve_weekend_range(expression: str, now_ts: datetime, timezone: str) -> tuple[date, date]:
    normalized = " ".join(expression.lower().split())
    if normalized not in {"next weekend", "this weekend"}:
        raise DateGuardrailError(
            "Weekend resolver supports only 'this weekend' and 'next weekend'."
        )

    local_now = _to_local_now(now_ts, timezone)
    days_until_saturday = (5 - local_now.weekday()) % 7
    saturday = local_now.date() + timedelta(days=days_until_saturday)
    if normalized == "next weekend":
        saturday = saturday + timedelta(days=7)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday


def parse_date_range(
    start_expression: str,
    end_expression: str,
    now_ts: datetime,
    timezone: str,
    *,
    min_trip_days: int = 1,
    max_trip_days: int = 30,
) -> tuple[date, date]:
    start_date = parse_date_expression(start_expression, now_ts, timezone)
    end_date = parse_date_expression(end_expression, now_ts, timezone)
    validate_date_range(start_date, end_date, min_trip_days=min_trip_days, max_trip_days=max_trip_days)
    return start_date, end_date


def validate_date_range(
    start_date: date,
    end_date: date,
    *,
    min_trip_days: int = 1,
    max_trip_days: int = 30,
) -> None:
    if min_trip_days <= 0:
        raise DateGuardrailError("min_trip_days must be > 0.")
    if max_trip_days < min_trip_days:
        raise DateGuardrailError("max_trip_days must be >= min_trip_days.")
    if end_date < start_date:
        raise DateGuardrailError(
            f"Invalid date range: end_date ({end_date.isoformat()}) is before "
            f"start_date ({start_date.isoformat()})."
        )

    trip_days = (end_date - start_date).days + 1
    if trip_days < min_trip_days:
        raise DateGuardrailError(
            f"Trip duration {trip_days} day(s) is below minimum {min_trip_days}."
        )
    if trip_days > max_trip_days:
        raise DateGuardrailError(
            f"Trip duration {trip_days} day(s) exceeds maximum {max_trip_days}."
        )


def validate_non_overlapping_legs(legs: list[tuple[date, date]]) -> None:
    sorted_legs = sorted(legs, key=lambda leg: leg[0])
    for index in range(1, len(sorted_legs)):
        previous_end = sorted_legs[index - 1][1]
        current_start = sorted_legs[index][0]
        if current_start <= previous_end:
            raise DateGuardrailError(
                "Leg date ranges overlap or are not strictly ordered."
            )
