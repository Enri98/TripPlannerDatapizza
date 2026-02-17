"""Tests for Orchestrator Intake (step 06)."""

from __future__ import annotations

from datetime import datetime

from tripplanner.orchestrator_intake import OrchestratorIntake, TripSpecDraft


FIXED_NOW_TS = datetime.fromisoformat("2026-02-16T10:30:00+00:00")
TIMEZONE = "Europe/Rome"


class FakeExtractor:
    def __init__(self, draft: TripSpecDraft) -> None:
        self.draft = draft

    def extract(self, query: str, now_ts: datetime, timezone_name: str) -> TripSpecDraft:
        return self.draft


def test_spain_next_weekend_requests_city_or_region_clarification() -> None:
    draft = TripSpecDraft(
        destination_text="Spain",
        date_expression="next weekend",
        budget_amount=1200,
        budget_currency="EUR",
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a trip to Spain next weekend",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "clarification_needed"
    assert "city or region" in result.clarifying_question.questions[0].lower()


def test_missing_dates_requests_date_clarification() -> None:
    draft = TripSpecDraft(
        destination_text="Rome",
        budget_amount=900,
        budget_currency="EUR",
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a trip to Rome",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "clarification_needed"
    assert "date" in " ".join(result.clarifying_question.questions).lower()


def test_budget_scope_defaults_to_total_when_missing() -> None:
    draft = TripSpecDraft(
        destination_text="Rome",
        start_date="2026-03-10",
        end_date="2026-03-12",
        budget_amount=1000,
        budget_currency="eur",
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a trip to Rome from 2026-03-10 to 2026-03-12 with 1000 EUR budget",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "ready"
    assert result.tripspec.budget.scope == "total"


def test_budget_scope_per_person_is_preserved() -> None:
    draft = TripSpecDraft(
        destination_text="Rome",
        start_date="2026-03-10",
        end_date="2026-03-12",
        budget_amount=500,
        budget_currency="EUR",
        budget_scope="per_person",
        num_travelers=2,
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a trip to Rome from 2026-03-10 to 2026-03-12 with 500 EUR per person",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "ready"
    assert result.tripspec.budget.scope == "per_person"


def test_combined_destination_text_is_split_into_multiple_legs() -> None:
    draft = TripSpecDraft(
        destination_text="Rome and Florence",
        start_date="2026-03-10",
        end_date="2026-03-14",
        budget_amount=1500,
        budget_currency="EUR",
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a 5-day trip to Rome and Florence",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "ready"
    assert [leg.destination_text for leg in result.tripspec.legs] == ["Rome", "Florence"]
    assert result.tripspec.legs[0].date_range.start_date.isoformat() == "2026-03-10"
    assert result.tripspec.legs[0].date_range.end_date.isoformat() == "2026-03-12"
    assert result.tripspec.legs[1].date_range.start_date.isoformat() == "2026-03-13"
    assert result.tripspec.legs[1].date_range.end_date.isoformat() == "2026-03-14"


def test_multi_destination_with_too_few_days_requests_clarification() -> None:
    draft = TripSpecDraft(
        destination_text="Rome and Florence",
        start_date="2026-03-10",
        end_date="2026-03-10",
        budget_amount=900,
        budget_currency="EUR",
    )
    intake = OrchestratorIntake(extractor=FakeExtractor(draft))

    result = intake.process(
        "Plan a same-day trip to Rome and Florence",
        now_ts=FIXED_NOW_TS,
        timezone_name=TIMEZONE,
    )

    assert result.status == "clarification_needed"
    assert "shorter than the number of destinations" in result.clarifying_question.questions[0]
