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
