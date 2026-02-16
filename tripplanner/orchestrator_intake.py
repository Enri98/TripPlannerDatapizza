"""Orchestrator intake step: TripSpec draft extraction + clarification logic."""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from tripplanner.contracts import TripSpec
from tripplanner.guardrails import (
    DateGuardrailError,
    parse_date_expression,
    resolve_weekend_range,
    validate_date_range,
)


class TripSpecDraft(BaseModel):
    destination_text: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    date_expression: str | None = None
    budget_amount: float | None = None
    budget_currency: str | None = None
    budget_scope: Literal["total", "per_person"] | None = None
    num_travelers: int | None = None
    preferences_tags: list[str] = Field(default_factory=list)
    pace: Literal["relaxed", "standard", "packed"] | None = None
    mobility: Literal["walk_only", "public_transport", "car"] | None = None
    accessibility: str | None = None
    input_language: str | None = None
    output_language: str | None = None


class ClarifyingQuestion(BaseModel):
    reason: str
    questions: list[str] = Field(min_length=1, max_length=2)


class IntakeReady(BaseModel):
    status: Literal["ready"]
    tripspec: TripSpec


class IntakeClarification(BaseModel):
    status: Literal["clarification_needed"]
    clarifying_question: ClarifyingQuestion


IntakeResult = IntakeReady | IntakeClarification


class DraftExtractor(Protocol):
    def extract(self, query: str, now_ts: datetime, timezone_name: str) -> TripSpecDraft:
        ...


class DatapizzaGeminiDraftExtractor:
    """Gemini draft extractor implemented through Datapizza Google client."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is required for Datapizza Gemini extraction.")

        resolved_model = model or os.getenv("TRIPPLANNER_GEMINI_MODEL", "gemini-2.0-flash")
        system_prompt = (
            "Extract a TripSpecDraft JSON from user trip requests. "
            "Do not ask questions. Return only structured fields."
        )
        from datapizza.clients.factory import ClientFactory

        self._client = ClientFactory.create(
            provider="google",
            api_key=resolved_api_key,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=0.0,
        )

    def extract(self, query: str, now_ts: datetime, timezone_name: str) -> TripSpecDraft:
        prompt = (
            "Extract a TripSpecDraft from the user request.\n"
            f"request_now_ts: {now_ts.isoformat()}\n"
            f"request_timezone: {timezone_name}\n"
            f"user_query: {query}"
        )
        response = self._client.structured_response(input=prompt, output_cls=TripSpecDraft)
        if not response.structured_data:
            raise ValueError("Gemini extractor returned no structured draft.")
        payload = response.structured_data[0]
        if isinstance(payload, TripSpecDraft):
            return payload
        return TripSpecDraft.model_validate(payload.model_dump())


class OrchestratorIntake:
    """Converts user query into either a valid TripSpec or clarifying questions."""

    _AMBIGUOUS_COUNTRIES = {
        "spain",
        "italy",
        "france",
        "germany",
        "portugal",
        "greece",
        "japan",
        "usa",
        "united states",
    }

    def __init__(
        self,
        extractor: DraftExtractor | None = None,
        *,
        min_trip_days: int = 1,
        max_trip_days: int = 30,
    ) -> None:
        self._extractor = extractor
        self._min_trip_days = min_trip_days
        self._max_trip_days = max_trip_days

    def _get_extractor(self) -> DraftExtractor:
        if self._extractor is None:
            self._extractor = DatapizzaGeminiDraftExtractor()
        return self._extractor

    def process(
        self,
        query: str,
        *,
        now_ts: datetime | None = None,
        timezone_name: str = "UTC",
    ) -> IntakeResult:
        request_now = now_ts or datetime.now(timezone.utc)
        draft = self._get_extractor().extract(query, request_now, timezone_name)

        questions: list[str] = []
        reason = "missing_information"

        destination_text = (draft.destination_text or "").strip()
        if not destination_text:
            questions.append("Which city or region would you like to visit?")
            reason = "destination_missing"
        elif self._is_ambiguous_destination(destination_text):
            questions.append(
                "Which city or region in that country should I plan for?"
            )
            reason = "destination_ambiguous"

        parsed_range = self._resolve_dates(draft, request_now, timezone_name)
        if parsed_range is None:
            questions.append("What are your exact travel dates (start and end)?")
            if reason == "missing_information":
                reason = "dates_missing"
        else:
            start_date, end_date = parsed_range
            try:
                validate_date_range(
                    start_date,
                    end_date,
                    min_trip_days=self._min_trip_days,
                    max_trip_days=self._max_trip_days,
                )
            except DateGuardrailError as exc:
                questions.append(str(exc))
                if reason == "missing_information":
                    reason = "dates_invalid"

        if draft.budget_amount is None:
            questions.append("What is your budget and currency for this trip?")
            if reason == "missing_information":
                reason = "budget_missing"

        if questions:
            return IntakeClarification(
                status="clarification_needed",
                clarifying_question=ClarifyingQuestion(
                    reason=reason,
                    questions=questions[:2],
                ),
            )

        start_date, end_date = parsed_range  # type: ignore[misc]
        budget_scope = draft.budget_scope or "total"
        input_language = draft.input_language or "en"
        output_language = draft.output_language or input_language

        tripspec = TripSpec.model_validate(
            {
                "request_context": {
                    "now_ts": request_now.isoformat(),
                    "timezone": timezone_name,
                    "input_language": input_language,
                    "output_language": output_language,
                },
                "budget": {
                    "amount": draft.budget_amount,
                    "currency": (draft.budget_currency or "EUR").upper(),
                    "scope": budget_scope,
                    "num_travelers": draft.num_travelers,
                },
                "legs": [
                    {
                        "destination_text": destination_text,
                        "date_range": {
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                    }
                ],
                "preferences": {
                    "tags": draft.preferences_tags,
                },
                "constraints": {
                    "pace": draft.pace or "standard",
                    "mobility": draft.mobility or "public_transport",
                    "accessibility": draft.accessibility,
                },
            }
        )
        return IntakeReady(status="ready", tripspec=tripspec)

    def _resolve_dates(
        self,
        draft: TripSpecDraft,
        now_ts: datetime,
        timezone_name: str,
    ) -> tuple[date, date] | None:
        try:
            if draft.start_date and draft.end_date:
                return (
                    parse_date_expression(draft.start_date, now_ts, timezone_name),
                    parse_date_expression(draft.end_date, now_ts, timezone_name),
                )

            if draft.date_expression:
                expression = " ".join(draft.date_expression.lower().split())
                if expression in {"next weekend", "this weekend"}:
                    return resolve_weekend_range(expression, now_ts, timezone_name)
                resolved = parse_date_expression(draft.date_expression, now_ts, timezone_name)
                return resolved, resolved
        except DateGuardrailError:
            return None

        return None

    def _is_ambiguous_destination(self, destination_text: str) -> bool:
        normalized = " ".join(destination_text.lower().split())
        return normalized in self._AMBIGUOUS_COUNTRIES
