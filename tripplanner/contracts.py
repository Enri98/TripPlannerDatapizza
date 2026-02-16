"""Structured data contracts for orchestration and specialist outputs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestContext(BaseModel):
    now_ts: datetime
    timezone: str
    input_language: str
    output_language: str


class Budget(BaseModel):
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    scope: Literal["total", "per_person"]
    num_travelers: int | None = Field(default=None, gt=0)


class GeoPoint(BaseModel):
    lat: float
    lon: float
    bbox: list[float] = Field(min_length=4, max_length=4)
    place_name: str
    country_code: str = Field(min_length=2, max_length=2)


class DateRange(BaseModel):
    start_date: date
    end_date: date


class TripLeg(BaseModel):
    destination_text: str
    geo: GeoPoint | None = None
    date_range: DateRange


class Preferences(BaseModel):
    tags: list[str] = Field(default_factory=list)


class Constraints(BaseModel):
    pace: Literal["relaxed", "standard", "packed"]
    mobility: Literal["walk_only", "public_transport", "car"]
    accessibility: str | None = None


class TripSpec(BaseModel):
    request_context: RequestContext
    budget: Budget
    legs: list[TripLeg] = Field(min_length=1)
    preferences: Preferences
    constraints: Constraints


class PlanTask(BaseModel):
    task_id: str
    agent: Literal["geo", "weather", "poi", "transport", "synth"]
    input_ref: str
    depends_on: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    stop_condition: str | None = None


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(min_length=1)


class EvidenceItem(BaseModel):
    source: str
    title: str
    snippet: str
    retrieved_at: datetime
    url: str | None = None


class StandardAgentResult(BaseModel):
    data: dict[str, Any]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    cache_key: str


class ContractsBundle(BaseModel):
    """Convenience container used in tests/examples if needed later."""

    model_config = ConfigDict(extra="forbid")

    tripspec: TripSpec
    plan: Plan
    result: StandardAgentResult
