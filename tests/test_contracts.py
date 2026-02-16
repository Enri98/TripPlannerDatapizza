"""Tests for JSON data contracts and example fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tripplanner.contracts import Plan, StandardAgentResult, TripSpec


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "contracts"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_tripspec_fixture_validates() -> None:
    payload = _load_fixture("tripspec.json")
    model = TripSpec.model_validate(payload)
    assert model.legs[0].destination_text == "Rome, Italy"


def test_plan_fixture_validates() -> None:
    payload = _load_fixture("plan.json")
    model = Plan.model_validate(payload)
    assert model.tasks[0].agent == "geo"
    assert model.tasks[-1].agent == "synth"


def test_standard_agent_result_fixture_validates() -> None:
    payload = _load_fixture("standard_agent_result.json")
    model = StandardAgentResult.model_validate(payload)
    assert 0.0 <= model.confidence <= 1.0
    assert model.evidence[0].source == "open-meteo"


def test_standard_agent_result_rejects_invalid_confidence() -> None:
    payload = _load_fixture("standard_agent_result.json")
    payload["confidence"] = 1.2
    with pytest.raises(ValidationError):
        StandardAgentResult.model_validate(payload)


def test_plan_rejects_invalid_agent_name() -> None:
    payload = _load_fixture("plan.json")
    payload["tasks"][0]["agent"] = "invalid_agent"
    with pytest.raises(ValidationError):
        Plan.model_validate(payload)
