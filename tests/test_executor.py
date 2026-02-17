from __future__ import annotations

from datetime import datetime, timezone

from tripplanner.contracts import EvidenceItem, Plan, PlanTask, StandardAgentResult
from tripplanner.executor import OrchestratorExecutor


def _result(*, confidence: float = 0.8, cache_key: str = "k") -> StandardAgentResult:
    return StandardAgentResult(
        data={"ok": True},
        evidence=[
            EvidenceItem(
                source="test",
                title="e",
                snippet="e",
                retrieved_at=datetime.now(tz=timezone.utc),
            )
        ],
        confidence=confidence,
        warnings=[],
        cache_key=cache_key,
    )


def test_executor_runs_dependencies_in_deterministic_stages() -> None:
    plan = Plan(
        tasks=[
            PlanTask(task_id="geo_leg_0", agent="geo", input_ref="legs[0]"),
            PlanTask(
                task_id="weather_leg_0",
                agent="weather",
                input_ref="legs[0]",
                depends_on=["geo_leg_0"],
                parallel_group="leg_0_enrichment",
            ),
            PlanTask(
                task_id="poi_leg_0",
                agent="poi",
                input_ref="legs[0]",
                depends_on=["geo_leg_0"],
                parallel_group="leg_0_enrichment",
            ),
            PlanTask(
                task_id="transport_leg_0_1",
                agent="transport",
                input_ref="legs[0]->legs[1]",
                depends_on=["weather_leg_0", "poi_leg_0"],
            ),
            PlanTask(
                task_id="synth_trip",
                agent="synth",
                input_ref="trip",
                depends_on=["transport_leg_0_1"],
            ),
        ]
    )

    calls: list[str] = []

    def handler(task: PlanTask) -> StandardAgentResult:
        calls.append(task.task_id)
        return _result(cache_key=f"cache:{task.task_id}")

    executor = OrchestratorExecutor(
        handlers={
            "geo": handler,
            "weather": handler,
            "poi": handler,
            "transport": handler,
            "synth": handler,
        }
    )

    outcome = executor.execute(plan)

    assert outcome.status == "completed"
    assert calls == [
        "geo_leg_0",
        "poi_leg_0",
        "weather_leg_0",
        "transport_leg_0_1",
        "synth_trip",
    ]
    assert outcome.stages == [
        ["geo_leg_0"],
        ["poi_leg_0", "weather_leg_0"],
        ["transport_leg_0_1"],
        ["synth_trip"],
    ]


def test_executor_retries_once_and_recovers_on_retryable_issue() -> None:
    plan = Plan(
        tasks=[PlanTask(task_id="weather_leg_0", agent="weather", input_ref="legs[0]")]
    )
    calls = {"count": 0}

    def weather_handler(_task: PlanTask) -> StandardAgentResult:
        calls["count"] += 1
        if calls["count"] == 1:
            return _result(confidence=0.1, cache_key="cache:low")
        return _result(confidence=0.9, cache_key="cache:ok")

    executor = OrchestratorExecutor(handlers={"weather": weather_handler})

    outcome = executor.execute(plan)

    assert outcome.status == "completed"
    assert calls["count"] == 2
    assert outcome.records[0].success is True
    assert outcome.records[0].attempts == 2


def test_executor_returns_clarification_when_critical_task_fails() -> None:
    plan = Plan(tasks=[PlanTask(task_id="weather_leg_0", agent="weather", input_ref="legs[0]")])

    def invalid_weather_handler(_task: PlanTask) -> dict:
        return {"bad": "payload"}

    executor = OrchestratorExecutor(handlers={"weather": invalid_weather_handler})

    outcome = executor.execute(plan)

    assert outcome.status == "clarification_needed"
    assert outcome.clarifying_question is not None
    assert "weather_leg_0" in outcome.clarifying_question
    assert outcome.records[0].success is False
    assert outcome.records[0].attempts == 2
    assert outcome.records[0].issues == ["schema_invalid"]
