"""Unit tests for deterministic orchestrator planning."""

from __future__ import annotations

from tripplanner.contracts import TripSpec
from tripplanner.planner import OrchestratorPlanner


def _tripspec_with_legs(*, resolved_geo_by_leg: list[bool]) -> TripSpec:
    legs = []
    for idx, resolved in enumerate(resolved_geo_by_leg):
        leg = {
            "destination_text": f"City {idx}",
            "date_range": {
                "start_date": "2026-03-10",
                "end_date": "2026-03-12",
            },
        }
        if resolved:
            leg["geo"] = {
                "lat": 41.9 + idx,
                "lon": 12.4 + idx,
                "bbox": [41.8 + idx, 12.3 + idx, 42.0 + idx, 12.7 + idx],
                "place_name": f"City {idx}",
                "country_code": "it",
            }
        legs.append(leg)

    return TripSpec.model_validate(
        {
            "request_context": {
                "now_ts": "2026-02-16T10:30:00Z",
                "timezone": "Europe/Rome",
                "input_language": "en",
                "output_language": "en",
            },
            "budget": {
                "amount": 1000,
                "currency": "EUR",
                "scope": "total",
                "num_travelers": 2,
            },
            "legs": legs,
            "preferences": {"tags": []},
            "constraints": {"pace": "standard", "mobility": "public_transport"},
        }
    )


def test_single_leg_unresolved_geo_requires_geo_before_enrichment() -> None:
    tripspec = _tripspec_with_legs(resolved_geo_by_leg=[False])
    plan = OrchestratorPlanner().generate(tripspec)
    task_ids = [task.task_id for task in plan.tasks]

    assert "geo_leg_0" in task_ids
    weather = next(task for task in plan.tasks if task.task_id == "weather_leg_0")
    poi = next(task for task in plan.tasks if task.task_id == "poi_leg_0")
    assert weather.depends_on == ["geo_leg_0"]
    assert poi.depends_on == ["geo_leg_0"]


def test_single_leg_resolved_geo_has_no_geo_task() -> None:
    tripspec = _tripspec_with_legs(resolved_geo_by_leg=[True])
    plan = OrchestratorPlanner().generate(tripspec)
    task_ids = [task.task_id for task in plan.tasks]

    assert "geo_leg_0" not in task_ids
    weather = next(task for task in plan.tasks if task.task_id == "weather_leg_0")
    poi = next(task for task in plan.tasks if task.task_id == "poi_leg_0")
    assert weather.depends_on == []
    assert poi.depends_on == []


def test_multi_leg_creates_transport_tasks_with_dependencies() -> None:
    tripspec = _tripspec_with_legs(resolved_geo_by_leg=[False, False])
    plan = OrchestratorPlanner().generate(tripspec)
    task_ids = [task.task_id for task in plan.tasks]

    assert "transport_leg_0_1" in task_ids
    transport = next(task for task in plan.tasks if task.task_id == "transport_leg_0_1")
    assert "weather_leg_0" in transport.depends_on
    assert "poi_leg_0" in transport.depends_on
    assert "geo_leg_1" in transport.depends_on

    synth = next(task for task in plan.tasks if task.task_id == "synth_trip")
    assert "transport_leg_0_1" in synth.depends_on


def test_generated_plan_is_schema_valid() -> None:
    tripspec = _tripspec_with_legs(resolved_geo_by_leg=[False, True, False])
    plan = OrchestratorPlanner().generate(tripspec)

    assert plan.tasks
    assert plan.tasks[-1].agent == "synth"
