"""Deterministic orchestrator planner for specialist task graph creation."""

from __future__ import annotations

from tripplanner.contracts import Plan, PlanTask, TripSpec


class OrchestratorPlanner:
    """Build a schema-valid execution plan from a validated TripSpec."""

    def generate(self, tripspec: TripSpec) -> Plan:
        tasks: list[PlanTask] = []
        leg_geo_task: dict[int, str] = {}
        leg_weather_task: dict[int, str] = {}
        leg_poi_task: dict[int, str] = {}
        transport_task_ids: list[str] = []

        for idx, leg in enumerate(tripspec.legs):
            input_ref = f"legs[{idx}]"
            if leg.geo is None:
                geo_id = f"geo_leg_{idx}"
                leg_geo_task[idx] = geo_id
                tasks.append(
                    PlanTask(
                        task_id=geo_id,
                        agent="geo",
                        input_ref=input_ref,
                        depends_on=[],
                    )
                )

            if idx > 0:
                transport_id = f"transport_leg_{idx - 1}_{idx}"
                depends_on = [
                    leg_weather_task[idx - 1],
                    leg_poi_task[idx - 1],
                ]
                if idx in leg_geo_task:
                    depends_on.append(leg_geo_task[idx])
                tasks.append(
                    PlanTask(
                        task_id=transport_id,
                        agent="transport",
                        input_ref=f"legs[{idx - 1}]->legs[{idx}]",
                        depends_on=depends_on,
                        parallel_group=f"transfer_{idx - 1}_{idx}",
                    )
                )
                transport_task_ids.append(transport_id)

            weather_id = f"weather_leg_{idx}"
            weather_deps = [leg_geo_task[idx]] if idx in leg_geo_task else []
            tasks.append(
                PlanTask(
                    task_id=weather_id,
                    agent="weather",
                    input_ref=input_ref,
                    depends_on=weather_deps,
                    parallel_group=f"leg_{idx}_enrichment",
                )
            )
            leg_weather_task[idx] = weather_id

            poi_id = f"poi_leg_{idx}"
            poi_deps = [leg_geo_task[idx]] if idx in leg_geo_task else []
            tasks.append(
                PlanTask(
                    task_id=poi_id,
                    agent="poi",
                    input_ref=input_ref,
                    depends_on=poi_deps,
                    parallel_group=f"leg_{idx}_enrichment",
                )
            )
            leg_poi_task[idx] = poi_id

        synth_deps: list[str] = []
        for idx in range(len(tripspec.legs)):
            synth_deps.append(leg_weather_task[idx])
            synth_deps.append(leg_poi_task[idx])
        synth_deps.extend(transport_task_ids)

        tasks.append(
            PlanTask(
                task_id="synth_trip",
                agent="synth",
                input_ref="trip",
                depends_on=synth_deps,
                stop_condition="validated_inputs_present",
            )
        )
        return Plan(tasks=tasks)
