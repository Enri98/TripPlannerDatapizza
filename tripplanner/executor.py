"""Plan execution + result evaluation with deterministic retry policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import ValidationError

from tripplanner.contracts import Plan, PlanTask, StandardAgentResult


TaskHandler = Callable[[PlanTask], StandardAgentResult | dict]


@dataclass
class TaskExecutionRecord:
    task_id: str
    agent: str
    success: bool
    attempts: int
    issues: list[str] = field(default_factory=list)


@dataclass
class ExecutionOutcome:
    status: Literal["completed", "clarification_needed"]
    results: dict[str, StandardAgentResult]
    records: list[TaskExecutionRecord]
    stages: list[list[str]]
    clarifying_question: str | None = None


class OrchestratorExecutor:
    """Executes plan tasks with dependency-aware batching and retry policy."""

    def __init__(
        self,
        handlers: dict[str, TaskHandler],
        *,
        confidence_threshold: float = 0.5,
        max_retries: int = 1,
        critical_agents: set[str] | None = None,
    ) -> None:
        self._handlers = handlers
        self._confidence_threshold = confidence_threshold
        self._max_retries = max_retries
        self._critical_agents = critical_agents or {"geo", "weather", "poi", "transport"}

    def execute(self, plan: Plan) -> ExecutionOutcome:
        pending: dict[str, PlanTask] = {task.task_id: task for task in plan.tasks}
        completed: set[str] = set()
        results: dict[str, StandardAgentResult] = {}
        records: list[TaskExecutionRecord] = []
        stages: list[list[str]] = []

        while pending:
            ready = [
                task
                for task in pending.values()
                if all(dep in completed for dep in task.depends_on)
            ]
            if not ready:
                raise RuntimeError("Plan has unresolved/circular dependencies.")

            grouped = self._group_ready_tasks(ready)
            for group in grouped:
                stage_task_ids: list[str] = []
                for task in group:
                    outcome = self._execute_task_with_retry(task)
                    records.append(outcome["record"])
                    if not outcome["success"]:
                        if task.agent in self._critical_agents:
                            question = (
                                f"I need clarification because task '{task.task_id}' failed "
                                f"({', '.join(outcome['issues'])})."
                            )
                            return ExecutionOutcome(
                                status="clarification_needed",
                                results=results,
                                records=records,
                                stages=stages,
                                clarifying_question=question,
                            )
                        completed.add(task.task_id)
                        pending.pop(task.task_id, None)
                        continue

                    result = outcome["result"]
                    results[task.task_id] = result
                    completed.add(task.task_id)
                    pending.pop(task.task_id, None)
                    stage_task_ids.append(task.task_id)

                if stage_task_ids:
                    stages.append(stage_task_ids)

        return ExecutionOutcome(
            status="completed",
            results=results,
            records=records,
            stages=stages,
            clarifying_question=None,
        )

    def _group_ready_tasks(self, ready: list[PlanTask]) -> list[list[PlanTask]]:
        buckets: dict[str, list[PlanTask]] = {}
        for task in ready:
            key = task.parallel_group or f"solo:{task.task_id}"
            buckets.setdefault(key, []).append(task)

        grouped: list[list[PlanTask]] = []
        for key in sorted(buckets.keys()):
            grouped.append(sorted(buckets[key], key=lambda t: t.task_id))
        return grouped

    def _execute_task_with_retry(self, task: PlanTask) -> dict:
        max_attempts = self._max_retries + 1
        last_issues: list[str] = []
        last_attempt = 1
        for attempt in range(1, max_attempts + 1):
            last_attempt = attempt
            result, issues = self._invoke_and_evaluate(task)
            if not issues:
                return {
                    "success": True,
                    "result": result,
                    "issues": [],
                    "record": TaskExecutionRecord(
                        task_id=task.task_id,
                        agent=task.agent,
                        success=True,
                        attempts=attempt,
                        issues=[],
                    ),
                }
            last_issues = issues

            retryable = any(
                item in {"schema_invalid", "evidence_empty", "confidence_low"}
                for item in issues
            )
            if attempt < max_attempts and retryable:
                continue
            break

        return {
            "success": False,
            "result": None,
            "issues": last_issues,
            "record": TaskExecutionRecord(
                task_id=task.task_id,
                agent=task.agent,
                success=False,
                attempts=last_attempt,
                issues=last_issues,
            ),
        }

    def _invoke_and_evaluate(self, task: PlanTask) -> tuple[StandardAgentResult | None, list[str]]:
        handler = self._handlers.get(task.agent)
        if handler is None:
            return None, ["handler_missing"]

        try:
            raw = handler(task)
        except Exception:
            return None, ["handler_exception"]

        try:
            result = StandardAgentResult.model_validate(raw)
        except ValidationError:
            return None, ["schema_invalid"]

        issues: list[str] = []
        if not result.evidence:
            issues.append("evidence_empty")
        if result.confidence < self._confidence_threshold:
            issues.append("confidence_low")
        if not result.cache_key:
            issues.append("consistency_invalid")
        return result, issues
