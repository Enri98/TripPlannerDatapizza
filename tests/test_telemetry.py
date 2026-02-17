"""Telemetry baseline tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_demo_with_env(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "TRIPPLANNER_DEMO_OFFLINE": "1",
            "TRIPPLANNER_NOW_TS": "2026-02-17T10:00:00Z",
            "TRIPPLANNER_TIMEZONE": "Europe/Rome",
        }
    )
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "tripplanner", "demo", "Plan a 3-day trip to Rome next weekend"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_demo_runs_with_tracing_disabled() -> None:
    completed = _run_demo_with_env({"TRIPPLANNER_TRACING_ENABLED": "0"})
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"


def test_demo_runs_with_tracing_enabled() -> None:
    completed = _run_demo_with_env(
        {
            "TRIPPLANNER_TRACING_ENABLED": "1",
            "TRIPPLANNER_TRACING_EXPORTER": "console",
        }
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
