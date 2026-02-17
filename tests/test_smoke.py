"""Smoke tests for the initial TripPlanner CLI scaffold."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def test_module_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tripplanner", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "demo" in completed.stdout


def test_demo_command_returns_stub_payload() -> None:
    env = {
        **os.environ,
        "TRIPPLANNER_DEMO_OFFLINE": "1",
        "TRIPPLANNER_NOW_TS": "2026-02-17T10:00:00Z",
        "TRIPPLANNER_TIMEZONE": "Europe/Rome",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "tripplanner", "demo", "Plan a 3-day trip to Rome next weekend"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["query"] == "Plan a 3-day trip to Rome next weekend"
    assert payload["days"]
