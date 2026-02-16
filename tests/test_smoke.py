"""Smoke tests for the initial TripPlanner CLI scaffold."""

from __future__ import annotations

import json
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
    completed = subprocess.run(
        [sys.executable, "-m", "tripplanner", "demo", "test"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "stub"
    assert payload["query"] == "test"
