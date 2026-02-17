"""Command line interface for TripPlanner."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from typing import Sequence

from tripplanner.demo_flow import run_demo_flow
from tripplanner.telemetry import start_span


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tripplanner",
        description="TripPlanner demo CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a stub itinerary planner output.",
    )
    demo_parser.add_argument("query", help="Natural-language trip query.")
    return parser


def run_demo(query: str) -> dict[str, object]:
    timezone_name = os.getenv("TRIPPLANNER_TIMEZONE", "UTC")
    output_language = os.getenv("TRIPPLANNER_OUTPUT_LANGUAGE")
    now_ts_raw = os.getenv("TRIPPLANNER_NOW_TS")
    now_ts = None
    if now_ts_raw:
        now_ts = datetime.fromisoformat(now_ts_raw.replace("Z", "+00:00"))

    with start_span("orchestrator.demo"):
        payload = run_demo_flow(
            query,
            now_ts=now_ts,
            timezone_name=timezone_name,
            output_language=output_language,
        )
    payload["query"] = query
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        print(json.dumps(run_demo(args.query), ensure_ascii=True))
        return 0

    parser.print_help()
    return 0
