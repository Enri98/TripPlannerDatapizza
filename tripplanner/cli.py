"""Command line interface for TripPlanner."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from typing import Sequence

from tripplanner.demo_flow import run_demo_flow
from tripplanner.pipeline_runner import run_pipeline
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

    run_parser = subparsers.add_parser(
        "run",
        help="Run the real Datapizza+Gemini pipeline.",
    )
    run_parser.add_argument("query", help="Natural-language trip query.")
    run_parser.add_argument(
        "--timezone",
        default=None,
        help="IANA timezone (defaults to TRIPPLANNER_TIMEZONE or UTC).",
    )
    run_parser.add_argument(
        "--output-language",
        choices=["en", "it"],
        default=None,
        help="Force itinerary output language.",
    )
    run_parser.add_argument(
        "--now-ts",
        default=None,
        help="Optional fixed timestamp (ISO-8601) for deterministic runs.",
    )
    run_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format for run command.",
    )
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


def run_real_pipeline(
    query: str,
    *,
    timezone_name: str | None = None,
    output_language: str | None = None,
    now_ts: datetime | None = None,
) -> dict[str, object]:
    tz_name = timezone_name or os.getenv("TRIPPLANNER_TIMEZONE", "UTC")
    return run_pipeline(
        query,
        now_ts=now_ts,
        timezone_name=tz_name,
        output_language=output_language or os.getenv("TRIPPLANNER_OUTPUT_LANGUAGE"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        print(json.dumps(run_demo(args.query), ensure_ascii=True))
        return 0

    if args.command == "run":
        explicit_now = None
        if args.now_ts:
            explicit_now = datetime.fromisoformat(args.now_ts.replace("Z", "+00:00"))
        payload = run_real_pipeline(
            args.query,
            timezone_name=args.timezone,
            output_language=args.output_language,
            now_ts=explicit_now,
        )
        if args.format == "text":
            if payload.get("status") == "completed":
                print(str(payload.get("itinerary_text", "")).strip())
            else:
                print(str(payload.get("clarifying_question", "")).strip())
        else:
            print(json.dumps(payload, ensure_ascii=True))
        return 0

    parser.print_help()
    return 0
