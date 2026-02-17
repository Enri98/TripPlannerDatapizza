"""Standalone script entrypoint for the real pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json

from tripplanner.pipeline_runner import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TripPlanner real Datapizza+Gemini pipeline.")
    parser.add_argument("query", help="Natural-language trip query.")
    parser.add_argument("--timezone", default="UTC", help="IANA timezone for relative dates.")
    parser.add_argument("--output-language", choices=["en", "it"], default=None)
    parser.add_argument("--now-ts", default=None, help="Optional fixed ISO-8601 timestamp.")
    args = parser.parse_args()

    now_ts = datetime.fromisoformat(args.now_ts.replace("Z", "+00:00")) if args.now_ts else None
    payload = run_pipeline(
        args.query,
        now_ts=now_ts,
        timezone_name=args.timezone,
        output_language=args.output_language,
    )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
