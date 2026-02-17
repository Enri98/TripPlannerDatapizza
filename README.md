# TripPlanner

TripPlanner is a Datapizza AI + Gemini multi-agent travel planner that takes a natural-language trip request and produces a day-by-day itinerary.

## What It Does

- Uses an orchestrator + specialist agents flow (`geo`, `weather`, `poi`, `transport`, `synth`).
- Supports single-destination and multi-destination trips.
- Handles relative dates (for example: `next weekend`) with request timestamp + timezone context.
- Uses free data sources only:
  - Open-Meteo (weather)
  - OSM Nominatim (geocoding)
  - OSM Overpass (POIs)
  - DuckDuckGo web search (grounding/transport context)
- Returns structured JSON and (with CLI text mode) a human-readable itinerary.
- Includes tracing, caching, throttling, and deterministic tests.

## Quick Start

1. Create and activate a virtual environment
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux:
     - `python -m venv .venv`
     - `source .venv/bin/activate`
2. Install dependencies
   - `python -m pip install --upgrade pip`
   - `python -m pip install -e ".[dev]"`
3. Configure environment variables
   - Add your Gemini key in `.env` (for example `GEMINI_API_KEY=...`).

## Run

- Real pipeline (JSON output):
  - `python -m tripplanner run "Plan a 5-day trip to Rome and Florence next weekend with 1800 EUR"`
- Real pipeline (text itinerary output):
  - `python -m tripplanner run "Plan a 5-day trip to Rome and Florence next weekend with 1800 EUR" --format text`
- Demo flow:
  - `python -m tripplanner demo "Plan a 3-day trip to Rome next weekend"`

## Tests

- Run all tests:
  - `pytest -q`

## Telemetry

- Console tracing is enabled by default and rendered in compact form.
- Useful environment flags:
  - `TRIPPLANNER_TRACING_ENABLED=1|0`
  - `TRIPPLANNER_TRACING_EXPORTER=console|otlp`
  - `TRIPPLANNER_TRACING_CONSOLE_MODE=compact|raw`
