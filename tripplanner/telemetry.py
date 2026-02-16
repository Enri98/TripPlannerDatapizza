"""OpenTelemetry setup and helper utilities."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from typing import ContextManager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_INITIALIZED = False
_ENABLED = False


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _as_bool(os.getenv("TRIPPLANNER_TRACING_ENABLED"), default=True)


def configure_telemetry() -> bool:
    """Configure global tracer provider once. Returns tracing enabled state."""
    global _INITIALIZED
    global _ENABLED

    if _INITIALIZED:
        return _ENABLED

    _ENABLED = is_enabled()
    if not _ENABLED:
        _INITIALIZED = True
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": "tripplanner"}),
    )
    exporter_kind = os.getenv("TRIPPLANNER_TRACING_EXPORTER", "console").strip().lower()
    if exporter_kind == "otlp":
        endpoint = os.getenv(
            "TRIPPLANNER_OTLP_ENDPOINT",
            "http://localhost:4318/v1/traces",
        )
        timeout_ms = int(os.getenv("TRIPPLANNER_OTLP_TIMEOUT_MS", "1000"))
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=timeout_ms / 1000)
    else:
        exporter = ConsoleSpanExporter(out=sys.stderr)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _INITIALIZED = True
    return True


def start_span(name: str) -> ContextManager[object]:
    """Start span when tracing is enabled; no-op otherwise."""
    if not configure_telemetry():
        return nullcontext()
    tracer = trace.get_tracer("tripplanner")
    return tracer.start_as_current_span(name)
