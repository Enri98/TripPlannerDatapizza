"""OpenTelemetry setup and helper utilities."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from typing import Any, ContextManager, Sequence, TextIO

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)

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
        console_mode = os.getenv("TRIPPLANNER_TRACING_CONSOLE_MODE", "compact").strip().lower()
        if console_mode == "raw":
            exporter = ConsoleSpanExporter(out=sys.stderr)
        else:
            exporter = _CompactConsoleSpanExporter(out=sys.stderr)
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


class _CompactConsoleSpanExporter(SpanExporter):
    """Console exporter with concise one-line span summaries."""

    _INTERESTING_ATTRS = (
        "type",
        "model_name",
        "stop_reason",
        "prompt_tokens_used",
        "completion_tokens_used",
        "demo.execution_status",
        "demo.completed_tasks",
        "demo.legs",
        "demo.cache_key",
    )

    def __init__(self, out: TextIO) -> None:
        self._out = out

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                self._out.write(_format_span_line(span, self._INTERESTING_ATTRS) + "\n")
            except ValueError:
                # Stream may be closed during process shutdown under capture.
                return SpanExportResult.SUCCESS
        try:
            self._out.flush()
        except ValueError:
            return SpanExportResult.SUCCESS
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _format_span_line(span: ReadableSpan, attrs_whitelist: tuple[str, ...]) -> str:
    duration_ms = max(0.0, (span.end_time - span.start_time) / 1_000_000)
    status_code_name = span.status.status_code.name
    status_text = "ERR" if status_code_name == "ERROR" else "OK"
    attr_bits: list[str] = []
    for key in attrs_whitelist:
        if key in span.attributes:
            value = _safe_text(span.attributes[key])
            if value:
                attr_bits.append(f"{key}={value}")
    if span.status.description:
        attr_bits.append(f"error={_safe_text(span.status.description)}")
    attrs_joined = " | ".join(attr_bits[:5])
    if attrs_joined:
        return f"[trace] {span.name} | {duration_ms:.1f}ms | {status_text} | {attrs_joined}"
    return f"[trace] {span.name} | {duration_ms:.1f}ms | {status_text}"


def _safe_text(value: Any, max_len: int = 80) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text
