from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from opentelemetry.sdk.trace.export import SpanExportResult

from tripplanner.telemetry import _CompactConsoleSpanExporter, _format_span_line


def _fake_span(
    *,
    name: str = "agent.geo",
    start_ns: int = 0,
    end_ns: int = 12_300_000,
    status_name: str = "UNSET",
    status_description: str | None = None,
    attributes: dict | None = None,
):
    return SimpleNamespace(
        name=name,
        start_time=start_ns,
        end_time=end_ns,
        status=SimpleNamespace(
            status_code=SimpleNamespace(name=status_name),
            description=status_description,
        ),
        attributes=attributes or {},
    )


def test_format_span_line_compact_includes_main_info() -> None:
    span = _fake_span(
        attributes={
            "type": "tool",
            "model_name": "gemini-2.5-flash",
            "completion_tokens_used": 42,
        }
    )
    line = _format_span_line(
        span, ("type", "model_name", "completion_tokens_used", "demo.execution_status")
    )
    assert line.startswith("[trace] agent.geo | 12.3ms | OK")
    assert "type=tool" in line
    assert "model_name=gemini-2.5-flash" in line
    assert "completion_tokens_used=42" in line


def test_format_span_line_error_status_and_description() -> None:
    span = _fake_span(
        status_name="ERROR",
        status_description="HTTP Error 504: Gateway Timeout",
    )
    line = _format_span_line(span, ("type",))
    assert "| ERR |" in line
    assert "error=HTTP Error 504: Gateway Timeout" in line


def test_compact_exporter_ignores_closed_stream() -> None:
    stream = StringIO()
    exporter = _CompactConsoleSpanExporter(out=stream)
    stream.close()
    result = exporter.export([_fake_span()])
    assert result == SpanExportResult.SUCCESS
