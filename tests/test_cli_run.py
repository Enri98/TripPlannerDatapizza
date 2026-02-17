from __future__ import annotations

import json

from tripplanner import cli


def test_cli_run_command_invokes_real_pipeline(monkeypatch, capsys) -> None:
    def fake_run_real_pipeline(query: str, **kwargs):  # type: ignore[no-untyped-def]
        assert query == "Plan a trip to Rome"
        assert kwargs["timezone_name"] == "Europe/Rome"
        assert kwargs["output_language"] == "en"
        return {"status": "completed", "title": "ok"}

    monkeypatch.setattr(cli, "run_real_pipeline", fake_run_real_pipeline)
    exit_code = cli.main(
        ["run", "Plan a trip to Rome", "--timezone", "Europe/Rome", "--output-language", "en"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "completed"
