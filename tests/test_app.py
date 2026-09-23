from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_errors(pipeline_output, monkeypatch):
    monkeypatch.setenv("AML_OUT_DIR", str(pipeline_output))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=60)
    assert len(app.exception) == 0
    assert len(app.error) == 0
