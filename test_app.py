from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_errors(pipeline_output, monkeypatch):
    monkeypatch.setenv("AML_OUT_DIR", str(pipeline_output))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=60)
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert any("Финансовый обзор" in subheader.value for subheader in app.subheader)
    assert not any(
        list(widget.options) == ["Midnight Signal", "Capital Ivory", "Electric Market"]
        for widget in app.selectbox
    )
    tab_labels = [tab.label for tab in app.tabs]
    for expected_label in (
        "Обзор", "Клиенты", "Карта переводов", "Очередь", "Группы",
        "Сигналы", "Помощник", "Документы", "О методике",
    ):
        assert expected_label in tab_labels


def test_theme_modes_persist_in_query_params(pipeline_output, monkeypatch):
    monkeypatch.setenv("AML_OUT_DIR", str(pipeline_output))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py")
    app.query_params["theme"] = "dark"
    app.run(timeout=60)

    theme = next(widget for widget in app.radio if widget.label == "Тема")
    assert theme.value == "Тёмная"
    assert len(app.exception) == 0

    theme.set_value("Светлая").run(timeout=60)
    assert dict(app.query_params)["theme"] == ["light"]
    assert len(app.exception) == 0

    theme = next(widget for widget in app.radio if widget.label == "Тема")
    theme.set_value("Системная").run(timeout=60)
    assert dict(app.query_params)["theme"] == ["system"]
    assert len(app.exception) == 0
