from __future__ import annotations

from typing import Any


THEME_LABELS = {
    "Светлая": "light",
    "Тёмная": "dark",
    "Системная": "system",
}
THEME_BY_SLUG = {slug: label for label, slug in THEME_LABELS.items()}

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#F4F7F5",
        "surface": "#FFFFFF",
        "surface_raised": "#FFFFFF",
        "surface2": "#EDF5F0",
        "surface3": "#E4EEE8",
        "text": "#17211B",
        "muted": "#64736B",
        "accent": "#119D5A",
        "on_accent": "#FFFFFF",
        "accent_hover": "#0B8249",
        "accent_soft": "#E3F5EB",
        "accent_text": "#087442",
        "accent2": "#557DDF",
        "border": "#D9E4DD",
        "shadow": "rgba(26, 60, 41, .08)",
        "graph_bg": "#FFFFFF",
        "graph_text": "#23312A",
        "edge": "#94AA9E",
        "grid": "#E5ECE8",
        "plot": "plotly_white",
        "bank_body": "#EEF4F0",
        "bank_column": "#D5E1DA",
    },
    "dark": {
        "bg": "#101713",
        "surface": "#17211B",
        "surface_raised": "#1C2821",
        "surface2": "#223128",
        "surface3": "#2A3A31",
        "text": "#F2F7F4",
        "muted": "#A7B7AE",
        "accent": "#2DD27F",
        "on_accent": "#082517",
        "accent_hover": "#49E394",
        "accent_soft": "#193B2A",
        "accent_text": "#75E8AA",
        "accent2": "#80A4FF",
        "border": "#304239",
        "shadow": "rgba(0, 0, 0, .24)",
        "graph_bg": "#17211B",
        "graph_text": "#EAF3EE",
        "edge": "#668174",
        "grid": "#2B3B32",
        "plot": "plotly_dark",
        "bank_body": "#2A3A31",
        "bank_column": "#3B5145",
    },
}


def _context_theme(streamlit: Any) -> str:
    try:
        detected = streamlit.context.theme.get("type")
    except (AttributeError, KeyError, TypeError):
        detected = None
    return detected if detected in THEMES else "light"


def theme_selector(streamlit: Any) -> tuple[str, str, dict[str, str]]:
    """Render a persistent selector and return mode, resolved mode and tokens."""
    query_slug = str(streamlit.query_params.get("theme", "system")).lower()
    if query_slug not in THEME_BY_SLUG:
        query_slug = "system"

    if streamlit.session_state.get("_theme_url_seen") != query_slug:
        streamlit.session_state["theme_choice"] = THEME_BY_SLUG[query_slug]
        streamlit.session_state["_theme_url_seen"] = query_slug

    def persist_theme() -> None:
        selected_slug = THEME_LABELS[streamlit.session_state["theme_choice"]]
        streamlit.session_state["_theme_url_seen"] = selected_slug
        streamlit.query_params["theme"] = selected_slug

    streamlit.sidebar.radio(
        "Тема",
        list(THEME_LABELS),
        horizontal=True,
        key="theme_choice",
        on_change=persist_theme,
        help="Системная тема следует настройке оформления браузера.",
    )
    mode = THEME_LABELS[streamlit.session_state["theme_choice"]]
    resolved = _context_theme(streamlit) if mode == "system" else mode
    return mode, resolved, THEMES[resolved].copy()


def plot_layout(tokens: dict[str, str]) -> dict[str, Any]:
    return {
        "paper_bgcolor": tokens["surface"],
        "plot_bgcolor": tokens["surface"],
        "font": {"color": tokens["text"], "family": "Inter, Segoe UI, Arial"},
        "title": {"font": {"color": tokens["text"], "size": 18}},
        "xaxis": {"gridcolor": tokens["grid"], "zerolinecolor": tokens["grid"]},
        "yaxis": {"gridcolor": tokens["grid"], "zerolinecolor": tokens["grid"]},
        "hoverlabel": {"bgcolor": tokens["surface_raised"], "font_color": tokens["text"]},
    }
