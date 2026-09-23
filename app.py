from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
from pyvis.network import Network

from src.agent import AMLAnalystAgent
from src.documents import CaseDocumentStore


OUT = Path(os.getenv("AML_OUT_DIR", "out"))
DOCUMENT_DIR = Path(os.getenv("AML_DOCUMENT_DIR", "case_documents"))
ROLE_COLORS = {
    "consolidator": "#8b5cf6", "transit": "#06b6d4", "distributor": "#f59e0b",
    "terminal": "#ef4444", "coordinator": "#10b981", "peripheral": "#64748b",
}

THEMES = {
    "Midnight Signal": {
        "bg": "#07111A", "surface": "#0D1B26", "surface2": "#132635",
        "text": "#EAF2F6", "muted": "#93A8B6", "accent": "#21D4B4",
        "accent2": "#FFB547", "border": "#20394A", "glow": "rgba(33,212,180,.18)",
        "graph_bg": "#08131E", "graph_text": "#D8E2EC", "edge": "#486477", "plot": "plotly_dark",
    },
    "Capital Ivory": {
        "bg": "#F4F1E8", "surface": "#FFFFFF", "surface2": "#EBE7DB",
        "text": "#14201D", "muted": "#65716D", "accent": "#E9A51A",
        "accent2": "#087F6E", "border": "#D8D4C8", "glow": "rgba(233,165,26,.20)",
        "graph_bg": "#FAF8F1", "graph_text": "#1D2B27", "edge": "#8A9A94", "plot": "plotly_white",
    },
    "Electric Market": {
        "bg": "#0C100D", "surface": "#151B16", "surface2": "#202920",
        "text": "#F3F8EF", "muted": "#A4B0A1", "accent": "#B8F136",
        "accent2": "#68E4FF", "border": "#303C2D", "glow": "rgba(184,241,54,.18)",
        "graph_bg": "#101510", "graph_text": "#EEF6E9", "edge": "#52614E", "plot": "plotly_dark",
    },
}


def configured_openai_api_key() -> str:
    """Load a server-side API key without exposing it in the browser."""
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError, AttributeError):
        return ""


st.set_page_config(
    page_title="AML Graph Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded"
)
st.sidebar.markdown(
    '<div class="side-brand"><span>◈</span><div><b>MONEYGRAPH</b><small>AML intelligence studio</small></div></div>',
    unsafe_allow_html=True,
)
theme_name = st.sidebar.selectbox("Интерфейс", list(THEMES), index=0)
THEME = THEMES[theme_name]
PLOT_TEMPLATE = THEME["plot"]
NETWORK_BG = THEME["graph_bg"]
NETWORK_TEXT = THEME["graph_text"]
EDGE_COLOR = THEME["edge"]
st.sidebar.markdown(
    '<div class="system-status"><span class="status-dot"></span><div><b>Система готова</b><small>Локальный расчёт · explainable AI</small></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style>
    :root{{--mg-bg:{THEME['bg']};--mg-surface:{THEME['surface']};--mg-surface-2:{THEME['surface2']};--mg-text:{THEME['text']};--mg-muted:{THEME['muted']};--mg-accent:{THEME['accent']};--mg-accent-2:{THEME['accent2']};--mg-border:{THEME['border']};--mg-glow:{THEME['glow']};}}
    html,body,[class*="css"]{{font-family:Inter,"Segoe UI",Arial,sans-serif}}
    .stApp{{background:var(--mg-bg);color:var(--mg-text)}}
    .stApp::before{{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 88% 8%,var(--mg-glow),transparent 25%),radial-gradient(circle at 3% 70%,color-mix(in srgb,var(--mg-accent-2) 10%,transparent),transparent 26%)}}
    .block-container{{padding-top:1.35rem;max-width:1500px;padding-bottom:4rem}}header[data-testid="stHeader"]{{background:transparent}}
    [data-testid="stSidebar"]{{background:color-mix(in srgb,var(--mg-surface) 94%,transparent);border-right:1px solid var(--mg-border)}}
    .side-brand{{display:flex;gap:.8rem;align-items:center;padding:.45rem .2rem 1.25rem;color:var(--mg-text)}}
    .side-brand>span{{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:var(--mg-accent);color:#07110d;font-size:1.25rem;box-shadow:0 10px 28px var(--mg-glow)}}
    .side-brand b{{font-size:.86rem;letter-spacing:.12em}}.side-brand small,.system-status small{{display:block;color:var(--mg-muted);font-size:.69rem;margin-top:.12rem}}
    .system-status{{display:flex;align-items:center;gap:.65rem;margin-top:1rem;padding:.8rem;border:1px solid var(--mg-border);border-radius:14px;background:var(--mg-surface-2);color:var(--mg-text)}}
    .status-dot{{width:9px;height:9px;border-radius:50%;background:var(--mg-accent);box-shadow:0 0 0 5px var(--mg-glow)}}
    .hero-shell{{position:relative;overflow:hidden;border:1px solid var(--mg-border);border-radius:30px;padding:clamp(1.5rem,4vw,3.8rem);margin:.2rem 0 1.35rem;background:linear-gradient(135deg,color-mix(in srgb,var(--mg-surface) 97%,transparent),color-mix(in srgb,var(--mg-surface-2) 88%,transparent));box-shadow:0 24px 80px rgba(0,0,0,.14)}}
    .hero-shell::after{{content:"";position:absolute;width:420px;height:420px;right:-150px;top:-210px;border:1px solid var(--mg-accent);border-radius:50%;opacity:.22;box-shadow:0 0 0 54px color-mix(in srgb,var(--mg-accent) 7%,transparent),0 0 0 110px color-mix(in srgb,var(--mg-accent) 4%,transparent)}}
    .hero-grid{{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1.4fr) minmax(280px,.6fr);gap:2.5rem;align-items:end}}
    .eyebrow{{display:flex;align-items:center;gap:.55rem;color:var(--mg-accent);font-size:.72rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:1.1rem}}.eyebrow::before{{content:"";width:28px;height:2px;background:var(--mg-accent)}}
    .hero-title{{margin:0;color:var(--mg-text);font-size:clamp(2.35rem,5vw,5.2rem);line-height:.94;letter-spacing:-.055em;font-weight:720;max-width:950px}}.hero-title span{{color:var(--mg-accent)}}
    .hero-copy{{max-width:740px;color:var(--mg-muted);font-size:1.02rem;line-height:1.65;margin:1.35rem 0 0}}.hero-chips{{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.5rem}}
    .hero-chip{{padding:.48rem .72rem;border-radius:999px;border:1px solid var(--mg-border);background:color-mix(in srgb,var(--mg-surface-2) 82%,transparent);font-size:.72rem;color:var(--mg-text)}}
    .signal-card{{position:relative;border:1px solid color-mix(in srgb,var(--mg-accent) 42%,var(--mg-border));border-radius:22px;padding:1.35rem;background:color-mix(in srgb,var(--mg-bg) 58%,transparent);backdrop-filter:blur(18px)}}
    .signal-head{{display:flex;justify-content:space-between;color:var(--mg-muted);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase}}.signal-live{{color:var(--mg-accent)}}
    .signal-score{{font-size:3.6rem;line-height:1;color:var(--mg-text);letter-spacing:-.07em;margin:1rem 0 .35rem;font-weight:700}}.signal-label{{color:var(--mg-muted);font-size:.76rem}}.signal-gid{{color:var(--mg-text);font-family:"Cascadia Code",monospace;font-size:.88rem;margin:.9rem 0}}.signal-role{{display:inline-flex;padding:.38rem .62rem;border-radius:9px;background:var(--mg-accent);color:#07110d;font-weight:800;font-size:.72rem;text-transform:uppercase}}
    .insight-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin:0 0 1.35rem}}.insight-item{{border:1px solid var(--mg-border);border-radius:17px;background:var(--mg-surface);padding:1rem 1.1rem;color:var(--mg-text)}}.insight-item small{{display:block;color:var(--mg-muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.35rem}}.insight-item b{{font-size:1.05rem}}
    [data-testid="stMetric"]{{background:var(--mg-surface);border:1px solid var(--mg-border);padding:16px 17px;border-radius:18px;box-shadow:0 8px 32px rgba(0,0,0,.06);transition:transform .2s ease,border-color .2s ease}}[data-testid="stMetric"]:hover{{transform:translateY(-3px);border-color:var(--mg-accent)}}[data-testid="stMetricLabel"]{{color:var(--mg-muted)}}[data-testid="stMetricValue"]{{color:var(--mg-text);letter-spacing:-.035em;font-size:clamp(1.55rem,2.15vw,2.25rem)}}
    div[data-testid="stDataFrame"]{{border:1px solid var(--mg-border);border-radius:16px;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,.06)}}
    .stTabs [data-baseweb="tab-list"]{{gap:.3rem;background:var(--mg-surface);border:1px solid var(--mg-border);border-radius:16px;padding:.38rem;overflow-x:auto}}.stTabs [data-baseweb="tab"]{{height:42px;border-radius:11px;color:var(--mg-muted);padding:0 .9rem;white-space:nowrap}}.stTabs [aria-selected="true"]{{background:var(--mg-accent)!important;color:#07110d!important;font-weight:750}}.stTabs [data-baseweb="tab-highlight"]{{display:none}}
    .stButton>button[kind="primary"]{{border:0;border-radius:14px;background:var(--mg-accent);color:#07110d;font-weight:800;box-shadow:0 12px 30px var(--mg-glow)}}.stButton>button:not([kind="primary"]){{border-radius:14px;border-color:var(--mg-border);background:var(--mg-surface);color:var(--mg-text)}}
    div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea{{background:var(--mg-surface)!important;border-color:var(--mg-border)!important;color:var(--mg-text)!important;border-radius:12px!important}}
    .node-card{{background:linear-gradient(135deg,var(--mg-surface),var(--mg-surface-2));border:1px solid var(--mg-border);border-left:4px solid var(--mg-accent);padding:18px;border-radius:16px;color:var(--mg-text);box-shadow:0 12px 36px rgba(0,0,0,.08)}}
    .risk{{color:var(--mg-accent);letter-spacing:.14em;font-weight:800}}h1,h2,h3,p,label{{color:var(--mg-text)}}.section-kicker{{color:var(--mg-accent);font-size:.7rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;margin-top:.35rem}}
    @media(max-width:900px){{.hero-grid{{grid-template-columns:1fr}}.signal-card{{max-width:420px}}.insight-strip{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:560px){{.hero-shell{{border-radius:22px;padding:1.3rem}}.hero-title{{font-size:2.45rem}}}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = [OUT / "node_features.parquet", OUT / "graph_edges.parquet", OUT / "clusters.csv", OUT / "metadata.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    features = pd.read_parquet(required[0])
    edges = pd.read_parquet(required[1])
    clusters = pd.read_csv(required[2])
    top = pd.read_csv(OUT / "top_nodes.csv")
    metadata = json.loads(required[3].read_text(encoding="utf-8"))
    return features, edges, clusters, top, metadata


@st.cache_resource
def load_agent() -> AMLAnalystAgent:
    return AMLAnalystAgent(OUT, document_dir=DOCUMENT_DIR)


def kzt(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f} млрд ₸"
    if value >= 1e6:
        return f"{value / 1e6:.2f} млн ₸"
    return f"{value:,.0f} ₸"


def render_hero(metadata: dict, top_nodes: pd.DataFrame, cluster_count: int) -> None:
    eda = metadata["eda"]
    leader = top_nodes.iloc[0]
    gid = html.escape(str(leader.gid))
    role = html.escape(str(leader.role))
    runtime = float(metadata["runtime_seconds"])
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Financial network intelligence</div>
              <h1 class="hero-title">Видеть структуру денег.<br><span>Раньше риска.</span></h1>
              <p class="hero-copy">Объяснимая карта финансовой сети превращает 2 248 клиентов в управляемую очередь проверки: роли, потоки, кластеры, аномалии и следующий запрос — в одном аналитическом пространстве.</p>
              <div class="hero-chips">
                <span class="hero-chip">Directed cash flow</span><span class="hero-chip">Explainable roles</span>
                <span class="hero-chip">Agentic AML</span><span class="hero-chip">Local-first</span>
              </div>
            </div>
            <div class="signal-card">
              <div class="signal-head"><span>Приоритет проверки</span><span class="signal-live">● LIVE</span></div>
              <div class="signal-score">{float(leader.priority_score):.3f}</div>
              <div class="signal-label">максимальный priority score</div>
              <div class="signal-gid">{gid}</div><span class="signal-role">{role}</span>
            </div>
          </div>
        </section>
        <div class="insight-strip">
          <div class="insight-item"><small>Наблюдаемая сеть</small><b>{int(eda['nodes']):,} клиентов</b></div>
          <div class="insight-item"><small>Денежные связи</small><b>{int(eda['edges_rows']):,} рёбер</b></div>
          <div class="insight-item"><small>Структура</small><b>{cluster_count} кластеров</b></div>
          <div class="insight-item"><small>Полный расчёт</small><b>{runtime:.2f} секунды</b></div>
        </div>
        """.replace(",", " "),
        unsafe_allow_html=True,
    )


def network_html(node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> str:
    allowed = set(node_frame["gid"].astype(str))
    network = Network(height="650px", width="100%", bgcolor=NETWORK_BG, font_color=NETWORK_TEXT, directed=True)
    network.set_options(json.dumps({
        "nodes": {"borderWidth": 1, "font": {"size": 13}},
        "edges": {
            "color": {"color": EDGE_COLOR, "opacity": 0.65},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
            "smooth": {"type": "dynamic"},
        },
        "physics": {
            "barnesHut": {"gravitationalConstant": -7000, "springLength": 120},
            "stabilization": {"iterations": 180},
        },
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True},
    }))
    for row in node_frame.itertuples(index=False):
        title = (f"GID: {row.gid}<br>Роль: {row.role}<br>Priority: {row.priority_score:.3f}"
                 f"<br>Кластер: {row.cluster_id}<br>{row.evidence}")
        network.add_node(str(row.gid), label=str(row.gid), title=title, color=ROLE_COLORS.get(row.role, "#64748b"),
                         size=12 + 20 * float(row.priority_score))
    visible = edge_frame[edge_frame["source"].astype(str).isin(allowed) & edge_frame["target"].astype(str).isin(allowed)]
    max_amount = max(float(visible["sum_kzt"].max()) if len(visible) else 1, 1)
    for row in visible.itertuples(index=False):
        network.add_edge(str(row.source), str(row.target), value=1 + 7 * float(row.sum_kzt) / max_amount,
                         title=f"{kzt(float(row.sum_kzt))}; tx={int(row.n_tx)}")
    return network.generate_html(notebook=False)


def network_uri(node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> str:
    encoded = base64.b64encode(network_html(node_frame, edge_frame).encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded}"


try:
    features, edges, clusters, top_nodes, metadata = load_outputs()
except FileNotFoundError as exc:
    from src.pipeline import run_pipeline

    st.markdown('<div class="risk">◈ AML GRAPH INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Подготовка данных")
    st.info("Загрузите три исходных parquet-файла. Они будут обработаны локально и не отправляются во внешние сервисы.")

    data_dir = Path("data")
    input_names = ("nodes.parquet", "edges.parquet", "transactions.parquet")
    existing = {name: (data_dir / name).is_file() for name in input_names}
    status_columns = st.columns(3)
    for column, name in zip(status_columns, input_names):
        column.metric(name, "Найден" if existing[name] else "Не загружен")

    uploads = {
        name: st.file_uploader(name, type=["parquet"], key=f"upload-{name}")
        for name in input_names if not existing[name]
    }
    ready = all(existing[name] or uploads.get(name) is not None for name in input_names)

    if st.button("Построить AML-граф", type="primary", disabled=not ready, width="stretch"):
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, uploaded in uploads.items():
            if uploaded is not None:
                (data_dir / name).write_bytes(uploaded.getvalue())
        try:
            progress = st.status("Выполняется pipeline…", expanded=True)
            messages: list[str] = []

            def show_stage(message: str) -> None:
                messages.append(message)
                progress.write(message)

            run_pipeline(data_dir, OUT, logger=show_stage)
            progress.update(label="Pipeline завершён", state="complete")
            load_outputs.clear()
            load_agent.clear()
            st.success("Данные обработаны. Открываю dashboard…")
            st.rerun()
        except Exception as pipeline_error:
            st.error(f"Pipeline остановлен: {pipeline_error}")

    st.caption("Альтернативный запуск: `py -3 run_pipeline.py --data ./data --out ./out`")
    st.caption(f"Результаты пока отсутствуют: {exc}")
    st.stop()

render_hero(metadata, top_nodes, len(clusters))
st.sidebar.divider()
st.sidebar.caption("КОНТЕКСТ АНАЛИЗА")
st.sidebar.metric("Период", "Июль 2026")
st.sidebar.metric("Оборот сети", kzt(float(metadata["eda"]["edge_turnover_kzt"])))
st.sidebar.caption("Данные обезличены · роли являются аналитическими гипотезами")

overview, search, network_tab, priority_tab, cluster_tab, patterns_tab, agent_tab, documents_tab, method_tab = st.tabs(
    ["Обзор", "Поиск GID", "Граф", "Приоритет", "Кластеры", "AML-паттерны", "AI-аналитик", "Документы", "Методология"]
)

with overview:
    st.markdown('<div class="section-kicker">Executive overview</div>', unsafe_allow_html=True)
    st.subheader("Сеть в цифрах")
    eda = metadata["eda"]
    metrics = st.columns(6)
    metrics[0].metric("Узлы", f"{eda['nodes']:,}")
    metrics[1].metric("Связи", f"{len(edges):,}")
    metrics[2].metric("Транзакции", f"{eda['transaction_rows']:,}")
    metrics[3].metric("Оборот", kzt(float(eda["edge_turnover_kzt"])))
    metrics[4].metric("Seed", f"{eda['seed_nodes']:,}")
    metrics[5].metric("Кластеры", f"{len(clusters):,}")
    left, right = st.columns(2)
    with left:
        role_counts = features["role"].value_counts().rename_axis("role").reset_index(name="nodes")
        fig = px.bar(role_counts, x="role", y="nodes", color="role", color_discrete_map=ROLE_COLORS,
                     title="Распределение ролей", template=PLOT_TEMPLATE)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(features, x="betweenness_pct", y="priority_score", color="role", size="pagerank_pct",
                         hover_name="gid", color_discrete_map=ROLE_COLORS, title="Структурная значимость",
                         template=PLOT_TEMPLATE)
        st.plotly_chart(fig, width="stretch")
    resilience_file = OUT / "resilience.csv"
    if resilience_file.exists():
        resilience = pd.read_csv(resilience_file)
        st.subheader("Устойчивость сети")
        st.line_chart(resilience.set_index("removed_top_n")[["largest_component_share", "fragmentation"]])

with search:
    st.markdown('<div class="section-kicker">Node intelligence</div>', unsafe_allow_html=True)
    st.subheader("Карточка клиента")
    query = st.text_input("GID", placeholder="Введите точный gid")
    if query:
        matched = features[features["gid"].astype(str).eq(query.strip())]
        if matched.empty:
            st.warning("GID не найден")
        else:
            row = matched.iloc[0]
            st.markdown(f'<div class="node-card"><b>{row.gid}</b> · {row.role}<br>{row.evidence}</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            for col, label, value in zip(cols, ["Role score", "Priority", "Кластер", "Depth", "In degree", "Out degree"],
                                         [row.role_score, row.priority_score, int(row.cluster_id), row.depth, int(row.in_deg), int(row.out_deg)]):
                col.metric(label, f"{value:.3f}" if isinstance(value, float) else value)
            st.dataframe(pd.DataFrame({"Метрика": ["Входящий поток", "Исходящий поток", "PageRank", "Betweenness", "Seed", "Усечение depth"],
                                      "Значение": [kzt(row.in_kzt), kzt(row.out_kzt), f"{row.pagerank:.6f}",
                                                   f"{row.betweenness:.6f}", bool(row.is_seed), bool(row.truncated_by_depth)]}),
                         hide_index=True, width="stretch")
            st.caption(
                f"Cycles: {int(row.cycle_count)} · Recurring edges: {int(row.recurring_route_count)} · "
                f"A→B→C chains: {int(row.recurring_chain_count)} · AML anomaly: {row.aml_anomaly_score:.3f}"
            )
            st.subheader("Полнота данных и следующий запрос")
            st.progress(float(row.completeness_score), text=f"Наблюдаемая полнота: {row.completeness_score:.0%}")
            st.write(f"**Белые пятна:** {row.observed_gaps}")
            st.write(f"**Дальнейшее действие:** {row.recommended_request}")

with network_tab:
    st.markdown('<div class="section-kicker">Interactive topology</div>', unsafe_allow_html=True)
    st.subheader("Карта движения денег")
    mode = st.radio("Режим", ["Top-risk", "Ego-network", "Кластер"], horizontal=True)
    if mode == "Top-risk":
        count = st.slider("Число узлов", 20, min(150, len(features)), min(80, len(features)), 10)
        shown = features.nlargest(count, "priority_score")
    elif mode == "Ego-network":
        selected = st.selectbox("Центральный GID", features.sort_values("priority_score", ascending=False)["gid"])
        radius = st.slider("Радиус", 1, 3, 1)
        graph = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
        graph.add_nodes_from(features["gid"].astype(str))
        ids = list(nx.ego_graph(graph, str(selected), radius=radius, undirected=True).nodes)
        shown = features[features["gid"].astype(str).isin(ids)].nlargest(250, "priority_score")
    else:
        selected_cluster = st.selectbox("Кластер", clusters["cluster_id"].tolist())
        shown = features[features["cluster_id"].eq(selected_cluster)].nlargest(250, "priority_score")
    legend = " · ".join(f"<span style='color:{color}'>●</span> {role}" for role, color in ROLE_COLORS.items())
    st.markdown(legend, unsafe_allow_html=True)
    st.iframe(network_uri(shown, edges), height=670)

with priority_tab:
    st.markdown('<div class="section-kicker">Review queue</div>', unsafe_allow_html=True)
    st.subheader("Кого смотреть первым")
    limit = st.radio("Показать", [20, 50], horizontal=True)
    display = features.nlargest(limit, "priority_score").copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    st.dataframe(display[["rank", "gid", "role", "priority_score", "cluster_id", "evidence"]],
                 hide_index=True, width="stretch", height=720)

with cluster_tab:
    st.markdown('<div class="section-kicker">Community intelligence</div>', unsafe_allow_html=True)
    st.subheader("Структура кластеров")
    cluster_id = st.selectbox("Выберите cluster_id", clusters["cluster_id"].tolist(), key="cluster-explorer")
    cluster = clusters.loc[clusters["cluster_id"].eq(cluster_id)].iloc[0]
    cols = st.columns(3)
    cols[0].metric("Узлы", int(cluster.n_nodes))
    cols[1].metric("Seed", int(cluster.n_seed))
    cols[2].metric("Внутренний оборот", kzt(cluster.sum_kzt_internal))
    st.info(cluster.hypothesis)
    subset = features[features["cluster_id"].eq(cluster_id)]
    counts = subset["role"].value_counts().rename_axis("role").reset_index(name="nodes")
    st.plotly_chart(px.bar(counts, x="role", y="nodes", color="role", color_discrete_map=ROLE_COLORS,
                           template=PLOT_TEMPLATE), width="stretch")
    st.dataframe(subset.nlargest(10, "priority_score")[["gid", "role", "priority_score", "evidence"]], hide_index=True)

with patterns_tab:
    st.markdown('<div class="section-kicker">Behavioral signals</div>', unsafe_allow_html=True)
    st.subheader("AML-паттерны")
    cycles = pd.read_csv(OUT / "cycles.csv")
    recurring = pd.read_csv(OUT / "recurring_routes.csv")
    chains = pd.read_csv(OUT / "recurring_chains.csv")
    synchronous = pd.read_csv(OUT / "synchronous_inflows.csv")
    structuring = pd.read_csv(OUT / "structuring_events.csv")
    anomalies = pd.read_csv(OUT / "anomalies.csv")
    metrics = st.columns(6)
    metrics[0].metric("Циклы 2–6", f"{len(cycles):,}")
    metrics[1].metric("Повторные связи", f"{len(recurring):,}")
    metrics[2].metric("Цепочки A→B→C", f"{len(chains):,}")
    metrics[3].metric("Синхронные входы", f"{len(synchronous):,}")
    metrics[4].metric("Дробление", f"{len(structuring):,}")
    metrics[5].metric("Аномальные узлы", f"{len(anomalies):,}")
    cycle_view, route_view, chain_view, sync_view, structuring_view, anomaly_view = st.tabs(
        ["Циклические потоки", "Recurring edges", "Цепочки A→B→C", "Синхронные входы", "Дробление", "AML anomalies"]
    )

    with cycle_view:
        st.caption("Направленные циклы длиной 2–6; score учитывает bottleneck-сумму, длину и наличие seed.")
        if cycles.empty:
            st.info("Циклы заданной длины не обнаружены.")
        else:
            st.dataframe(cycles.head(100), hide_index=True, width="stretch", height=420)
            selected_cycle = st.selectbox("Показать цикл", cycles["cycle_id"].tolist())
            cycle_row = cycles.loc[cycles["cycle_id"].eq(selected_cycle)].iloc[0]
            cycle_gids = str(cycle_row.gids).split(" → ")[:-1]
            cycle_nodes = features[features["gid"].astype(str).isin(cycle_gids)]
            st.iframe(network_uri(cycle_nodes, edges), height=500)

    with route_view:
        st.caption("Маршруты с повторениями в разные дни; порог определяется 75-м percentile среди многодневных связей.")
        if recurring.empty:
            st.info("Повторяющиеся маршруты не обнаружены.")
        else:
            fig = px.scatter(
                recurring, x="active_days", y="recurrence_score", size="sum_kzt", color="cadence_regularity",
                hover_data=["source", "target", "n_tx"], template=PLOT_TEMPLATE,
                title="Регулярность маршрутов",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(recurring.head(100), hide_index=True, width="stretch", height=420)

    with chain_view:
        st.caption("Устойчивые двухзвенные маршруты A→B→C, повторяющиеся минимум в два разных дня.")
        st.dataframe(chains.head(100), hide_index=True, width="stretch", height=480)

    with sync_view:
        st.caption("Дни, когда не менее трёх разных плательщиков направляли средства одному получателю.")
        st.dataframe(synchronous.head(100), hide_index=True, width="stretch", height=480)

    with structuring_view:
        st.caption(
            "Объяснимые сигналы дробления выше наблюдаемого порога 5 000 KZT: несколько операций и контрагентов, "
            "сходные, округлённые или близкие к порогу суммы. Это гипотеза для проверки."
        )
        st.dataframe(structuring.head(100), hide_index=True, width="stretch", height=480)

    with anomaly_view:
        st.caption(
            "TOP-5% composite: временные сигналы, синхронные входы, дробление, recurring chains, cycles "
            "и отклонение от профиля своего depth."
        )
        st.dataframe(anomalies, hide_index=True, width="stretch", height=600)

with agent_tab:
    st.markdown('<div class="section-kicker">Agentic investigation</div>', unsafe_allow_html=True)
    st.subheader("AI AML Analyst")
    st.caption(
        "Отвечает по рассчитанным артефактам pipeline и локальному досье, всегда показывает источники "
        "и отделяет непроверенный текст документов от графовых фактов."
    )
    mode = st.radio("Режим агента", ["Локальный evidence-agent", "OpenAI Agents SDK"], horizontal=True)
    preset = st.selectbox(
        "Вопрос",
        [
            "Кого из 2 248 клиентов смотреть первым и почему?",
            "Какие узлы имеют наиболее сильные аномальные паттерны?",
            "Покажи главные циклы и повторяющиеся маршруты",
            "Другой вопрос",
        ],
    )
    question = preset
    if preset == "Другой вопрос":
        question = st.text_area(
            "Вопрос аналитику",
            placeholder="Например: почему GID 100000003684369100 стоит проверить первым?",
        )

    use_openai = mode == "OpenAI Agents SDK"
    api_key = configured_openai_api_key() if use_openai else ""
    model = "gpt-6-astra"
    consent = True
    if use_openai:
        st.warning(
            "OpenAI-режим отправляет вопрос и минимальный контекст выбранных узлов или найденный фрагмент документа в OpenAI API. "
            "Серверный ключ не передаётся в браузер, ответы остаются grounded на read-only инструментах."
        )
        if api_key:
            st.success("API key настроен локально на сервере.")
        else:
            api_key = st.text_input(
                "OPENAI_API_KEY",
                type="password",
                help="Резервный ввод только для текущей сессии. Для постоянной работы используйте окружение или .streamlit/secrets.toml.",
            )
        model = st.text_input("Модель", value="gpt-6-astra")
        consent = st.checkbox("Разрешаю передать выбранный аналитический контекст в OpenAI API")

    disabled = not question.strip() or (use_openai and (not api_key or not consent))
    if st.button("Спросить AI-аналитика", type="primary", disabled=disabled, width="stretch"):
        try:
            with st.spinner("Агент проверяет графовые факты…"):
                answer = load_agent().ask(
                    question, use_openai=use_openai, api_key=api_key, model=model,
                )
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.markdown(answer.text)
                st.caption(
                    f"Режим: {answer.mode} · {answer.latency_seconds:.2f}s · "
                    f"Источники: {', '.join(answer.sources)}"
                )
        except Exception as agent_error:
            st.error(f"Агент не смог ответить: {agent_error}")

with documents_tab:
    st.markdown('<div class="section-kicker">Case evidence</div>', unsafe_allow_html=True)
    st.subheader("Документы дела")
    st.caption(
        "Добавляйте внешние PDF, DOCX, TXT, Markdown, CSV и JSON. Файлы сохраняются локально, "
        "не меняют рассчитанные роли или priority score и рассматриваются как непроверенный контекст аналитика."
    )
    st.info(
        "Лимит — 15 MB на файл. Одинаковые файлы определяются по SHA-256 и не дублируются. "
        "Для сканированных PDF без текстового слоя требуется предварительный OCR."
    )
    document_store = CaseDocumentStore(DOCUMENT_DIR)
    uploads = st.file_uploader(
        "Выберите документы",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        key="case-document-upload",
        help="Документы хранятся только в локальной папке case_documents, исключённой из Git.",
    )
    if st.button(
        "Добавить в досье",
        type="primary",
        disabled=not uploads,
        key="add-case-documents",
    ):
        known_gids = set(features["gid"].astype(str))
        added = 0
        for uploaded in uploads:
            try:
                record, created = document_store.add_document(
                    uploaded.name, uploaded.getvalue(), known_gids=known_gids
                )
                if created:
                    added += 1
                    st.success(
                        f"{record.filename}: добавлен; извлечено {record.text_chars:,} символов, "
                        f"GID из графа — {len(record.detected_gids)}."
                    )
                else:
                    st.info(f"{record.filename}: такой файл уже есть в досье.")
            except ValueError as document_error:
                st.error(f"{uploaded.name}: {document_error}")
        if added:
            load_agent.clear()

    documents = document_store.list_documents()
    st.markdown("#### Реестр документов")
    if documents:
        registry = pd.DataFrame(
            [
                {
                    "Файл": item.filename,
                    "Формат": item.extension.removeprefix(".").upper(),
                    "Размер, KB": round(item.size_bytes / 1024, 1),
                    "Добавлен, UTC": item.added_at.replace("T", " ")[:19],
                    "Символов": item.text_chars,
                    "GID из графа": len(item.detected_gids),
                    "SHA-256": item.sha256[:12] + "…",
                }
                for item in documents
            ]
        )
        st.dataframe(registry, hide_index=True, width="stretch")
        document_query = st.text_input(
            "Поиск по документам",
            placeholder="Введите GID, имя, организацию или фрагмент текста",
            key="document-search",
        )
        if document_query.strip():
            matches = document_store.search(document_query, limit=10)
            if matches:
                st.caption(f"Найдено документов: {len(matches)}")
                for match in matches:
                    with st.expander(f"{match['filename']} · релевантность {match['score']}"):
                        st.write(match["snippet"])
                        if match["detected_gids"]:
                            st.caption("GID из графа: " + ", ".join(match["detected_gids"]))
            else:
                st.warning("Совпадений в извлечённом тексте не найдено.")
    else:
        st.warning("В досье пока нет документов. Добавьте первый файл выше.")

    st.caption(
        "В локальном режиме AI использует найденные фрагменты без передачи данных наружу. "
        "В OpenAI-режиме документный фрагмент может быть отправлен API только после явного согласия в разделе AI-аналитика."
    )

with method_tab:
    st.markdown('<div class="section-kicker">Transparent methodology</div>', unsafe_allow_html=True)
    st.subheader("Методология и ограничения")
    st.markdown("""
    - Роли объясняются нормализованными graph/flow/temporal-признаками; black-box модель не используется.
    - Денежные потоки и роли считаются на направленном графе. Только Louvain использует взвешенную неориентированную проекцию.
    - `role_score` измеряет соответствие роли; `priority_score` — ценность проверки с учётом структуры, seed, времени и объёма.
    - Узел `depth=4` без исходящих связей находится на границе выгрузки и не считается надёжным terminal.
    - У seed неполон входящий поток, поэтому обычный pass-through для них отключён.
    - Ground truth ролей отсутствует: результат — прозрачная аналитическая гипотеза для проверки человеком.
    - Bonus-паттерны: циклы 2–6, recurring edges и A→B→C chains, rapid forwarding, синхронные входы, дробление и depth-peer anomalies.
    - Для каждого GID рассчитана полнота наблюдения, перечислены белые пятна и следующий рекомендуемый запрос аналитику.
    - AI-аналитик использует read-only инструменты над результатами pipeline; локальный режим не требует API и не передаёт данные наружу.
    """)
    st.caption(f"Pipeline runtime: {metadata['runtime_seconds']:.2f}s")
