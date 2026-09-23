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

THEME = {
    "bg": "#F4F7F5",
    "surface": "#FFFFFF",
    "surface2": "#EDF5F0",
    "text": "#17211B",
    "muted": "#6E7C74",
    "accent": "#16B86A",
    "accent2": "#5D86EA",
    "border": "#DFE8E2",
    "graph_bg": "#FFFFFF",
    "graph_text": "#23312A",
    "edge": "#A6B8AE",
    "plot": "plotly_white",
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
    page_title="MoneyGraph — аналитика переводов",
    page_icon="₸",
    layout="wide",
    initial_sidebar_state="expanded",
)
PLOT_TEMPLATE = THEME["plot"]
NETWORK_BG = THEME["graph_bg"]
NETWORK_TEXT = THEME["graph_text"]
EDGE_COLOR = THEME["edge"]
st.sidebar.markdown(
    '<div class="side-brand"><span>₸</span><div><b>MoneyGraph</b><small>аналитика переводов</small></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div class="system-status"><span class="status-dot"></span><div><b>Данные готовы</b><small>защищённая локальная обработка</small></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style>
    :root{{--mg-bg:{THEME['bg']};--mg-surface:{THEME['surface']};--mg-surface-2:{THEME['surface2']};--mg-text:{THEME['text']};--mg-muted:{THEME['muted']};--mg-accent:{THEME['accent']};--mg-accent-2:{THEME['accent2']};--mg-border:{THEME['border']};}}
    html,body,[class*="css"]{{font-family:Inter,"Segoe UI",Arial,sans-serif}}
    .stApp{{background:var(--mg-bg);color:var(--mg-text)}}
    .block-container{{padding-top:1rem;max-width:1480px;padding-bottom:4rem}}
    header[data-testid="stHeader"]{{background:rgba(244,247,245,.88);backdrop-filter:blur(12px)}}
    [data-testid="stSidebar"]{{background:#fff;border-right:1px solid var(--mg-border)}}
    .side-brand{{display:flex;gap:.75rem;align-items:center;padding:.45rem .1rem 1.15rem;color:var(--mg-text)}}
    .side-brand>span{{display:grid;place-items:center;width:40px;height:40px;border-radius:13px;background:var(--mg-accent);color:#fff;font-size:1.18rem;font-weight:800}}
    .side-brand b{{font-size:1.02rem;letter-spacing:-.02em}}.side-brand small,.system-status small{{display:block;color:var(--mg-muted);font-size:.72rem;margin-top:.1rem}}
    .system-status{{display:flex;align-items:center;gap:.7rem;margin-top:.6rem;padding:.85rem;border:1px solid var(--mg-border);border-radius:15px;background:#f8faf9;color:var(--mg-text)}}
    .status-dot{{width:9px;height:9px;border-radius:50%;background:var(--mg-accent);box-shadow:0 0 0 4px rgba(22,184,106,.11)}}
    .hero-shell{{position:relative;overflow:hidden;border:1px solid var(--mg-border);border-radius:34px;padding:1rem clamp(1.5rem,4.2vw,4.5rem) clamp(2rem,4vw,4rem);margin:.15rem 0 1rem;background:linear-gradient(125deg,#fff 0%,#fff 48%,#edf8f2 78%,#eaf2ff 100%);box-shadow:0 18px 60px rgba(28,56,40,.06)}}
    .product-nav{{position:relative;z-index:5;display:flex;align-items:center;justify-content:space-between;padding:.45rem 0 2.1rem;border-bottom:1px solid rgba(31,58,43,.08)}}
    .product-logo{{display:flex;align-items:center;gap:.55rem;font-weight:800;font-size:1rem}}.product-logo i{{display:grid;place-items:center;width:31px;height:31px;border-radius:10px;background:var(--mg-accent);color:#fff;font-style:normal}}
    .product-links{{display:flex;gap:1.65rem;color:var(--mg-muted);font-size:.78rem}}.product-links span:first-child{{color:var(--mg-text);font-weight:700}}
    .hero-grid{{position:relative;z-index:2;display:grid;grid-template-columns:minmax(0,1.05fr) minmax(390px,.95fr);gap:2rem;align-items:center;min-height:410px}}
    .eyebrow{{display:inline-flex;align-items:center;gap:.5rem;padding:.48rem .7rem;border-radius:999px;background:#eaf8f0;color:#087944;font-size:.7rem;font-weight:750;margin-bottom:1.25rem}}
    .eyebrow::before{{content:"";width:7px;height:7px;border-radius:50%;background:var(--mg-accent)}}
    .hero-title{{margin:0;color:var(--mg-text);font-size:clamp(2.65rem,5vw,5.4rem);line-height:.96;letter-spacing:-.06em;font-weight:750;max-width:780px}}.hero-title span{{color:var(--mg-accent)}}
    .hero-copy{{max-width:650px;color:var(--mg-muted);font-size:1.03rem;line-height:1.65;margin:1.35rem 0 0}}
    .hero-chips{{display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1.5rem}}.hero-chip{{padding:.58rem .82rem;border-radius:12px;border:1px solid var(--mg-border);background:rgba(255,255,255,.76);font-size:.74rem;color:var(--mg-text)}}
    .hero-action{{display:inline-flex;margin-top:1.4rem;padding:.72rem 1.05rem;border-radius:13px;background:var(--mg-accent);color:#fff;font-weight:750;font-size:.8rem}}
    .bank-scene{{position:relative;height:390px;perspective:1000px;transform-style:preserve-3d}}
    .bank-floor{{position:absolute;left:8%;right:4%;bottom:25px;height:90px;border-radius:50%;background:radial-gradient(ellipse,rgba(47,114,76,.16),rgba(47,114,76,0) 68%);transform:rotateX(70deg)}}
    .bank-world{{position:absolute;left:50%;top:52%;width:280px;height:230px;transform-style:preserve-3d;transform:translate(-50%,-48%) rotateX(5deg) rotateY(-12deg);animation:bankFloat 5s ease-in-out infinite}}
    .bank-roof{{position:absolute;left:20px;top:0;width:240px;height:72px;background:linear-gradient(135deg,#1ecb7a,#0ca45b);clip-path:polygon(50% 0,100% 82%,94% 100%,6% 100%,0 82%);filter:drop-shadow(0 14px 12px rgba(15,107,62,.16));transform:translateZ(24px)}}
    .bank-roof::after{{content:"₸";position:absolute;left:111px;top:24px;color:#fff;font-size:24px;font-weight:850}}
    .bank-body{{position:absolute;left:35px;top:68px;width:210px;height:132px;border-radius:5px;background:linear-gradient(90deg,#eef4f0,#fff 22%,#f5f8f6);box-shadow:22px 20px 35px rgba(22,71,45,.16);transform:translateZ(8px)}}
    .bank-columns{{position:absolute;inset:14px 19px 18px;display:flex;justify-content:space-between}}.bank-columns i{{width:22px;border-radius:4px;background:linear-gradient(90deg,#dce6e0,#fff,#d5e1da);box-shadow:0 6px 0 #c9d7cf}}
    .bank-door{{position:absolute;left:88px;bottom:0;width:38px;height:54px;border-radius:12px 12px 0 0;background:#21342a;box-shadow:inset 0 0 0 6px #324b3e}}
    .bank-steps{{position:absolute;left:25px;top:194px;width:230px;height:38px;background:linear-gradient(#e3ebe6 0 32%,#d5e1da 33% 65%,#c8d6ce 66%);clip-path:polygon(8% 0,92% 0,100% 100%,0 100%);transform:translateZ(12px)}}
    .coin{{position:absolute;display:grid;place-items:center;width:48px;height:48px;border-radius:50%;background:linear-gradient(145deg,#ffd66b,#f6ae24);border:5px solid #ffe396;color:#7e5612;font-weight:900;box-shadow:0 12px 25px rgba(152,106,20,.18);animation:coinFloat 4s ease-in-out infinite}}
    .coin-one{{right:8%;top:15%;animation-delay:-1s}}.coin-two{{left:4%;bottom:20%;width:38px;height:38px;animation-delay:-2.4s}}
    .transfer-card{{position:absolute;padding:.65rem .8rem;border:1px solid rgba(255,255,255,.9);border-radius:14px;background:rgba(255,255,255,.87);box-shadow:0 14px 35px rgba(26,67,44,.11);backdrop-filter:blur(10px);font-size:.7rem;color:var(--mg-muted);animation:cardFloat 5.5s ease-in-out infinite}}.transfer-card b{{display:block;color:var(--mg-text);font-size:.92rem;margin-top:.14rem}}
    .transfer-a{{left:0;top:12%}}.transfer-b{{right:0;bottom:12%;animation-delay:-2.7s}}.transfer-dot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--mg-accent);margin-right:.35rem}}
    .flow{{position:absolute;height:2px;background:linear-gradient(90deg,transparent,var(--mg-accent),transparent);opacity:.55;animation:flowPulse 2.6s linear infinite}}.flow-a{{left:10%;right:13%;top:34%;transform:rotate(8deg)}}.flow-b{{left:16%;right:5%;bottom:28%;transform:rotate(-10deg);animation-delay:-1.2s}}
    .risk-float{{position:absolute;right:4%;top:4%;padding:.55rem .68rem;border-radius:12px;background:#fff;border:1px solid var(--mg-border);font-size:.66rem;color:var(--mg-muted);box-shadow:0 10px 30px rgba(28,56,40,.08)}}.risk-float b{{display:block;color:var(--mg-text);font-size:.86rem;margin-top:.12rem}}
    .insight-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin:0 0 1.25rem}}.insight-item{{border:1px solid var(--mg-border);border-radius:18px;background:#fff;padding:1.05rem 1.15rem;color:var(--mg-text)}}.insight-item small{{display:block;color:var(--mg-muted);font-size:.68rem;margin-bottom:.42rem}}.insight-item b{{font-size:1.16rem;letter-spacing:-.025em}}
    [data-testid="stMetric"]{{background:#fff;border:1px solid var(--mg-border);padding:16px 17px;border-radius:17px;box-shadow:none}}[data-testid="stMetricLabel"]{{color:var(--mg-muted)}}[data-testid="stMetricValue"]{{color:var(--mg-text);letter-spacing:-.035em;font-size:clamp(1.45rem,2vw,2.1rem)}}
    div[data-testid="stDataFrame"]{{border:1px solid var(--mg-border);border-radius:16px;overflow:hidden;box-shadow:none}}
    .stTabs [data-baseweb="tab-list"]{{gap:.15rem;background:#fff;border:1px solid var(--mg-border);border-radius:16px;padding:.34rem;overflow-x:auto}}.stTabs [data-baseweb="tab"]{{height:40px;border-radius:11px;color:var(--mg-muted);padding:0 .82rem;white-space:nowrap;font-size:.83rem}}.stTabs [aria-selected="true"]{{background:#e8f7ef!important;color:#087944!important;font-weight:750}}.stTabs [data-baseweb="tab-highlight"]{{display:none}}
    .stButton>button[kind="primary"]{{border:0;border-radius:13px;background:var(--mg-accent);color:#fff;font-weight:750;box-shadow:none}}.stButton>button:not([kind="primary"]){{border-radius:13px;border-color:var(--mg-border);background:#fff;color:var(--mg-text)}}
    div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea{{background:#fff!important;border-color:var(--mg-border)!important;color:var(--mg-text)!important;border-radius:12px!important}}
    .node-card{{background:#fff;border:1px solid var(--mg-border);border-left:4px solid var(--mg-accent);padding:18px;border-radius:16px;color:var(--mg-text);box-shadow:none}}
    .risk{{color:var(--mg-accent);font-weight:800}}h1,h2,h3,p,label{{color:var(--mg-text)}}.section-kicker{{color:#0b8f52;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;margin-top:.35rem}}
    @keyframes bankFloat{{0%,100%{{transform:translate(-50%,-48%) rotateX(5deg) rotateY(-12deg) translateY(0)}}50%{{transform:translate(-50%,-48%) rotateX(7deg) rotateY(-7deg) translateY(-10px)}}}}
    @keyframes coinFloat{{0%,100%{{transform:translateY(0) rotateY(0)}}50%{{transform:translateY(-14px) rotateY(180deg)}}}}
    @keyframes cardFloat{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-9px)}}}}
    @keyframes flowPulse{{0%{{opacity:.15;filter:saturate(.7)}}50%{{opacity:.75;filter:saturate(1.3)}}100%{{opacity:.15;filter:saturate(.7)}}}}
    @media(max-width:980px){{.hero-grid{{grid-template-columns:1fr}}.bank-scene{{height:330px}}.product-links{{display:none}}.insight-strip{{grid-template-columns:repeat(2,1fr)}}}}
    @media(max-width:560px){{.hero-shell{{border-radius:24px;padding:1rem 1.15rem 1.8rem}}.hero-title{{font-size:2.65rem}}.bank-scene{{height:285px;transform:scale(.88)}}.insight-strip{{grid-template-columns:1fr 1fr}}}}
    @media(prefers-reduced-motion:reduce){{.bank-world,.coin,.transfer-card,.flow{{animation:none!important}}}}
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
    transactions = int(eda["transaction_rows"])
    turnover = float(eda["edge_turnover_kzt"])
    average_transfer = turnover / max(transactions, 1)
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="product-nav">
            <div class="product-logo"><i>₸</i><span>MoneyGraph</span></div>
            <div class="product-links"><span>Обзор</span><span>Переводы</span><span>Клиенты</span><span>Кластеры</span><span>Методология</span></div>
          </div>
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Финансовая сеть за июль 2026</div>
              <h1 class="hero-title">Переводы<br><span>под контролем</span></h1>
              <p class="hero-copy">Понятная карта движения денег: клиенты, связи, объёмы, повторяющиеся маршруты и приоритет проверки — без визуального шума.</p>
              <div class="hero-chips">
                <span class="hero-chip">{int(eda['nodes']):,} клиентов</span>
                <span class="hero-chip">{cluster_count} кластеров</span>
                <span class="hero-chip">Расчёт {runtime:.2f} сек.</span>
              </div>
              <span class="hero-action">Смотреть движение денег →</span>
            </div>
            <div class="bank-scene" role="img" aria-label="Анимированная модель банка и денежных переводов">
              <div class="bank-floor"></div>
              <div class="flow flow-a"></div><div class="flow flow-b"></div>
              <div class="transfer-card transfer-a"><span class="transfer-dot"></span>Входящий поток<b>{kzt(turnover * .54)}</b></div>
              <div class="transfer-card transfer-b"><span class="transfer-dot"></span>Исходящий поток<b>{kzt(turnover * .46)}</b></div>
              <div class="coin coin-one">₸</div><div class="coin coin-two">₸</div>
              <div class="risk-float">Первый к проверке<b>{gid} · {role}</b></div>
              <div class="bank-world">
                <div class="bank-roof"></div>
                <div class="bank-body"><div class="bank-columns"><i></i><i></i><i></i><i></i></div><div class="bank-door"></div></div>
                <div class="bank-steps"></div>
              </div>
            </div>
          </div>
        </section>
        <div class="insight-strip">
          <div class="insight-item"><small>Всего переводов</small><b>{transactions:,}</b></div>
          <div class="insight-item"><small>Оборот сети</small><b>{kzt(turnover)}</b></div>
          <div class="insight-item"><small>Средний перевод</small><b>{kzt(average_transfer)}</b></div>
          <div class="insight-item"><small>Активные связи</small><b>{int(eda['edges_rows']):,}</b></div>
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
st.sidebar.caption("ФИНАНСОВЫЙ ОБЗОР")
st.sidebar.metric("Период", "Июль 2026")
st.sidebar.metric("Оборот сети", kzt(float(metadata["eda"]["edge_turnover_kzt"])))
st.sidebar.caption("Данные обезличены · расчёты выполняются локально")

overview, search, network_tab, priority_tab, cluster_tab, patterns_tab, agent_tab, documents_tab, method_tab = st.tabs(
    ["Обзор", "Клиенты", "Карта переводов", "Очередь", "Группы", "Сигналы", "Помощник", "Документы", "О методике"]
)

with overview:
    st.markdown('<div class="section-kicker">Статистика переводов</div>', unsafe_allow_html=True)
    st.subheader("Финансовый обзор")
    eda = metadata["eda"]
    edge_turnover = float(edges["sum_kzt"].sum())
    average_route_transfer = (edges["sum_kzt"] / edges["n_tx"].clip(lower=1)).median()
    largest_route = float(edges["sum_kzt"].max())
    top_share = float(edges.nlargest(min(10, len(edges)), "sum_kzt")["sum_kzt"].sum() / max(edge_turnover, 1))
    metrics = st.columns(4)
    metrics[0].metric("Переводов", f"{int(eda['transaction_rows']):,}")
    metrics[1].metric("Медианный перевод", kzt(float(average_route_transfer)))
    metrics[2].metric("Крупнейший маршрут", kzt(largest_route))
    metrics[3].metric("Доля TOP-10 маршрутов", f"{top_share:.1%}")

    route_stats = edges.nlargest(12, "sum_kzt").copy()
    route_stats["Маршрут"] = route_stats["source"].astype(str).str[-6:] + " → " + route_stats["target"].astype(str).str[-6:]
    route_stats["Средний перевод"] = route_stats["sum_kzt"] / route_stats["n_tx"].clip(lower=1)
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            route_stats.sort_values("sum_kzt"),
            x="sum_kzt",
            y="Маршрут",
            orientation="h",
            title="Крупнейшие денежные маршруты",
            labels={"sum_kzt": "Оборот, ₸"},
            color_discrete_sequence=[THEME["accent"]],
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(showlegend=False, margin=dict(l=12, r=12, t=55, b=12))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(
            route_stats,
            x="n_tx",
            y="sum_kzt",
            size="Средний перевод",
            color="Средний перевод",
            hover_name="Маршрут",
            title="Частота и объём переводов",
            labels={"n_tx": "Количество переводов", "sum_kzt": "Оборот, ₸"},
            color_continuous_scale=["#DDF5E8", THEME["accent"]],
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=12, r=12, t=55, b=12))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-kicker">Структура участников</div>', unsafe_allow_html=True)
    st.subheader("Роли и значимость в сети")
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
    st.markdown('<div class="section-kicker">Профиль клиента</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-kicker">Движение денег</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-kicker">Группы клиентов</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-kicker">Финансовые сигналы</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-kicker">Помощник аналитика</div>', unsafe_allow_html=True)
    st.subheader("Разобрать данные простым языком")
    st.caption(
        "Отвечает по рассчитанным артефактам pipeline и локальному досье, всегда показывает источники "
        "и отделяет непроверенный текст документов от графовых фактов."
    )
    mode = st.radio("Режим ответа", ["Локально по расчётам", "Расширенный OpenAI"], horizontal=True)
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

    use_openai = mode == "Расширенный OpenAI"
    api_key = configured_openai_api_key() if use_openai else ""
    model = "gpt-6-astra"
    consent = True
    if use_openai:
        st.warning(
            "Расширенный режим отправляет вопрос и минимальный контекст выбранных узлов или найденный фрагмент документа в OpenAI API. "
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
    if st.button("Получить разбор", type="primary", disabled=disabled, width="stretch"):
        try:
            with st.spinner("Проверяем расчёты и источники…"):
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
            st.error(f"Не удалось подготовить ответ: {agent_error}")

with documents_tab:
    st.markdown('<div class="section-kicker">Материалы проверки</div>', unsafe_allow_html=True)
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
        "В локальном режиме помощник использует найденные фрагменты без передачи данных наружу. "
        "В расширенном режиме документный фрагмент может быть отправлен API только после явного согласия в разделе «Помощник»."
    )

with method_tab:
    st.markdown('<div class="section-kicker">Прозрачный расчёт</div>', unsafe_allow_html=True)
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
    - Помощник использует read-only инструменты над результатами pipeline; локальный режим не требует API и не передаёт данные наружу.
    """)
    st.caption(f"Pipeline runtime: {metadata['runtime_seconds']:.2f}s")
