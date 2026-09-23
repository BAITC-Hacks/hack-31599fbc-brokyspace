from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


OUT = Path("out")
ROLE_COLORS = {
    "consolidator": "#8b5cf6", "transit": "#06b6d4", "distributor": "#f59e0b",
    "terminal": "#ef4444", "coordinator": "#10b981", "peripheral": "#64748b",
}


st.set_page_config(page_title="AML Graph Intelligence", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#071019;color:#d8e2ec}.block-container{padding-top:1.2rem;max-width:1500px}
    [data-testid="stMetric"]{background:#0d1925;border:1px solid #1d3448;padding:14px;border-radius:8px}
    div[data-testid="stDataFrame"]{border:1px solid #1d3448}.risk{color:#22d3ee;letter-spacing:.12em;font-weight:700}
    .node-card{background:#0d1925;border-left:4px solid #22d3ee;padding:16px;border-radius:6px}
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


def kzt(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f} млрд ₸"
    if value >= 1e6:
        return f"{value / 1e6:.2f} млн ₸"
    return f"{value:,.0f} ₸"


def network_html(node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> str:
    allowed = set(node_frame["gid"].astype(str))
    network = Network(height="650px", width="100%", bgcolor="#08131e", font_color="#d8e2ec", directed=True)
    network.set_options("""
    {"nodes":{"borderWidth":1,"font":{"size":13}},"edges":{"color":{"color":"#486477","opacity":0.65},
    "arrows":{"to":{"enabled":true,"scaleFactor":0.55}},"smooth":{"type":"dynamic"}},
    "physics":{"barnesHut":{"gravitationalConstant":-7000,"springLength":120},"stabilization":{"iterations":180}},
    "interaction":{"hover":true,"navigationButtons":true,"keyboard":true}}
    """)
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
            st.success("Данные обработаны. Открываю dashboard…")
            st.rerun()
        except Exception as pipeline_error:
            st.error(f"Pipeline остановлен: {pipeline_error}")

    st.caption("Альтернативный запуск: `py -3 run_pipeline.py --data ./data --out ./out`")
    st.caption(f"Результаты пока отсутствуют: {exc}")
    st.stop()

st.markdown('<div class="risk">◈ AML GRAPH INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("Карта финансовой структуры группы")
st.caption("Explainable network analytics · directed cash flow · local processing")

overview, search, network_tab, priority_tab, cluster_tab, method_tab = st.tabs(
    ["Обзор", "Поиск GID", "Граф", "Приоритет", "Кластеры", "Методология"]
)

with overview:
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
                     title="Распределение ролей", template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(features, x="betweenness_pct", y="priority_score", color="role", size="pagerank_pct",
                         hover_name="gid", color_discrete_map=ROLE_COLORS, title="Структурная значимость",
                         template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
    resilience_file = OUT / "resilience.csv"
    if resilience_file.exists():
        resilience = pd.read_csv(resilience_file)
        st.subheader("Устойчивость сети")
        st.line_chart(resilience.set_index("removed_top_n")[["largest_component_share", "fragmentation"]])

with search:
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

with network_tab:
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
    components.html(network_html(shown, edges), height=670, scrolling=False)

with priority_tab:
    limit = st.radio("Показать", [20, 50], horizontal=True)
    display = features.nlargest(limit, "priority_score").copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    st.dataframe(display[["rank", "gid", "role", "priority_score", "cluster_id", "evidence"]],
                 hide_index=True, width="stretch", height=720)

with cluster_tab:
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
                           template="plotly_dark"), width="stretch")
    st.dataframe(subset.nlargest(10, "priority_score")[["gid", "role", "priority_score", "evidence"]], hide_index=True)

with method_tab:
    st.subheader("Методология и ограничения")
    st.markdown("""
    - Роли объясняются нормализованными graph/flow/temporal-признаками; black-box модель не используется.
    - Денежные потоки и роли считаются на направленном графе. Только Louvain использует взвешенную неориентированную проекцию.
    - `role_score` измеряет соответствие роли; `priority_score` — ценность проверки с учётом структуры, seed, времени и объёма.
    - Узел `depth=4` без исходящих связей находится на границе выгрузки и не считается надёжным terminal.
    - У seed неполон входящий поток, поэтому обычный pass-through для них отключён.
    - Ground truth ролей отсутствует: результат — прозрачная аналитическая гипотеза для проверки человеком.
    """)
    st.caption(f"Pipeline runtime: {metadata['runtime_seconds']:.2f}s")
