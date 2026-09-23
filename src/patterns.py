from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


CYCLE_COLUMNS = [
    "cycle_id", "length", "gids", "sum_kzt_total", "sum_kzt_bottleneck",
    "contains_seed", "cycle_score",
]
ROUTE_COLUMNS = [
    "source", "target", "n_tx", "active_days", "span_days", "sum_kzt",
    "cadence_regularity", "recurrence_score",
]


def _percentile(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float, index=series.index)
    return series.rank(method="average", pct=True).fillna(0.0)


def _cadence_regularity(days: pd.Series) -> float:
    unique_days = np.sort(days.dropna().dt.floor("D").unique())
    if len(unique_days) < 2:
        return 0.0
    if len(unique_days) == 2:
        return 0.5
    gaps = np.diff(unique_days).astype("timedelta64[D]").astype(float)
    mean_gap = float(gaps.mean())
    if mean_gap <= 0:
        return 0.0
    return float(1 / (1 + gaps.std() / mean_gap))


def detect_recurring_routes(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.dropna(subset=["timestamp"]).copy()
    if tx.empty:
        return pd.DataFrame(columns=ROUTE_COLUMNS)
    tx["day"] = tx["timestamp"].dt.floor("D")
    grouped = tx.groupby(["source", "target"], sort=False)
    routes = grouped.agg(
        n_tx=("n_tx", "sum"), active_days=("day", "nunique"),
        first_day=("day", "min"), last_day=("day", "max"), sum_kzt=("sum_kzt", "sum"),
    ).reset_index()
    routes["span_days"] = (routes["last_day"] - routes["first_day"]).dt.days + 1
    cadence = grouped["timestamp"].apply(_cadence_regularity).rename("cadence_regularity").reset_index()
    routes = routes.merge(cadence, on=["source", "target"], how="left")
    routes["recurrence_score"] = (
        0.45 * _percentile(routes["n_tx"])
        + 0.35 * _percentile(routes["active_days"])
        + 0.20 * routes["cadence_regularity"].fillna(0)
    ).clip(0, 1)
    eligible = routes["active_days"].ge(2)
    if not eligible.any():
        return pd.DataFrame(columns=ROUTE_COLUMNS)
    threshold = float(routes.loc[eligible, "recurrence_score"].quantile(0.75))
    recurring = routes.loc[eligible & routes["recurrence_score"].ge(threshold), ROUTE_COLUMNS]
    return recurring.sort_values(["recurrence_score", "sum_kzt"], ascending=False).reset_index(drop=True)


def detect_cycles(
    graph: nx.DiGraph,
    max_length: int = 6,
    max_cycles: int = 10_000,
) -> tuple[pd.DataFrame, bool]:
    edge_amount = {(str(u), str(v)): float(data.get("sum_kzt", 0)) for u, v, data in graph.edges(data=True)}
    seeds = {str(node) for node, data in graph.nodes(data=True) if data.get("is_seed", False)}
    raw_rows: list[dict[str, object]] = []
    truncated = False
    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        cycle = [str(node) for node in cycle]
        if len(cycle) < 2:
            continue
        if len(raw_rows) >= max_cycles:
            truncated = True
            break
        amounts = [edge_amount.get((cycle[index], cycle[(index + 1) % len(cycle)]), 0.0) for index in range(len(cycle))]
        raw_rows.append(
            {
                "length": len(cycle), "gids": " → ".join(cycle + [cycle[0]]),
                "sum_kzt_total": float(sum(amounts)), "sum_kzt_bottleneck": float(min(amounts, default=0)),
                "contains_seed": bool(seeds.intersection(cycle)),
            }
        )
    if not raw_rows:
        return pd.DataFrame(columns=CYCLE_COLUMNS), truncated
    cycles = pd.DataFrame(raw_rows)
    cycles["cycle_score"] = (
        0.45 * _percentile(cycles["sum_kzt_bottleneck"])
        + 0.30 * ((max_length + 1 - cycles["length"]) / max_length)
        + 0.25 * cycles["contains_seed"].astype(float)
    ).clip(0, 1)
    cycles = cycles.sort_values(["cycle_score", "sum_kzt_total"], ascending=False).reset_index(drop=True)
    cycles.insert(0, "cycle_id", range(1, len(cycles) + 1))
    return cycles[CYCLE_COLUMNS], truncated


def add_pattern_features(
    features: pd.DataFrame,
    cycles: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    result = features.copy()
    cycle_count: dict[str, int] = {}
    cycle_strength: dict[str, float] = {}
    for row in cycles.itertuples(index=False):
        nodes = str(row.gids).split(" → ")[:-1]
        for gid in nodes:
            cycle_count[gid] = cycle_count.get(gid, 0) + 1
            cycle_strength[gid] = max(cycle_strength.get(gid, 0.0), float(row.cycle_score))
    result["cycle_count"] = result["gid"].map(cycle_count).fillna(0).astype(int)
    result["cycle_score"] = result["gid"].map(cycle_strength).fillna(0.0)
    result["cycle_pct"] = _percentile(result["cycle_count"]) * result["cycle_count"].gt(0)

    route_nodes = pd.concat([routes.get("source", pd.Series(dtype=str)), routes.get("target", pd.Series(dtype=str))])
    route_counts = route_nodes.value_counts()
    result["recurring_route_count"] = result["gid"].map(route_counts).fillna(0).astype(int)
    result["recurring_route_pct"] = _percentile(result["recurring_route_count"]) * result["recurring_route_count"].gt(0)

    burst_pct = _percentile(result["activity_burst"]) * result["activity_burst"].gt(0)
    rapid_pct = _percentile(result["rapid_pass_through"]) * result["rapid_pass_through"].gt(0)
    result["temporal_anomaly_score"] = (
        0.40 * burst_pct + 0.30 * rapid_pct
        + 0.18 * result["recurring_route_pct"] + 0.12 * result["cycle_pct"]
    ).clip(0, 1)
    return result


def build_anomaly_report(features: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    columns = [
        "gid", "role", "priority_score", "temporal_anomaly_score", "activity_burst",
        "rapid_pass_through", "recurring_route_count", "cycle_count", "reason",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)
    threshold = float(features["temporal_anomaly_score"].quantile(0.95))
    anomalous = features.loc[features["temporal_anomaly_score"].ge(threshold)].copy()

    def reason(row: pd.Series) -> str:
        signals = []
        if row.activity_burst > 0:
            signals.append(f"burst={row.activity_burst:.2f}")
        if row.rapid_pass_through > 0:
            signals.append(f"rapid={row.rapid_pass_through:.0%}")
        if row.recurring_route_count:
            signals.append(f"recurring routes={int(row.recurring_route_count)}")
        if row.cycle_count:
            signals.append(f"cycles={int(row.cycle_count)}")
        return "; ".join(signals) or "структурная временная аномалия"

    anomalous["reason"] = anomalous.apply(reason, axis=1)
    return anomalous.nlargest(limit, "temporal_anomaly_score")[columns].reset_index(drop=True)
