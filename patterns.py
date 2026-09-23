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
CHAIN_COLUMNS = [
    "source", "middle", "target", "n_occurrences", "active_days", "span_days",
    "sum_kzt_bottleneck", "cadence_regularity", "recurrence_score",
]
SYNC_COLUMNS = ["target", "day", "distinct_senders", "n_tx", "sum_kzt", "synchronous_score"]
STRUCTURING_COLUMNS = [
    "gid", "direction", "day", "n_tx", "counterparties", "sum_kzt", "mean_kzt",
    "amount_cv", "near_threshold_share", "round_amount_share", "structuring_score",
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


def detect_recurring_chains(transactions: pd.DataFrame) -> pd.DataFrame:
    """Find repeated same-day A→B→C footprints on at least two dates."""
    tx = transactions.dropna(subset=["timestamp"]).copy()
    if tx.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    tx["day"] = tx["timestamp"].dt.floor("D")
    daily = tx.groupby(["source", "target", "day"], as_index=False).agg(
        n_tx=("n_tx", "sum"), sum_kzt=("sum_kzt", "sum")
    )
    first = daily.rename(
        columns={"target": "middle", "n_tx": "first_n_tx", "sum_kzt": "first_sum_kzt"}
    )
    second = daily.rename(
        columns={"source": "middle", "n_tx": "second_n_tx", "sum_kzt": "second_sum_kzt"}
    )
    joined = first.merge(second, on=["middle", "day"], how="inner", suffixes=("", "_second"))
    joined = joined.loc[
        joined["source"].ne(joined["target"])
        & joined["source"].ne(joined["middle"])
        & joined["middle"].ne(joined["target"])
    ].copy()
    if joined.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    joined["bottleneck_kzt"] = joined[["first_sum_kzt", "second_sum_kzt"]].min(axis=1)
    grouped = joined.groupby(["source", "middle", "target"], sort=False)
    chains = grouped.agg(
        n_occurrences=("day", "size"), active_days=("day", "nunique"),
        first_day=("day", "min"), last_day=("day", "max"),
        sum_kzt_bottleneck=("bottleneck_kzt", "sum"),
    ).reset_index()
    chains["span_days"] = (chains["last_day"] - chains["first_day"]).dt.days + 1
    cadence = grouped["day"].apply(_cadence_regularity).rename("cadence_regularity").reset_index()
    chains = chains.merge(cadence, on=["source", "middle", "target"], how="left")
    chains["recurrence_score"] = (
        0.35 * _percentile(chains["active_days"])
        + 0.25 * _percentile(chains["n_occurrences"])
        + 0.25 * _percentile(chains["sum_kzt_bottleneck"])
        + 0.15 * chains["cadence_regularity"].fillna(0)
    ).clip(0, 1)
    eligible = chains["active_days"].ge(2)
    if not eligible.any():
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    threshold = float(chains.loc[eligible, "recurrence_score"].quantile(0.75))
    result = chains.loc[eligible & chains["recurrence_score"].ge(threshold), CHAIN_COLUMNS]
    return result.sort_values(["recurrence_score", "sum_kzt_bottleneck"], ascending=False).reset_index(drop=True)


def detect_synchronous_inflows(transactions: pd.DataFrame) -> pd.DataFrame:
    """Find days when at least three distinct payers converge on one target."""
    tx = transactions.dropna(subset=["timestamp"]).copy()
    if tx.empty:
        return pd.DataFrame(columns=SYNC_COLUMNS)
    tx["day"] = tx["timestamp"].dt.floor("D")
    daily = tx.groupby(["target", "day"], as_index=False).agg(
        distinct_senders=("source", "nunique"), n_tx=("n_tx", "sum"), sum_kzt=("sum_kzt", "sum")
    )
    daily = daily.loc[daily["distinct_senders"].ge(3)].copy()
    if daily.empty:
        return pd.DataFrame(columns=SYNC_COLUMNS)
    daily["synchronous_score"] = (
        0.50 * _percentile(daily["distinct_senders"])
        + 0.25 * _percentile(daily["n_tx"])
        + 0.25 * _percentile(daily["sum_kzt"])
    ).clip(0, 1)
    return daily[SYNC_COLUMNS].sort_values(
        ["synchronous_score", "sum_kzt"], ascending=False
    ).reset_index(drop=True)


def detect_structuring_events(transactions: pd.DataFrame, threshold_kzt: float = 5_000) -> pd.DataFrame:
    """Flag explainable same-day amount-splitting signals above the observed-data floor."""
    tx = transactions.dropna(subset=["timestamp"]).copy()
    if tx.empty:
        return pd.DataFrame(columns=STRUCTURING_COLUMNS)
    tx["day"] = tx["timestamp"].dt.floor("D")
    frames = []
    for direction, gid_column, counterparty_column in (
        ("outgoing", "source", "target"), ("incoming", "target", "source")
    ):
        view = tx.assign(
            gid=tx[gid_column], counterparty=tx[counterparty_column], direction=direction,
            near_threshold=tx["sum_kzt"].le(threshold_kzt * 4),
            round_amount=np.isclose(tx["sum_kzt"] % 1_000, 0),
        )
        grouped = view.groupby(["gid", "direction", "day"], sort=False)
        stats = grouped.agg(
            n_tx=("n_tx", "sum"), counterparties=("counterparty", "nunique"),
            sum_kzt=("sum_kzt", "sum"), mean_kzt=("sum_kzt", "mean"),
            amount_std=("sum_kzt", "std"), near_threshold_share=("near_threshold", "mean"),
            round_amount_share=("round_amount", "mean"),
        ).reset_index()
        stats["amount_cv"] = (stats["amount_std"] / stats["mean_kzt"].replace(0, np.nan)).fillna(0)
        eligible = stats["n_tx"].ge(3) & stats["counterparties"].ge(2) & (
            stats["amount_cv"].le(0.20)
            | stats["near_threshold_share"].ge(0.50)
            | stats["round_amount_share"].ge(0.80)
        )
        frames.append(stats.loc[eligible])
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if events.empty:
        return pd.DataFrame(columns=STRUCTURING_COLUMNS)
    similarity = (1 - events["amount_cv"].clip(0, 1))
    events["structuring_score"] = (
        0.25 * _percentile(events["n_tx"])
        + 0.20 * _percentile(events["counterparties"])
        + 0.20 * similarity
        + 0.20 * events["near_threshold_share"]
        + 0.15 * events["round_amount_share"]
    ).clip(0, 1)
    return events[STRUCTURING_COLUMNS].sort_values(
        ["structuring_score", "sum_kzt"], ascending=False
    ).reset_index(drop=True)


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
    chains: pd.DataFrame,
    structuring: pd.DataFrame,
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

    chain_nodes = pd.concat(
        [chains.get("source", pd.Series(dtype=str)), chains.get("middle", pd.Series(dtype=str)), chains.get("target", pd.Series(dtype=str))]
    )
    chain_counts = chain_nodes.value_counts()
    result["recurring_chain_count"] = result["gid"].map(chain_counts).fillna(0).astype(int)
    result["recurring_chain_pct"] = _percentile(result["recurring_chain_count"]) * result["recurring_chain_count"].gt(0)

    structuring_counts = structuring.groupby("gid").size() if not structuring.empty else pd.Series(dtype=float)
    structuring_strength = (
        structuring.groupby("gid")["structuring_score"].max() if not structuring.empty else pd.Series(dtype=float)
    )
    result["structuring_event_count"] = result["gid"].map(structuring_counts).fillna(0).astype(int)
    result["structuring_score"] = result["gid"].map(structuring_strength).fillna(0.0)

    burst_pct = _percentile(result["activity_burst"]) * result["activity_burst"].gt(0)
    rapid_pct = _percentile(result["rapid_pass_through"]) * result["rapid_pass_through"].gt(0)
    result["temporal_anomaly_score"] = (
        0.25 * burst_pct + 0.20 * rapid_pct + 0.15 * result["synchronous_in_score"]
        + 0.15 * result["structuring_score"] + 0.10 * result["recurring_route_pct"]
        + 0.08 * result["recurring_chain_pct"] + 0.07 * result["cycle_pct"]
    ).clip(0, 1)

    peer_columns = ["in_deg", "out_deg", "in_kzt", "out_kzt", "activity_burst", "rapid_pass_through"]
    peer_ranks = pd.DataFrame(index=result.index)
    for column in peer_columns:
        peer_ranks[column] = result.groupby("depth", dropna=False)[column].rank(method="average", pct=True)
    result["depth_peer_anomaly_score"] = (
        0.30 * peer_ranks[["in_deg", "out_deg"]].max(axis=1)
        + 0.25 * peer_ranks[["in_kzt", "out_kzt"]].max(axis=1)
        + 0.20 * peer_ranks[["activity_burst", "rapid_pass_through"]].max(axis=1)
        + 0.15 * result["synchronous_in_score"]
        + 0.10 * result["structuring_score"]
    ).clip(0, 1)
    result["aml_anomaly_score"] = (
        0.70 * result["temporal_anomaly_score"] + 0.30 * result["depth_peer_anomaly_score"]
    ).clip(0, 1)
    return result


def build_anomaly_report(features: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    columns = [
        "gid", "role", "priority_score", "aml_anomaly_score", "temporal_anomaly_score",
        "depth_peer_anomaly_score", "activity_burst", "rapid_pass_through",
        "synchronous_in_days", "structuring_event_count", "recurring_route_count",
        "recurring_chain_count", "cycle_count", "reason",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)
    threshold = float(features["aml_anomaly_score"].quantile(0.95))
    anomalous = features.loc[features["aml_anomaly_score"].ge(threshold)].copy()

    def reason(row: pd.Series) -> str:
        signals = []
        if row.activity_burst > 0:
            signals.append(f"burst={row.activity_burst:.2f}")
        if row.rapid_pass_through > 0:
            signals.append(f"rapid={row.rapid_pass_through:.0%}")
        if row.synchronous_in_days:
            signals.append(f"sync days={int(row.synchronous_in_days)}")
        if row.structuring_event_count:
            signals.append(f"structuring={int(row.structuring_event_count)}")
        if row.recurring_route_count:
            signals.append(f"recurring routes={int(row.recurring_route_count)}")
        if row.recurring_chain_count:
            signals.append(f"A→B→C chains={int(row.recurring_chain_count)}")
        if row.cycle_count:
            signals.append(f"cycles={int(row.cycle_count)}")
        return "; ".join(signals) or "структурная временная аномалия"

    anomalous["reason"] = anomalous.apply(reason, axis=1)
    return anomalous.nlargest(limit, "aml_anomaly_score")[columns].reset_index(drop=True)
