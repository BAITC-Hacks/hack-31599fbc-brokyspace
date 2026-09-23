from __future__ import annotations

import pandas as pd


def calculate_priority(features: pd.DataFrame, cluster_score: pd.Series) -> pd.DataFrame:
    frame = features.copy()
    structural = (
        0.38 * frame["betweenness_pct"] + 0.24 * frame["pagerank_pct"]
        + 0.20 * frame["bridge_pct"] + 0.18 * frame[["hub_pct", "authority_pct"]].max(axis=1)
    )
    important_role = frame["role_score"] * frame["role"].ne("peripheral").map({True: 1.0, False: 0.2})
    volume = (frame["in_kzt_pct"] + frame["out_kzt_pct"]) / 2
    temporal = frame["aml_anomaly_score"]
    recurring = frame[["recurring_route_pct", "recurring_chain_pct"]].max(axis=1)
    frame["cluster_importance"] = frame["cluster_id"].map(cluster_score).fillna(0)
    frame["priority_score"] = (
        0.28 * structural + 0.18 * important_role + 0.16 * frame["seed_connectivity"]
        + 0.14 * temporal + 0.09 * frame["cluster_importance"] + 0.08 * volume
        + 0.04 * frame["cycle_score"] + 0.03 * recurring
    ).clip(0, 1)
    return frame
