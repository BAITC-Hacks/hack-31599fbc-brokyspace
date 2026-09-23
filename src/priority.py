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
    temporal = (
        0.50 * frame["rapid_pass_through"].rank(pct=True)
        + 0.30 * frame["activity_burst"].rank(pct=True)
        + 0.20 * frame["same_next_day_share"]
    )
    frame["cluster_importance"] = frame["cluster_id"].map(cluster_score).fillna(0)
    frame["priority_score"] = (
        0.30 * structural + 0.20 * important_role + 0.18 * frame["seed_connectivity"]
        + 0.12 * temporal + 0.10 * frame["cluster_importance"] + 0.10 * volume
    ).clip(0, 1)
    return frame
