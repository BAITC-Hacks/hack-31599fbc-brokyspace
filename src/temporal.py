from __future__ import annotations

import numpy as np
import pandas as pd


TEMPORAL_COLUMNS = [
    "active_in_days", "active_out_days", "median_forward_hours",
    "same_next_day_share", "rapid_pass_through", "activity_burst",
]


def temporal_features(transactions: pd.DataFrame, gids: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame(index=pd.Index(gids.astype(str), name="gid"))
    if "timestamp" not in transactions or transactions["timestamp"].notna().sum() == 0:
        for col in TEMPORAL_COLUMNS:
            base[col] = 0.0
        return base.reset_index()

    tx = transactions.dropna(subset=["timestamp"]).copy()
    tx["day"] = tx["timestamp"].dt.floor("D")
    base["active_in_days"] = tx.groupby("target")["day"].nunique()
    base["active_out_days"] = tx.groupby("source")["day"].nunique()

    daily = tx.groupby(["source", "day"], as_index=False)["n_tx"].sum()
    burst = daily.groupby("source")["n_tx"].agg(["max", "mean"])
    base["activity_burst"] = (burst["max"] / burst["mean"].replace(0, np.nan)).clip(upper=10).fillna(0) / 10

    incoming = tx[["target", "timestamp"]].rename(columns={"target": "gid", "timestamp": "in_time"})
    outgoing = tx[["source", "timestamp"]].rename(columns={"source": "gid", "timestamp": "out_time"})
    incoming = incoming.sort_values(["in_time", "gid"])
    outgoing = outgoing.sort_values(["out_time", "gid"])
    matched = pd.merge_asof(
        outgoing, incoming, by="gid", left_on="out_time", right_on="in_time",
        direction="backward", allow_exact_matches=True,
    ).dropna(subset=["in_time"])
    if len(matched):
        matched["hours"] = (matched["out_time"] - matched["in_time"]).dt.total_seconds() / 3600
        delays = matched.groupby("gid")["hours"]
        base["median_forward_hours"] = delays.median()
        base["same_next_day_share"] = delays.apply(lambda s: float(s.le(48).mean()))
        base["rapid_pass_through"] = delays.apply(lambda s: float(s.le(24).mean()))

    return base.fillna(0).reset_index()
