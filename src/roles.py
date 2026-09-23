from __future__ import annotations

import numpy as np
import pandas as pd


ROLE_SCORE_COLUMNS = [
    "consolidator_score", "transit_score", "distributor_score", "terminal_score", "coordinator_score"
]


def _pct(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0)


def score_roles(features: pd.DataFrame, min_signal: float = 0.38) -> pd.DataFrame:
    frame = features.copy()
    both_sides = (frame["in_deg"].gt(0) & frame["out_deg"].gt(0)).astype(float)
    retained = (1 - frame["out_kzt"] / frame["in_kzt"].replace(0, np.nan)).clip(0, 1).fillna(0)
    fan_in = (frame["in_deg"] / (frame["in_deg"] + frame["out_deg"]).replace(0, np.nan)).fillna(0)
    fan_out = (frame["out_deg"] / (frame["in_deg"] + frame["out_deg"]).replace(0, np.nan)).fillna(0)
    rapid_pct = _pct(frame["rapid_pass_through"])

    frame["consolidator_score"] = (
        0.20 * frame["in_deg_pct"] + 0.18 * frame["in_kzt_pct"] + 0.16 * frame["authority_pct"]
        + 0.14 * _pct(frame["seed_sources"]) + 0.10 * fan_in + 0.10 * retained
        + 0.12 * frame["synchronous_in_score"]
    )
    frame["distributor_score"] = (
        0.27 * frame["out_deg_pct"] + 0.20 * frame["out_tx_pct"] + 0.18 * frame["hub_pct"]
        + 0.18 * fan_out + 0.10 * frame["out_kzt_pct"] + 0.07 * frame["seed_connectivity"]
    )
    balance = frame["pass_through"].clip(0, 1)
    frame["transit_score"] = (
        0.27 * both_sides + 0.30 * balance + 0.18 * rapid_pct
        + 0.12 * frame["same_next_day_share"] + 0.13 * (1 - retained)
    )
    seed = frame["is_seed"].astype(bool)
    frame.loc[seed, "transit_score"] = (
        0.38 * both_sides[seed] + 0.22 * rapid_pct[seed]
        + 0.20 * frame.loc[seed, "same_next_day_share"] + 0.20 * frame.loc[seed, "out_deg_pct"]
    )

    no_out = frame["out_deg"].eq(0).astype(float)
    frame["terminal_score"] = (
        0.26 * frame["in_kzt_pct"] + 0.18 * frame["in_deg_pct"] + 0.24 * no_out
        + 0.20 * retained + 0.12 * (1 - frame["out_kzt_pct"])
    ) * frame["in_deg"].gt(0)
    frame.loc[frame["truncated_by_depth"], "terminal_score"] *= 0.35

    frame["coordinator_score"] = (
        0.29 * frame["betweenness_pct"] + 0.17 * frame["pagerank_pct"]
        + 0.12 * frame[["hub_pct", "authority_pct"]].max(axis=1)
        + 0.22 * frame["bridge_pct"] + 0.12 * frame["seed_connectivity"]
        + 0.08 * _pct(frame["in_deg"] + frame["out_deg"])
    )
    frame[ROLE_SCORE_COLUMNS] = frame[ROLE_SCORE_COLUMNS].clip(0, 1)

    names = [column.removesuffix("_score") for column in ROLE_SCORE_COLUMNS]
    strongest = frame[ROLE_SCORE_COLUMNS].max(axis=1)
    chosen = frame[ROLE_SCORE_COLUMNS].idxmax(axis=1).str.removesuffix("_score")
    frame["role"] = np.where(strongest.ge(min_signal), chosen, "peripheral")
    frame["role_score"] = np.where(frame["role"].eq("peripheral"), 1 - strongest, strongest).clip(0, 1)
    return frame
