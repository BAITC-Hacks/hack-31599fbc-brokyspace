from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from .graph import undirected_projection


def detect_communities(graph: nx.DiGraph, resolution: float = 1.0, seed: int = 42) -> dict[str, int]:
    projection = undirected_projection(graph)
    if projection.number_of_edges() == 0:
        return {node: index for index, node in enumerate(sorted(projection.nodes))}
    communities = nx.algorithms.community.louvain_communities(
        projection, weight="sum_kzt", resolution=resolution, seed=seed
    )
    ordered = sorted(communities, key=lambda group: (-len(group), min(map(str, group))))
    return {node: cluster_id for cluster_id, group in enumerate(ordered) for node in group}


def add_cluster_features(features: pd.DataFrame, edges: pd.DataFrame, cluster_map: dict[str, int]) -> pd.DataFrame:
    result = features.copy()
    result["cluster_id"] = result["gid"].map(cluster_map).astype(int)
    source_cluster = edges["source"].map(cluster_map)
    target_cluster = edges["target"].map(cluster_map)
    cross = source_cluster.ne(target_cluster)
    cross_nodes = pd.concat([edges.loc[cross, "source"], edges.loc[cross, "target"]]).value_counts()
    total_degree = result.set_index("gid")["in_deg"].add(result.set_index("gid")["out_deg"])
    result["cross_cluster_edges"] = result["gid"].map(cross_nodes).fillna(0)
    result["bridge_ratio"] = result["cross_cluster_edges"] / result["gid"].map(total_degree).replace(0, np.nan)
    result["bridge_ratio"] = result["bridge_ratio"].fillna(0).clip(0, 1)
    result["bridge_pct"] = result["bridge_ratio"].rank(pct=True)
    return result


def cluster_importance(features: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    cluster_by_gid = features.set_index("gid")["cluster_id"]
    internal = edges.loc[edges["source"].map(cluster_by_gid).eq(edges["target"].map(cluster_by_gid))].copy()
    internal["cluster_id"] = internal["source"].map(cluster_by_gid)
    turnover = internal.groupby("cluster_id")["sum_kzt"].sum()
    stats = features.groupby("cluster_id").agg(n_nodes=("gid", "size"), n_seed=("is_seed", "sum"))
    stats["turnover"] = turnover
    score = (
        0.35 * stats["n_nodes"].rank(pct=True)
        + 0.35 * stats["n_seed"].rank(pct=True)
        + 0.30 * stats["turnover"].fillna(0).rank(pct=True)
    )
    return score.clip(0, 1)


def build_cluster_report(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    cluster_by_gid = features.set_index("gid")["cluster_id"]
    internal = edges.loc[edges["source"].map(cluster_by_gid).eq(edges["target"].map(cluster_by_gid))].copy()
    internal["cluster_id"] = internal["source"].map(cluster_by_gid)
    turnover = internal.groupby("cluster_id")["sum_kzt"].sum()
    rows = []
    for cluster_id, group in features.groupby("cluster_id", sort=True):
        roles = Counter(group["role"])
        top = group.nlargest(5, "priority_score")["gid"].astype(str).tolist()
        n_seed = int(group["is_seed"].sum())
        money = float(turnover.get(cluster_id, 0.0))
        dominant = roles.most_common(2)
        role_text = ", ".join(f"{role}: {count}" for role, count in dominant)
        leader = dominant[0][0] if dominant else "peripheral"
        purpose = {
            "consolidator": "возможный контур сбора и аккумуляции средств",
            "distributor": "возможный контур веерного распределения средств",
            "transit": "возможный транзитный контур быстрого перенаправления",
            "terminal": "возможный контур конечных получателей",
            "coordinator": "возможный координирующий или межкластерный контур",
            "peripheral": "периферийная группа без выраженной общей функции",
        }[leader]
        hypothesis = (
            f"Гипотеза: {purpose}. Узлов={len(group)}, seed={n_seed}, роли: {role_text}; "
            f"внутренний оборот {money / 1_000_000:.2f} млн KZT. Требуется проверка аналитиком."
        )
        rows.append(
            {
                "cluster_id": int(cluster_id), "n_nodes": len(group), "n_seed": n_seed,
                "sum_kzt_internal": money, "top_gids": ";".join(top), "hypothesis": hypothesis,
            }
        )
    return pd.DataFrame(rows)
