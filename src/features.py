from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from .graph import undirected_projection


PERCENTILE_FEATURES = [
    "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx",
    "pagerank", "betweenness", "hub", "authority",
]


def _safe_hits(graph: nx.DiGraph) -> tuple[dict, dict]:
    try:
        hubs, authorities = nx.hits(graph, max_iter=500, tol=1e-8, normalized=True)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        hubs = {node: 0.0 for node in graph}
        authorities = {node: 0.0 for node in graph}
    return hubs, authorities


def _seed_distances(graph: nx.DiGraph, seeds: list[str]) -> dict[str, float]:
    if not seeds:
        return {node: math.inf for node in graph}
    distances = nx.multi_source_shortest_path_length(graph, seeds) if hasattr(nx, "multi_source_shortest_path_length") else None
    if distances is not None:
        return dict(distances)
    result: dict[str, float] = {node: math.inf for node in graph}
    for seed in seeds:
        for node, distance in nx.single_source_shortest_path_length(graph, seed).items():
            result[node] = min(result[node], distance)
    return result


def graph_features(graph: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    gids = list(nodes["gid"].astype(str))
    frame = nodes.set_index("gid")[["depth", "is_seed"]].copy()
    frame["in_deg"] = pd.Series(dict(graph.in_degree()), dtype=float)
    frame["out_deg"] = pd.Series(dict(graph.out_degree()), dtype=float)
    frame["in_kzt"] = pd.Series(dict(graph.in_degree(weight="sum_kzt")), dtype=float)
    frame["out_kzt"] = pd.Series(dict(graph.out_degree(weight="sum_kzt")), dtype=float)
    frame["in_tx"] = pd.Series(dict(graph.in_degree(weight="n_tx")), dtype=float)
    frame["out_tx"] = pd.Series(dict(graph.out_degree(weight="n_tx")), dtype=float)
    frame["pagerank"] = pd.Series(nx.pagerank(graph, weight="sum_kzt", max_iter=300), dtype=float)
    frame["betweenness"] = pd.Series(nx.betweenness_centrality(graph, weight=None, normalized=True), dtype=float)
    hubs, authorities = _safe_hits(graph)
    frame["hub"] = pd.Series(hubs, dtype=float).clip(lower=0)
    frame["authority"] = pd.Series(authorities, dtype=float).clip(lower=0)

    undirected = undirected_projection(graph)
    component_map = {}
    for component_id, component in enumerate(nx.connected_components(undirected)):
        component_map.update({node: component_id for node in component})
    frame["weak_component"] = pd.Series(component_map, dtype=int)

    seeds = nodes.loc[nodes["is_seed"], "gid"].astype(str).tolist()
    distance = _seed_distances(graph, seeds)
    frame["seed_distance"] = pd.Series(distance).replace(math.inf, np.nan)
    frame["seed_proximity"] = 1 / (1 + frame["seed_distance"])
    seed_set = set(seeds)
    frame["seed_neighbours"] = pd.Series(
        {node: sum(neighbour in seed_set for neighbour in set(graph.predecessors(node)) | set(graph.successors(node))) for node in gids},
        dtype=float,
    )
    frame["seed_sources"] = pd.Series(
        {node: sum(source in seed_set for source in graph.predecessors(node)) for node in gids}, dtype=float
    )
    frame["seed_connectivity"] = (frame["seed_proximity"] + frame["seed_neighbours"].rank(pct=True)) / 2
    frame.loc[frame["is_seed"], "seed_connectivity"] = 1.0

    observed_in = frame["in_kzt"]
    observed_out = frame["out_kzt"]
    frame["pass_through"] = np.minimum(observed_in, observed_out) / np.maximum(observed_in, observed_out).replace(0, np.nan)
    frame.loc[frame["is_seed"], "pass_through"] = np.nan
    max_depth = pd.to_numeric(frame["depth"], errors="coerce").max()
    frame["truncated_by_depth"] = (
        frame["depth"].eq(max_depth) & frame["out_deg"].eq(0) & frame["depth"].notna()
    )
    for feature in PERCENTILE_FEATURES:
        frame[f"{feature}_pct"] = frame[feature].rank(method="average", pct=True).fillna(0)
    return frame.fillna({"pass_through": 0, "seed_distance": -1, "seed_proximity": 0}).reset_index()
