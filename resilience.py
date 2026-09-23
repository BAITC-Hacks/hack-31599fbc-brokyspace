from __future__ import annotations

import networkx as nx
import pandas as pd


def analyse_resilience(graph: nx.DiGraph, ranked: pd.DataFrame, removals: tuple[int, ...] = (0, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    ordered = ranked.sort_values("priority_score", ascending=False)["gid"].tolist()
    baseline_nodes = max(graph.number_of_nodes(), 1)
    for count in removals:
        reduced = graph.copy()
        reduced.remove_nodes_from(ordered[:count])
        components = list(nx.weakly_connected_components(reduced))
        largest = max(map(len, components), default=0)
        rows.append(
            {
                "removed_top_n": count,
                "remaining_nodes": reduced.number_of_nodes(),
                "weak_components": len(components),
                "largest_weak_component": largest,
                "largest_component_share": largest / max(reduced.number_of_nodes(), 1),
                "fragmentation": 1 - largest / baseline_nodes,
            }
        )
    return pd.DataFrame(rows)
