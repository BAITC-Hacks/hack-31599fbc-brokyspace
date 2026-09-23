from __future__ import annotations

import networkx as nx
import pandas as pd


def aggregate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return (
        edges.groupby(["source", "target"], as_index=False, sort=False)
        .agg(sum_kzt=("sum_kzt", "sum"), n_tx=("n_tx", "sum"))
    )


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame]:
    aggregated = aggregate_edges(edges)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gid"])
    for row in aggregated.itertuples(index=False):
        graph.add_edge(row.source, row.target, sum_kzt=float(row.sum_kzt), n_tx=float(row.n_tx))
    nx.set_node_attributes(graph, nodes.set_index("gid")["is_seed"].to_dict(), "is_seed")
    nx.set_node_attributes(graph, nodes.set_index("gid")["depth"].to_dict(), "depth")
    return graph, aggregated


def undirected_projection(graph: nx.DiGraph) -> nx.Graph:
    """Weighted projection used only for communities/components, never for flow roles."""
    result = nx.Graph()
    result.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("sum_kzt", 0.0))
        if result.has_edge(u, v):
            result[u][v]["sum_kzt"] += weight
        else:
            result.add_edge(u, v, sum_kzt=weight)
    return result
