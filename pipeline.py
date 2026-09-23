from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .clustering import add_cluster_features, build_cluster_report, cluster_importance, detect_communities
from .completeness import add_completeness, completeness_report
from .data import describe_data, load_data
from .evidence import add_evidence
from .features import graph_features
from .graph import build_graph
from .patterns import (
    add_pattern_features,
    build_anomaly_report,
    detect_cycles,
    detect_recurring_chains,
    detect_recurring_routes,
    detect_structuring_events,
    detect_synchronous_inflows,
)
from .priority import calculate_priority
from .resilience import analyse_resilience
from .roles import score_roles
from .temporal import temporal_features
from .validation import NODE_COLUMNS, validate_completeness_output, validate_outputs, validate_pattern_outputs


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    expected_nodes: int = 2248,
    logger: Callable[[str], None] = print,
) -> dict[str, object]:
    started = time.perf_counter()
    stage_times: dict[str, float] = {}

    def stage(name: str, since: float) -> float:
        elapsed = time.perf_counter() - since
        stage_times[name] = round(elapsed, 4)
        logger(f"[{name}] {elapsed:.2f}s")
        return time.perf_counter()

    tick = time.perf_counter()
    data = load_data(data_dir)
    eda = describe_data(data)
    logger("EDA: " + json.dumps(eda, ensure_ascii=False, default=str))
    tick = stage("load_validate", tick)

    graph, edges = build_graph(data.nodes, data.edges)
    tick = stage("graph", tick)
    features = graph_features(graph, data.nodes)
    tick = stage("network_features", tick)
    temporal = temporal_features(data.transactions, data.nodes["gid"])
    features = features.merge(temporal, on="gid", how="left")
    tick = stage("temporal_features", tick)

    cycles, cycles_truncated = detect_cycles(graph)
    recurring_routes = detect_recurring_routes(data.transactions)
    recurring_chains = detect_recurring_chains(data.transactions)
    synchronous_inflows = detect_synchronous_inflows(data.transactions)
    structuring_events = detect_structuring_events(data.transactions)
    features = add_pattern_features(features, cycles, recurring_routes, recurring_chains, structuring_events)
    tick = stage("aml_patterns", tick)

    communities = detect_communities(graph)
    features = add_cluster_features(features, edges, communities)
    tick = stage("louvain", tick)
    features = score_roles(features)
    tick = stage("role_scores", tick)
    features = calculate_priority(features, cluster_importance(features, edges))
    features = add_completeness(
        features, has_timestamps=bool(data.transactions["timestamp"].notna().any())
    )
    features = add_evidence(features)
    tick = stage("priority_evidence", tick)

    nodes_roles = features[NODE_COLUMNS].sort_values("gid").reset_index(drop=True)
    top = features.nlargest(min(50, len(features)), "priority_score").reset_index(drop=True)
    top_nodes = pd.DataFrame(
        {
            "rank": top.index + 1,
            "gid": top["gid"],
            "role": top["role"],
            "priority_score": top["priority_score"],
            "why": top["evidence"],
        }
    )
    clusters = build_cluster_report(features, edges)
    anomalies = build_anomaly_report(features)
    completeness = completeness_report(features)
    report = validate_outputs(nodes_roles, clusters, top_nodes, expected_nodes=expected_nodes)
    validate_pattern_outputs(
        report, cycles, recurring_routes, recurring_chains, synchronous_inflows, structuring_events, anomalies
    )
    validate_completeness_output(report, completeness, expected_nodes=expected_nodes)
    tick = stage("validation", tick)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nodes_roles.to_csv(output / "nodes_roles.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(output / "clusters.csv", index=False, encoding="utf-8-sig")
    top_nodes.to_csv(output / "top_nodes.csv", index=False, encoding="utf-8-sig")
    features.to_parquet(output / "node_features.parquet", index=False)
    edges.to_parquet(output / "graph_edges.parquet", index=False)
    resilience = analyse_resilience(graph, features)
    resilience.to_csv(output / "resilience.csv", index=False, encoding="utf-8-sig")
    cycles.to_csv(output / "cycles.csv", index=False, encoding="utf-8-sig")
    recurring_routes.to_csv(output / "recurring_routes.csv", index=False, encoding="utf-8-sig")
    recurring_chains.to_csv(output / "recurring_chains.csv", index=False, encoding="utf-8-sig")
    synchronous_inflows.to_csv(output / "synchronous_inflows.csv", index=False, encoding="utf-8-sig")
    structuring_events.to_csv(output / "structuring_events.csv", index=False, encoding="utf-8-sig")
    anomalies.to_csv(output / "anomalies.csv", index=False, encoding="utf-8-sig")
    completeness.to_csv(output / "completeness.csv", index=False, encoding="utf-8-sig")

    runtime = time.perf_counter() - started
    metadata = {
        "runtime_seconds": round(runtime, 3), "stage_seconds": stage_times,
        "eda": eda, "role_distribution": features["role"].value_counts().to_dict(),
        "clusters": len(clusters), "cycles": len(cycles), "cycles_truncated": cycles_truncated,
        "recurring_routes": len(recurring_routes), "recurring_chains": len(recurring_chains),
        "synchronous_inflows": len(synchronous_inflows), "structuring_events": len(structuring_events),
        "anomalous_nodes": len(anomalies),
        "validation": report.checks,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger(f"[export] {runtime:.2f}s total; outputs: {output.resolve()}")
    return {
        "features": features, "clusters": clusters, "top_nodes": top_nodes,
        "cycles": cycles, "recurring_routes": recurring_routes, "recurring_chains": recurring_chains,
        "synchronous_inflows": synchronous_inflows, "structuring_events": structuring_events,
        "anomalies": anomalies, "completeness": completeness,
        "metadata": metadata,
    }
