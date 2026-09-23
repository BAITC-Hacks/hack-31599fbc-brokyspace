from __future__ import annotations

import json
import re

import pandas as pd


def test_must_have_contract(pipeline_output):
    nodes = pd.read_csv(pipeline_output / "nodes_roles.csv")
    clusters = pd.read_csv(pipeline_output / "clusters.csv")
    top = pd.read_csv(pipeline_output / "top_nodes.csv")
    required = {"gid", "role", "role_score", "cluster_id", "priority_score", "evidence"}
    assert len(nodes) == 2248
    assert nodes["gid"].is_unique
    assert required.issubset(nodes.columns)
    assert not nodes[list(required)].isna().any().any()
    assert nodes["role_score"].between(0, 1).all()
    assert nodes["priority_score"].between(0, 1).all()
    assert nodes["evidence"].str.len().le(200).all()
    assert nodes["evidence"].str.contains(r"\d", regex=True).all()
    assert clusters["n_nodes"].sum() == 2248
    assert len(top) >= 20
    assert top["priority_score"].is_monotonic_decreasing


def test_dataset_limitations_are_enforced(pipeline_output):
    features = pd.read_parquet(pipeline_output / "node_features.parquet")
    truncated = features["truncated_by_depth"]
    assert int(truncated.sum()) == 444
    assert not features.loc[truncated, "role"].eq("terminal").any()
    assert features.loc[features["is_seed"], "pass_through"].eq(0).all()
    seed_transit = features["is_seed"] & features["role"].eq("transit")
    assert features.loc[seed_transit, "evidence"].str.contains("входящим потоком").all()


def test_bonus_outputs(pipeline_output):
    cycles = pd.read_csv(pipeline_output / "cycles.csv")
    routes = pd.read_csv(pipeline_output / "recurring_routes.csv")
    anomalies = pd.read_csv(pipeline_output / "anomalies.csv")
    metadata = json.loads((pipeline_output / "metadata.json").read_text(encoding="utf-8"))
    assert not cycles.empty and cycles["length"].between(2, 6).all()
    assert not routes.empty and routes["active_days"].ge(2).all()
    assert not anomalies.empty and anomalies["reason"].str.strip().ne("").all()
    assert metadata["cycles_truncated"] is False
    assert metadata["runtime_seconds"] < 300


def test_evidence_mentions_real_numbers(pipeline_output):
    nodes = pd.read_csv(pipeline_output / "nodes_roles.csv")
    assert all(re.search(r"\d", text) for text in nodes["evidence"])
