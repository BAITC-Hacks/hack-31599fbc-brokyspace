from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


ALLOWED_ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
CYCLE_COLUMNS = ["cycle_id", "length", "gids", "sum_kzt_total", "sum_kzt_bottleneck", "contains_seed", "cycle_score"]
ROUTE_COLUMNS = ["source", "target", "n_tx", "active_days", "span_days", "sum_kzt", "cadence_regularity", "recurrence_score"]
ANOMALY_COLUMNS = ["gid", "role", "priority_score", "temporal_anomaly_score", "reason"]


@dataclass
class ValidationReport:
    checks: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"VALIDATION FAILED: {message}")
        self.checks.append(f"OK: {message}")


def validate_outputs(nodes: pd.DataFrame, clusters: pd.DataFrame, top: pd.DataFrame, expected_nodes: int = 2248) -> ValidationReport:
    report = ValidationReport()
    if expected_nodes > 0:
        report.require(len(nodes) == expected_nodes, f"ровно {expected_nodes} узлов")
    report.require(set(NODE_COLUMNS).issubset(nodes.columns), "схема nodes_roles.csv")
    report.require(not nodes[NODE_COLUMNS].isna().any().any(), "нет null в обязательных полях узлов")
    report.require(nodes["gid"].is_unique, "каждый gid встречается ровно один раз")
    report.require(set(nodes["role"]).issubset(ALLOWED_ROLES), "только допустимые роли")
    report.require(nodes["role_score"].between(0, 1).all(), "role_score в диапазоне [0,1]")
    report.require(nodes["priority_score"].between(0, 1).all(), "priority_score в диапазоне [0,1]")
    report.require(nodes["evidence"].astype(str).str.strip().ne("").all(), "evidence заполнен")
    report.require(nodes["cluster_id"].notna().all(), "cluster_id заполнен")
    report.require(set(CLUSTER_COLUMNS).issubset(clusters.columns) and len(clusters) > 0, "clusters.csv заполнен")
    report.require(set(TOP_COLUMNS).issubset(top.columns) and len(top) >= 20, "top_nodes.csv содержит не менее 20 строк")
    report.require(top["priority_score"].is_monotonic_decreasing, "TOP отсортирован по priority_score")
    return report


def validate_pattern_outputs(
    report: ValidationReport,
    cycles: pd.DataFrame,
    routes: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    report.require(set(CYCLE_COLUMNS).issubset(cycles.columns), "схема cycles.csv")
    report.require(set(ROUTE_COLUMNS).issubset(routes.columns), "схема recurring_routes.csv")
    report.require(set(ANOMALY_COLUMNS).issubset(anomalies.columns), "схема anomalies.csv")
    report.require(cycles["cycle_score"].between(0, 1).all(), "cycle_score в диапазоне [0,1]")
    report.require(routes["recurrence_score"].between(0, 1).all(), "recurrence_score в диапазоне [0,1]")
    report.require(anomalies["temporal_anomaly_score"].between(0, 1).all(), "anomaly score в диапазоне [0,1]")
