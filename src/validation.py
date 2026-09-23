from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


ALLOWED_ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]


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
