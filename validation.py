from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


ALLOWED_ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
CYCLE_COLUMNS = ["cycle_id", "length", "gids", "sum_kzt_total", "sum_kzt_bottleneck", "contains_seed", "cycle_score"]
ROUTE_COLUMNS = ["source", "target", "n_tx", "active_days", "span_days", "sum_kzt", "cadence_regularity", "recurrence_score"]
CHAIN_COLUMNS = ["source", "middle", "target", "n_occurrences", "active_days", "span_days", "sum_kzt_bottleneck", "cadence_regularity", "recurrence_score"]
SYNC_COLUMNS = ["target", "day", "distinct_senders", "n_tx", "sum_kzt", "synchronous_score"]
STRUCTURING_COLUMNS = ["gid", "direction", "day", "n_tx", "counterparties", "sum_kzt", "mean_kzt", "amount_cv", "near_threshold_share", "round_amount_share", "structuring_score"]
ANOMALY_COLUMNS = ["gid", "role", "priority_score", "aml_anomaly_score", "temporal_anomaly_score", "depth_peer_anomaly_score", "reason"]
COMPLETENESS_COLUMNS = ["gid", "completeness_score", "observed_gaps", "recommended_request"]


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
    chains: pd.DataFrame,
    synchronous: pd.DataFrame,
    structuring: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    report.require(set(CYCLE_COLUMNS).issubset(cycles.columns), "схема cycles.csv")
    report.require(set(ROUTE_COLUMNS).issubset(routes.columns), "схема recurring_routes.csv")
    report.require(set(CHAIN_COLUMNS).issubset(chains.columns), "схема recurring_chains.csv")
    report.require(set(SYNC_COLUMNS).issubset(synchronous.columns), "схема synchronous_inflows.csv")
    report.require(set(STRUCTURING_COLUMNS).issubset(structuring.columns), "схема structuring_events.csv")
    report.require(set(ANOMALY_COLUMNS).issubset(anomalies.columns), "схема anomalies.csv")
    report.require(cycles["cycle_score"].between(0, 1).all(), "cycle_score в диапазоне [0,1]")
    report.require(routes["recurrence_score"].between(0, 1).all(), "recurrence_score в диапазоне [0,1]")
    report.require(chains["recurrence_score"].between(0, 1).all(), "chain recurrence_score в диапазоне [0,1]")
    report.require(synchronous["synchronous_score"].between(0, 1).all(), "synchronous_score в диапазоне [0,1]")
    report.require(structuring["structuring_score"].between(0, 1).all(), "structuring_score в диапазоне [0,1]")
    report.require(anomalies["aml_anomaly_score"].between(0, 1).all(), "anomaly score в диапазоне [0,1]")


def validate_completeness_output(
    report: ValidationReport, completeness: pd.DataFrame, expected_nodes: int = 2248
) -> None:
    report.require(set(COMPLETENESS_COLUMNS).issubset(completeness.columns), "схема completeness.csv")
    if expected_nodes > 0:
        report.require(len(completeness) == expected_nodes, f"completeness покрывает {expected_nodes} узлов")
    report.require(completeness["gid"].is_unique, "completeness содержит уникальные gid")
    report.require(completeness["completeness_score"].between(0, 1).all(), "completeness_score в диапазоне [0,1]")
    report.require(completeness["recommended_request"].str.strip().ne("").all(), "следующий запрос заполнен")
