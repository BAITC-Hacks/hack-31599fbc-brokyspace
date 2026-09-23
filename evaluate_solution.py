from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agent import AMLAnalystAgent


def evaluate(output_dir: Path) -> dict[str, object]:
    nodes = pd.read_csv(output_dir / "nodes_roles.csv")
    clusters = pd.read_csv(output_dir / "clusters.csv")
    top = pd.read_csv(output_dir / "top_nodes.csv")
    features = pd.read_parquet(output_dir / "node_features.parquet")
    cycles = pd.read_csv(output_dir / "cycles.csv")
    routes = pd.read_csv(output_dir / "recurring_routes.csv")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    answer = AMLAnalystAgent(output_dir).ask("Кого из 2248 клиентов смотреть первым и почему?")
    checks = {
        "2248_unique_nodes": len(nodes) == 2248 and nodes["gid"].is_unique,
        "mandatory_fields_complete": not nodes.isna().any().any(),
        "scores_valid": nodes["role_score"].between(0, 1).all() and nodes["priority_score"].between(0, 1).all(),
        "evidence_numeric_and_short": nodes["evidence"].str.contains(r"\d").all() and nodes["evidence"].str.len().le(200).all(),
        "clusters_cover_all_nodes": int(clusters["n_nodes"].sum()) == 2248,
        "top_sorted": len(top) >= 20 and top["priority_score"].is_monotonic_decreasing,
        "depth_truncation_safe": not features.loc[features["truncated_by_depth"], "role"].eq("terminal").any(),
        "seed_pass_through_safe": features.loc[features["is_seed"], "pass_through"].eq(0).all(),
        "bonus_patterns_present": len(cycles) > 0 and len(routes) > 0 and metadata["cycles_truncated"] is False,
        "runtime_under_5_minutes": float(metadata["runtime_seconds"]) < 300,
        "agent_grounded": "priority=" in answer.text and len(answer.sources) > 0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "all_checks_passed": all(checks.values()),
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "checks": checks,
        "runtime_seconds": metadata["runtime_seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hackathon acceptance gates")
    parser.add_argument("--out", default="./out")
    args = parser.parse_args()
    report = evaluate(Path(args.out))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_checks_passed"] else 1)
