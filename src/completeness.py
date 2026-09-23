from __future__ import annotations

import pandas as pd


COMPLETENESS_COLUMNS = ["gid", "completeness_score", "observed_gaps", "recommended_request"]


def add_completeness(features: pd.DataFrame, *, has_timestamps: bool) -> pd.DataFrame:
    """Explain what remains unobserved and what an analyst should request next."""
    result = features.copy()
    scores: list[float] = []
    gaps: list[str] = []
    requests: list[str] = []
    for row in result.itertuples(index=False):
        score = 0.75
        node_gaps = ["нет межбанковских потоков и переводов <5 000 KZT", "нет KYC и остатков"]
        node_requests = []
        if bool(row.is_seed):
            score -= 0.20
            node_gaps.append("входящий поток seed неполон")
            node_requests.append("полную историю входящих seed")
        if bool(row.truncated_by_depth):
            score -= 0.25
            node_gaps.append("исходящий поток обрезан depth=4")
            node_requests.append("исходящие переводы следующего колена")
        if not bool(row.is_seed) and float(row.out_kzt) > float(row.in_kzt) * 1.05:
            score -= 0.10
            node_gaps.append("наблюдаемый выход превышает вход")
            node_requests.append("расширенный период входящих операций")
        if not has_timestamps:
            score -= 0.15
            node_gaps.append("нет времени транзакций")
            node_requests.append("timestamps операций")
        node_requests.extend(["межбанковские переводы", "остатки и назначение платежей"])
        scores.append(max(0.0, min(score, 1.0)))
        gaps.append("; ".join(node_gaps))
        requests.append("Запросить: " + ", ".join(dict.fromkeys(node_requests)) + ".")
    result["completeness_score"] = scores
    result["observed_gaps"] = gaps
    result["recommended_request"] = requests
    return result


def completeness_report(features: pd.DataFrame) -> pd.DataFrame:
    return features[COMPLETENESS_COLUMNS].sort_values(
        ["completeness_score", "gid"], ascending=[True, True]
    ).reset_index(drop=True)
