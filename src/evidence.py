from __future__ import annotations

import pandas as pd


def _money(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} млрд"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн"
    if value >= 1_000:
        return f"{value / 1_000:.1f} тыс."
    return f"{value:.0f}"


def _top(pct: float) -> str:
    return f"top {max(0.1, (1 - pct) * 100):.1f}%"


def explain(row: pd.Series) -> str:
    role = row["role"]
    if role == "consolidator":
        text = (f"Получает от {int(row.in_deg)} отправителей ({_top(row.in_deg_pct)}), вход {_money(row.in_kzt)} KZT; "
                f"наблюдаемый остаток {max(0, 1-row.out_kzt/max(row.in_kzt, 1))*100:.0f}%.")
    elif role == "distributor":
        text = (f"Отправляет {int(row.out_deg)} получателям ({_top(row.out_deg_pct)}), {int(row.out_tx)} переводов "
                f"на {_money(row.out_kzt)} KZT; выраженный fan-out.")
    elif role == "transit":
        text = (f"Получено {_money(row.in_kzt)}, отправлено {_money(row.out_kzt)} KZT; совпадение потоков "
                f"{row.pass_through*100:.0f}%, быстрое перенаправление {row.rapid_pass_through*100:.0f}%.")
    elif role == "terminal":
        text = (f"Получено {_money(row.in_kzt)} KZT от {int(row.in_deg)} отправителей, исходящий поток "
                f"{_money(row.out_kzt)} KZT; средства преимущественно остаются.")
    elif role == "coordinator":
        text = (f"Betweenness {_top(row.betweenness_pct)}, PageRank {_top(row.pagerank_pct)}; "
                f"межкластерных связей {int(row.cross_cluster_edges)}, близость к seed {row.seed_connectivity:.2f}.")
    else:
        text = (f"Периферийный узел: входящих/исходящих связей {int(row.in_deg)}/{int(row.out_deg)}, "
                f"оборот {_money(row.in_kzt + row.out_kzt)} KZT; сильной структурной роли нет.")
    patterns = []
    if int(row.get("cycle_count", 0)):
        patterns.append(f"циклов {int(row.cycle_count)}")
    if int(row.get("recurring_route_count", 0)):
        patterns.append(f"повторных маршрутов {int(row.recurring_route_count)}")
    if patterns:
        text += " Паттерны: " + ", ".join(patterns) + "."
    if bool(row.truncated_by_depth):
        text += " Граница depth=4: исходящий поток может быть усечён."
    return text[:200]


def add_evidence(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result["evidence"] = result.apply(explain, axis=1)
    return result
