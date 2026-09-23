from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALIASES = {
    "gid": ("gid", "node_id", "client_id", "id"),
    "source": ("source", "src", "source_gid", "from_gid", "sender_gid", "sender", "from"),
    "target": ("target", "dst", "target_gid", "to_gid", "receiver_gid", "receiver", "to"),
    "amount": ("sum_kzt", "amount_kzt", "amount", "total_amount", "volume", "sum"),
    "tx_count": ("n_tx", "tx_count", "transactions", "count", "n_transactions"),
    "timestamp": ("timestamp", "datetime", "transaction_time", "transaction_date", "date", "created_at"),
    "depth": ("depth", "hop", "level"),
    "is_seed": ("is_seed", "seed", "seed_flag", "is_suspicious", "target_flag"),
}


@dataclass(frozen=True)
class InputData:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    transactions: pd.DataFrame


def _pick(columns: Iterable[str], kind: str, required: bool = True) -> str | None:
    lookup = {str(c).lower().strip(): c for c in columns}
    for candidate in ALIASES[kind]:
        if candidate in lookup:
            return lookup[candidate]
    if required:
        raise ValueError(f"Не найдена колонка '{kind}'. Доступны: {list(columns)}")
    return None


def _normalise_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "да", "seed"})


def _canonical_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    gid = _pick(frame.columns, "gid")
    depth = _pick(frame.columns, "depth", False)
    seed = _pick(frame.columns, "is_seed", False)
    if frame[gid].isna().any():
        raise ValueError("nodes.parquet: gid содержит null")
    result = pd.DataFrame({"gid": frame[gid].astype(str).str.strip()})
    result["depth"] = pd.to_numeric(frame[depth], errors="coerce") if depth else np.nan
    result["is_seed"] = _normalise_bool(frame[seed]) if seed else False
    if result["gid"].eq("").any() or result["gid"].duplicated().any():
        raise ValueError("nodes.parquet: gid должен быть непустым и уникальным")
    return result


def _canonical_flows(frame: pd.DataFrame, transactions: bool) -> pd.DataFrame:
    source = _pick(frame.columns, "source")
    target = _pick(frame.columns, "target")
    amount = _pick(frame.columns, "amount")
    count = _pick(frame.columns, "tx_count", False)
    timestamp = _pick(frame.columns, "timestamp", False) if transactions else None
    if frame[[source, target]].isna().any().any():
        raise ValueError(f"{'transactions' if transactions else 'edges'}.parquet: source/target содержит null")
    result = pd.DataFrame(
        {
            "source": frame[source].astype(str).str.strip(),
            "target": frame[target].astype(str).str.strip(),
            "sum_kzt": pd.to_numeric(frame[amount], errors="coerce"),
        }
    )
    result["n_tx"] = pd.to_numeric(frame[count], errors="coerce") if count else 1
    if transactions:
        result["timestamp"] = pd.to_datetime(frame[timestamp], errors="coerce", utc=True) if timestamp else pd.NaT
    bad = result[["sum_kzt", "n_tx"]].isna().any(axis=1) | result["source"].eq("") | result["target"].eq("")
    if bad.any():
        raise ValueError(f"{'transactions' if transactions else 'edges'}.parquet: {int(bad.sum())} некорректных строк")
    if (result[["sum_kzt", "n_tx"]] < 0).any().any():
        raise ValueError("Суммы и количества транзакций не могут быть отрицательными")
    return result


def load_data(data_dir: str | Path) -> InputData:
    root = Path(data_dir)
    required = {name: root / f"{name}.parquet" for name in ("nodes", "edges", "transactions")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Отсутствуют входные файлы:\n- " + "\n- ".join(missing))
    nodes = _canonical_nodes(pd.read_parquet(required["nodes"]))
    edges = _canonical_flows(pd.read_parquet(required["edges"]), transactions=False)
    transactions = _canonical_flows(pd.read_parquet(required["transactions"]), transactions=True)

    known = set(nodes["gid"])
    referenced = set(edges["source"]) | set(edges["target"]) | set(transactions["source"]) | set(transactions["target"])
    unknown = referenced - known
    if unknown:
        raise ValueError(f"В потоках найдено {len(unknown)} gid, которых нет в nodes.parquet")
    return InputData(nodes, edges, transactions)


def describe_data(data: InputData) -> dict[str, object]:
    ts = data.transactions["timestamp"].dropna()
    return {
        "nodes": len(data.nodes),
        "edges_rows": len(data.edges),
        "transaction_rows": len(data.transactions),
        "seed_nodes": int(data.nodes["is_seed"].sum()),
        "depth_distribution": data.nodes["depth"].value_counts(dropna=False).sort_index().to_dict(),
        "edge_turnover_kzt": float(data.edges["sum_kzt"].sum()),
        "transaction_turnover_kzt": float(data.transactions["sum_kzt"].sum()),
        "period_start": ts.min().isoformat() if len(ts) else None,
        "period_end": ts.max().isoformat() if len(ts) else None,
    }
