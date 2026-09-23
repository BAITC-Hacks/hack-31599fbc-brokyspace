from __future__ import annotations

import argparse
import sys

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainable AML transaction-graph pipeline")
    parser.add_argument("--data", default="./data", help="Каталог с nodes/edges/transactions.parquet")
    parser.add_argument("--out", default="./out", help="Каталог результатов")
    parser.add_argument("--expected-nodes", type=int, default=2248, help="Ожидаемое число узлов; 0 отключает проверку")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_pipeline(args.data, args.out, expected_nodes=args.expected_nodes)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
