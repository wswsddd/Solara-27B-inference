#!/usr/bin/env python3
"""Score noul-focused runs (binary / RAGTruth F1), not the full Decision Index."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_index.scoring.added import report as added_report  # noqa: E402
from decision_index.scoring.report import load_results  # noqa: E402
from decision_index.suite.io import Suite, atomic_json, read_jsonl  # noqa: E402


def load_row_index(rows_path: Path) -> dict[int, list]:
    by_catalog: dict[int, list] = defaultdict(list)
    for row in read_jsonl(rows_path):
        by_catalog[row["_evaluation"]["catalog_id"]].append(row)
    return by_catalog


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True, help="noul rows file used for the run")
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--edition", default="0.3")
    args = p.parse_args()

    results_path = args.run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    results = load_results(results_path)
    by_catalog = load_row_index(args.rows)
    Suite(args.suite_dir, args.edition)  # validate suite dir / edition

    benchmarks = []
    for n in sorted(by_catalog):
        rows = by_catalog[n]
        rep = added_report(n, rows, results)
        benchmarks.append(rep)

    total_req = sum(b["requests"] for b in benchmarks)
    answered = sum(b["answered"] for b in benchmarks)
    failed = total_req - answered - sum(b.get("pending", 0) for b in benchmarks)

    scores = {
        "kind": "noul_subset",
        "edition": args.edition,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_file": str(args.rows),
        "suite_dir": str(args.suite_dir),
        "requests": total_req,
        "answered": answered,
        "complete": answered >= total_req,
        "benchmarks": benchmarks,
    }
    if benchmarks:
        scores["macro_metric"] = statistics.mean(b["score"] for b in benchmarks if b.get("score") is not None)

    atomic_json(args.run_dir / "noul-summary.json", scores)
    print(json.dumps({"complete": scores["complete"], "macro_metric": scores.get("macro_metric"), "out": str(args.run_dir / "noul-summary.json")}, indent=2))


if __name__ == "__main__":
    main()
