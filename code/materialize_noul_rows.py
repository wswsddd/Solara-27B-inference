#!/usr/bin/env python3
"""Write a rows file containing only noul-type decision requests from the frozen suite."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_index.suite.io import Suite, dumps  # noqa: E402


def row_matches(row: dict, mode: str) -> bool:
    qs = row.get("questions") or {}
    if not qs:
        return False
    types = [q.get("type") for q in qs.values()]
    if mode == "pure":
        return all(t == "noul" for t in types)
    if mode == "scored":
        from decision_index.scoring.added import scored_keys

        n = row["_evaluation"]["catalog_id"]
        keys = scored_keys(n, row)
        return bool(keys) and all(row["questions"][k]["type"] == "noul" for k in keys)
    return any(t == "noul" for t in types)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--edition", default="0.3")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("pure", "scored", "any"),
        default="scored",
        help="pure=all questions noul; scored=official noul metrics only (RAGTruth); any=any noul field",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    meta_path = Path(str(args.out) + ".meta.json")
    if args.out.exists() and not args.force and meta_path.exists():
        print(meta_path.read_text(encoding="utf-8"))
        return

    suite = Suite(args.suite_dir, args.edition)
    suite.verify(strict=True)
    rows = [r for r in suite.rows(apply_exclusions=True) if row_matches(r, args.mode)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(args.out).endswith(".gz") else open
    with opener(args.out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps(r) + "\n")

    by_bench: dict[int, int] = {}
    for r in rows:
        n = r["_evaluation"]["catalog_id"]
        by_bench[n] = by_bench.get(n, 0) + 1

    meta = {
        "edition": args.edition,
        "suite_dir": str(args.suite_dir),
        "mode": args.mode,
        "requests": len(rows),
        "benchmarks": {str(k): v for k, v in sorted(by_bench.items())},
        "path": str(args.out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
