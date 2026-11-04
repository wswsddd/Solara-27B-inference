#!/usr/bin/env python3
"""Score merged results.jsonl and write index.json / scores.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_index.pipeline import score_run  # noqa: E402
from decision_index.suite.io import Suite  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--edition", default="0.3")
    p.add_argument("--engine", default="transformers")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    results = args.run_dir / "results.jsonl"
    if not results.exists():
        raise FileNotFoundError(results)

    suite = Suite(args.suite_dir, args.edition)
    if not args.no_verify:
        suite.verify(strict=True)

    scores = score_run(suite, results, args.engine, args.run_dir)
    print(json.dumps({"decision_index": scores["decision_index"], "complete": scores["complete"], "out": str(args.run_dir)}, indent=2))


if __name__ == "__main__":
    main()
