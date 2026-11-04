#!/usr/bin/env python3
"""Merge shard */results.jsonl into one results.jsonl (last shard wins on duplicate run_id)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_index.suite.io import dumps, read_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="directory containing shard-0, shard-1, ...")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--out", type=Path, help="default: RUN_DIR/results.jsonl")
    args = p.parse_args()

    out_path = args.out or (args.run_dir / "results.jsonl")
    by_id: dict[str, dict] = {}
    for i in range(args.num_shards):
        path = args.run_dir / f"shard-{i}" / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path, complete_lines_only=True):
            by_id[row["run_id"]] = row

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rid in sorted(by_id):
            f.write(dumps(by_id[rid]) + "\n")
    tmp.replace(out_path)

    env_src = args.run_dir / "shard-0" / "environment.json"
    if env_src.exists():
        (args.run_dir / "environment.json").write_bytes(env_src.read_bytes())

    print(json.dumps({"merged": len(by_id), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
