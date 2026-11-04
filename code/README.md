# code/

| Path | Role |
|---|---|
| `../decision_index/` | 本项目的核心评测包，实体目录。项目根目录安装：`pip install -e '.[transformers]'`。 |
| `run_shard.py` | One data-parallel shard: `hash(run_id) % num_shards == shard`; local transformers or remote `sglang` next-token probabilities. |
| `merge_results.py` | Combine `shard-*/results.jsonl` → `results.jsonl`. |
| `score_run.py` | Run official scorers + public index on merged results. |
| `materialize_noul_rows.py` | Extract noul rows (`scored` ≈ RAGTruth for 0.3). |
| `score_noul_run.py` | Noul subset metrics → `noul-summary.json`. |

Python helpers prepend the project root to `sys.path`, so they import this checkout's `decision_index` without relying on an external source directory or global installation.
