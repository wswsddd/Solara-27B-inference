# Source snapshot

Consolidated on 2026-10-14 into a single project root.

- Core source: `/share/project/wangsen/PeishengJin/projects/decision-index`.
- Upstream: https://github.com/apolinario/decision-index.git
- Upstream HEAD: `9eb2dbe2a358004c8782c66e40a83ac07b953fec`.
- Existing local changes in `decision_index/__init__.py`, `decision_index/engines/transformers_engine.py`, and `decision_index/suite/build/acquire.py` were copied as-is. No original source files were reset.
- Multi-node wrappers and helpers: `/share/project/wangsen/PeishengJin/projects/decision-index-eval`.
- v46 outer launcher: `searchllm/code/phase2/constrained-ir/llm_hard/scripts/0921/pointwise/train/v46/eval_v46_12nodes.sh`.
- The original `code/upstream` symlink was replaced by the root-level `decision_index/` package. Python helper imports and shell project-root discovery were adjusted for this layout.
- Core LICENSE, pyproject, tests, docs, hub metadata and preparation script are included. Original Git metadata, build artifacts, caches and runtime logs are excluded from the copy.
- `suite-0.3/` and historical subset rows were copied for local use and are ignored by Git. Full historical run outputs and model weights were not copied.
- Compact CLINC diagnostic reports are retained under `test/reports/clinc/` with their original results and provenance. Diagnostic programs and their launcher are under `test/`; upstream unit tests are under `test/unit/`. All of `test/` is ignored by Git.

The initial source snapshot preserved its evaluation behavior. The consolidated
Transformers engine now uses `independent-key-eos-v1`: independent native
candidate histories, complete multi-token keys including tokenizer EOS, and a
single-token fast path. Thinking defaults to off. The runner rejects incompatible
scoring provenance on resume; explicit fresh runs replace previous predictions.
Historical packed behavior is retained only in the ignored
`test/legacy_packed_engine.py`. CPU regressions and real v46 sample validation
cover the implementation; a new full-suite score has not been produced by this
code-change task.
