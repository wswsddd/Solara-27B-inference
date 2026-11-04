#!/usr/bin/env bash
# Evaluate an already-deployed SGLang service. This script starts API clients,
# not model servers. No SSH or GPU allocation is needed on the client host.
# Usage: SGLANG_BASE_URLS='http://host:8000/v1' SGLANG_MODEL=served-name \
#          bash scripts/run_eval_sglang.sh /path/to/matching/tokenizer model_tag
set -euo pipefail

BASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/share/project/wangsen/envs/decision-index/bin/python}
TOKENIZER_PATH=${1:?"Usage: $0 TOKENIZER_PATH MODEL_TAG"}
TAG=${2:?"MODEL_TAG is required"}
: "${SGLANG_MODEL:?Set SGLANG_MODEL to the model name exposed by the service}"
URLS=${SGLANG_BASE_URLS:-${SGLANG_BASE_URL:-}}
[[ -n "$URLS" ]] || { echo "Set SGLANG_BASE_URL or SGLANG_BASE_URLS" >&2; exit 2; }
read -r -a ENDPOINTS <<<"${URLS//,/ }"
WORKERS=${WORKERS:-16}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be positive" >&2; exit 2; }
EDITION=${EDITION:-0.3}
SUITE_DIR=${SUITE_DIR:-"$BASE_DIR/suite-$EDITION"}
RUN_DATE=${RUN_DATE:-$(date +%m%d)_sglang}
[[ "$RUN_DATE" =~ ^[0-9]{4,8}(_[A-Za-z0-9]+)?$ ]] || { echo "Invalid RUN_DATE" >&2; exit 2; }
OUT_DIR="$BASE_DIR/runs/$TAG/$RUN_DATE"
mkdir -p "$OUT_DIR/launch"

READ_TWICE=${READ_TWICE:-0}
[[ "$READ_TWICE" =~ ^[01]$ ]] || { echo "READ_TWICE must be 0 or 1" >&2; exit 2; }

extra=(--label-mode "${LABEL_MODE:-unified_letters}" --logprob-backend "${LOGPROB_BACKEND:-auto}"
       --top-logprobs "${TOP_LOGPROBS:-20}" --request-timeout "${REQUEST_TIMEOUT:-600}")
[[ -n "${LABEL_ALPHABET:-}" ]] && extra+=(--label-alphabet "$LABEL_ALPHABET")
[[ "$READ_TWICE" == 1 ]] && extra+=(--read-twice)
[[ "${COMPACT:-0}" == 1 ]] && extra+=(--compact)
[[ "${FRESH:-0}" == 1 ]] && extra+=(--fresh)
[[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
[[ -n "${SAMPLE_ROWS:-}" ]] && extra+=(--rows "$SAMPLE_ROWS")

echo "SGLang model: $SGLANG_MODEL; API client workers: $WORKERS"
echo "Read Twice: $READ_TWICE (complete user prompt; system once)"
echo "Endpoints: ${ENDPOINTS[*]}"
echo "Output: $OUT_DIR"
pids=()
for ((shard = 0; shard < WORKERS; shard++)); do
  endpoint=${ENDPOINTS[shard % ${#ENDPOINTS[@]}]}
  "$PYTHON" "$BASE_DIR/code/run_shard.py" \
    --engine sglang --tokenizer-path "$TOKENIZER_PATH" --model-name "$SGLANG_MODEL" \
    --base-url "$endpoint" --model-tag "$TAG" --suite-dir "$SUITE_DIR" --edition "$EDITION" \
    --shard "$shard" --num-shards "$WORKERS" --out "$OUT_DIR/shard-$shard" \
    "${extra[@]}" >"$OUT_DIR/launch/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
((failed == 0)) || { echo "API client failed; see $OUT_DIR/launch/" >&2; exit 1; }
"$PYTHON" "$BASE_DIR/code/merge_results.py" --run-dir "$OUT_DIR" --num-shards "$WORKERS"
if [[ "${SCORE:-1}" != 0 ]]; then
  "$PYTHON" "$BASE_DIR/code/score_run.py" --run-dir "$OUT_DIR" --suite-dir "$SUITE_DIR" --edition "$EDITION" --engine sglang
fi
echo "Done: $OUT_DIR"
