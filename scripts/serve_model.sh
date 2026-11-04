#!/usr/bin/env bash
# Submission target: one NVIDIA H100 80GB, one BF16 Solara-27B model.
set -euo pipefail
MODEL_PATH=${1:?Usage: bash scripts/serve_model.sh /path/to/model}
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config" >&2; exit 1; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ "$CUDA_VISIBLE_DEVICES" != *,* && -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != -1 ]] || {
  echo "Select exactly one GPU with CUDA_VISIBLE_DEVICES (default: 0)" >&2
  exit 2
}
export TOKENIZERS_PARALLELISM=false
exec python -m decision_index.systemone_server \
  --tokenizer "$MODEL_PATH" \
  --model Solara-27B \
  --device cuda \
  --host 127.0.0.1 --port "${PORT:-8008}"
