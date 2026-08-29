#!/usr/bin/env bash
# Step 4: AlignDiff filter — RAD+τ keep/flip, then ANG top-K (paper: τ=20, K=30k).
# Usage: bash scripts/04_filter_aligndiff.sh qwen|llama
# Env: TAUS=20  TOP_K=30000  CURRICULUM=none|easy-to-hard
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

MODEL="${1:?usage: $0 qwen|llama}"
KEY="$(model_tag "${MODEL}")"
IM="${DATA_DIR}/margins/${KEY}_im_margins.jsonl"
RAD="${DATA_DIR}/margins/${KEY}_rad_margins.jsonl"
OUT_DIR="${DATA_DIR}/filtered/${KEY}"
CURRICULUM="${CURRICULUM:-none}"
TOKENIZER_FLAG=()

if [ "${KEY}" = "qwen" ] && [ -n "${TOKENIZER_QWEN:-}" ]; then
  TOKENIZER_FLAG=(--tokenizer "${TOKENIZER_QWEN}")
elif [ "${KEY}" = "llama" ] && [ -n "${TOKENIZER_LLAMA:-}" ]; then
  TOKENIZER_FLAG=(--tokenizer "${TOKENIZER_LLAMA}")
else
  TOKENIZER_FLAG=(--tokenizer "$(sft_model_path "${MODEL}")")
fi

IFS=',' read -r -a TAU_LIST <<< "${TAUS}"
log "AlignDiff filter ${KEY}  TOP_K=${TOP_K}  TAUS=${TAUS}  curriculum=${CURRICULUM}"

for tau in "${TAU_LIST[@]}"; do
  log "===== ${KEY} tau=${tau} ====="
  "${PYTHON}" -m aligndiff.filter \
    --im-margins "${IM}" \
    --rad-margins "${RAD}" \
    --out-dir "${OUT_DIR}" \
    --top-k "${TOP_K}" \
    --model-key "${KEY}" \
    --methods aligndiff \
    --stage1 \
    --tau "${tau}" \
    --ang-descending \
    --curriculum "${CURRICULUM}" \
    "${TOKENIZER_FLAG[@]}"
done

log "Filtered JSONL in ${OUT_DIR}"
