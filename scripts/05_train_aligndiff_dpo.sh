#!/usr/bin/env bash
# Step 5: DPO on AlignDiff-filtered UltraFeedback (paper: 30k, τ=20, lr=5e-7, β=0.01).
# Trainer: OpenRLHF (Appendix G).
# Usage: bash scripts/05_train_aligndiff_dpo.sh qwen|llama [tau]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

MODEL="${1:?usage: $0 qwen|llama [tau]}"
KEY="$(model_tag "${MODEL}")"
TAU="${2:-20}"
SFT="$(sft_model_path "${MODEL}")"
SUFFIX="${TOP_K}"
if [ $((TOP_K % 1000)) -eq 0 ]; then
  SUFFIX="$((TOP_K / 1000))k"
fi

METHODS="${METHODS:-aligndiff_tau${TAU}}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"

for method in "${METHOD_LIST[@]}"; do
  DATA_FILE="${DATA_DIR}/filtered/${KEY}/${method}_${SUFFIX}.jsonl"
  OUT="$(saves_for "${MODEL}")/dpo_${method}_${SUFFIX}"
  if [ ! -s "${DATA_FILE}" ]; then
    echo "Missing ${DATA_FILE}. Run scripts/04_filter_aligndiff.sh ${KEY}" >&2
    exit 1
  fi
  if [ "${SKIP_DONE}" = "1" ] && [ "${FORCE}" != "1" ] && weights_ok "${OUT}"; then
    log "SKIP ${KEY} ${method}: ${OUT}"
    continue
  fi
  log "===== ${KEY} DPO ${method}  data=${DATA_FILE}  out=${OUT} ====="
  orhf_dpo "${SFT}" "${DATA_FILE}" "${OUT}"
  log "DONE ${KEY} ${method}"
done
