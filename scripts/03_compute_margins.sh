#!/usr/bin/env bash
# Step 3: IM (π_θ^pos vs π_sft) and RAD (π_θ^pos vs π_θ^inv) on UltraFeedback.
# Usage: bash scripts/03_compute_margins.sh qwen|llama
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

MODEL="${1:?usage: $0 qwen|llama}"
KEY="$(model_tag "${MODEL}")"
DPO="$(saves_for "${MODEL}")/dpo"
INV="$(saves_for "${MODEL}")/dpo_flipped"
SFT="$(sft_model_path "${MODEL}")"
OUT_DIR="${DATA_DIR}/margins"
IM_OUT="${OUT_DIR}/${KEY}_im_margins.jsonl"
RAD_OUT="${OUT_DIR}/${KEY}_rad_margins.jsonl"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
TOKEN_WORKERS="${TOKEN_WORKERS:-16}"

if [ ! -s "${UF_JSONL}" ]; then
  echo "Missing ${UF_JSONL}. Run scripts/01_prepare_data.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
NGPU="$(n_gpus)"

score() {
  local label="$1" policy="$2" ref="$3" out="$4" extra="${5:-}"
  if [ -s "${out}" ] && [ "${FORCE}" != "1" ]; then
    log "${label}: skip existing ${out}"
    return 0
  fi
  log "${label}: policy=${policy}  ref=${ref}"
  # shellcheck disable=SC2086
  "${PYTHON}" -m aligndiff.compute_margins \
    --data-path "${UF_JSONL}" \
    --policy-model "${policy}" \
    --ref-model "${ref}" \
    --output-path "${out}" \
    --batch-size "${BATCH_SIZE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --max-length 2048 \
    --num-shards "${NGPU}" \
    --gpus "${GPUS}" \
    --token-workers "${TOKEN_WORKERS}" \
    ${extra}
}

score "${KEY} IM" "${DPO}" "${SFT}" "${IM_OUT}"
score "${KEY} RAD" "${DPO}" "${INV}" "${RAD_OUT}" "--save-rad"
log "Margins written to ${OUT_DIR}"
