#!/usr/bin/env bash
# Step 2: seed DPO π_θ^pos and inverse DPO π_θ^inv on UltraFeedback (paper Eqs. 6–7).
# Init from the paper SFT checkpoints. Trainer: OpenRLHF (Appendix G).
# Usage: bash scripts/02_train_seed_dpo.sh qwen|llama [normal|flipped|both]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

MODEL="${1:?usage: $0 qwen|llama [normal|flipped|both]}"
VARIANT="${2:-both}"
KEY="$(model_tag "${MODEL}")"
SFT="$(sft_model_path "${MODEL}")"

run_one() {
  local name="$1" dataset="$2" out="$3"
  if [ ! -s "${dataset}" ]; then
    echo "Missing ${dataset}. Run scripts/01_prepare_data.sh first." >&2
    exit 1
  fi
  if [ "${SKIP_DONE}" = "1" ] && [ "${FORCE}" != "1" ] && weights_ok "${out}"; then
    log "SKIP ${name}: ${out}"
    return 0
  fi
  log "===== ${name}  GPUs=${GPUS}  sft=${SFT}  out=${out} ====="
  orhf_dpo "${SFT}" "${dataset}" "${out}"
  log "DONE ${name}"
}

case "${VARIANT}" in
  normal)
    run_one "${KEY} π_θ^pos DPO" "${UF_JSONL}" "$(saves_for "${MODEL}")/dpo"
    ;;
  flipped)
    run_one "${KEY} π_θ^inv DPO" "${UF_FLIP}" "$(saves_for "${MODEL}")/dpo_flipped"
    ;;
  both)
    run_one "${KEY} π_θ^pos DPO" "${UF_JSONL}" "$(saves_for "${MODEL}")/dpo"
    run_one "${KEY} π_θ^inv DPO" "${UF_FLIP}" "$(saves_for "${MODEL}")/dpo_flipped"
    ;;
  *)
    echo "Unknown variant: ${VARIANT} (normal|flipped|both)" >&2
    exit 1
    ;;
esac
