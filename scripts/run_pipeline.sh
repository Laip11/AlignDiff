#!/usr/bin/env bash
# End-to-end AlignDiff for one model family (paper pipeline, no local SFT).
# Usage: bash scripts/run_pipeline.sh qwen|llama
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

MODEL="${1:?usage: $0 qwen|llama}"
log "===== AlignDiff pipeline: ${MODEL}  GPUs=${GPUS}  SFT=$(sft_model_path "${MODEL}") ====="

bash "${SCRIPT_DIR}/01_prepare_data.sh"
bash "${SCRIPT_DIR}/02_train_seed_dpo.sh" "${MODEL}" both
bash "${SCRIPT_DIR}/03_compute_margins.sh" "${MODEL}"
bash "${SCRIPT_DIR}/04_filter_aligndiff.sh" "${MODEL}"
bash "${SCRIPT_DIR}/05_train_aligndiff_dpo.sh" "${MODEL}" 20

log "===== Pipeline complete: ${MODEL} ====="
log "AlignDiff DPO checkpoint: $(saves_for "${MODEL}")/dpo_aligndiff_tau20_*"
