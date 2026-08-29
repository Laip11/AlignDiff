#!/usr/bin/env bash
# Step 1: UltraFeedback_Binarized (train_prefs) → prompt/chosen/rejected JSONL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

log "Preparing UltraFeedback_Binarized under ${DATA_DIR}"

if [ ! -s "${UF_JSONL}" ] || [ "${FORCE}" = "1" ]; then
  "${PYTHON}" -m aligndiff.convert ultrafeedback --output "${UF_JSONL}"
else
  log "skip existing ${UF_JSONL}"
fi

if [ ! -s "${UF_FLIP}" ] || [ "${FORCE}" = "1" ]; then
  "${PYTHON}" -m aligndiff.convert flip --input "${UF_JSONL}" --output "${UF_FLIP}"
else
  log "skip existing ${UF_FLIP}"
fi

log "Done. SFT init uses paper checkpoints (glorgao/Qwen2.5-7B-SFT, princeton-nlp/Llama-3-Base-8B-SFT)."
