#!/usr/bin/env bash
# Run OpenRLHF DPO with AlignDiff paper hyperparameters (Appendix G).
# Same trainer as scripts/02_train_seed_dpo.sh and scripts/05_train_aligndiff_dpo.sh.
#
#   PRETRAIN=glorgao/Qwen2.5-7B-SFT \
#     DATASET=data/ultrafeedback_binarized.jsonl \
#     SAVE_PATH=saves/qwen2.5-7b/full/dpo \
#     bash OpenRLHF/train_dpo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib.sh"

PRETRAIN="${PRETRAIN:?set PRETRAIN to the SFT checkpoint}"
DATASET="${DATASET:?set DATASET to a prompt/chosen/rejected JSONL}"
SAVE_PATH="${SAVE_PATH:-saves/dpo}"

orhf_dpo "${PRETRAIN}" "${DATASET}" "${SAVE_PATH}"
