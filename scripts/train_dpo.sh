#!/usr/bin/env bash
# Ad-hoc OpenRLHF DPO (same hyperparameters as scripts/02 and scripts/05).
#
#   PRETRAIN=glorgao/Qwen2.5-7B-SFT \
#     DATASET=data/ultrafeedback_binarized.jsonl \
#     SAVE_PATH=saves/qwen2.5-7b/full/dpo \
#     bash scripts/train_dpo.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

PRETRAIN="${PRETRAIN:?set PRETRAIN to the SFT checkpoint}"
DATASET="${DATASET:?set DATASET to a prompt/chosen/rejected JSONL}"
SAVE_PATH="${SAVE_PATH:-saves/dpo}"

orhf_dpo "${PRETRAIN}" "${DATASET}" "${SAVE_PATH}"
