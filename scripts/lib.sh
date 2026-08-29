#!/usr/bin/env bash
# Shared env for AlignDiff pipeline scripts. Source from other scripts, do not run.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ROOT}/configs/local.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/configs/local.env"
  set +a
fi

ALIGNDIFF_ROOT="${ALIGNDIFF_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ALIGNDIFF_ROOT}/OpenRLHF:${ALIGNDIFF_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Paper SFT checkpoints (UltraChat), not base / Instruct models.
MODEL_QWEN="${MODEL_QWEN:-glorgao/Qwen2.5-7B-SFT}"
MODEL_LLAMA="${MODEL_LLAMA:-princeton-nlp/Llama-3-Base-8B-SFT}"
GPUS="${GPUS:-0,1,2,3}"
TOP_K="${TOP_K:-30000}"
TAUS="${TAUS:-20}"
FORCE="${FORCE:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

# Appendix G DPO (OpenRLHF)
LR="${LR:-5e-7}"
BETA="${BETA:-0.01}"
EPOCHS="${EPOCHS:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-2}"
MAX_LEN="${MAX_LEN:-2048}"
ZERO_STAGE="${ZERO_STAGE:-3}"
if [ "${FLASH_ATTN:-0}" = "1" ]; then
  ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
else
  ATTN_IMPL="${ATTN_IMPL:-sdpa}"
fi

DATA_DIR="${DATA_DIR:-${ALIGNDIFF_ROOT}/data}"
SAVES_DIR="${SAVES_DIR:-${ALIGNDIFF_ROOT}/saves}"
LOG_DIR="${LOG_DIR:-${ALIGNDIFF_ROOT}/logs}"

UF_JSONL="${DATA_DIR}/ultrafeedback_binarized.jsonl"
UF_FLIP="${DATA_DIR}/ultrafeedback_binarized_flipped.jsonl"

mkdir -p "${DATA_DIR}" "${SAVES_DIR}" "${LOG_DIR}" "${DATA_DIR}/filtered" "${DATA_DIR}/margins"

log() { echo "[$(date '+%F %T')] $*"; }

n_gpus() { echo "${GPUS}" | awk -F, '{print NF}'; }

require_orhf() {
  if ! command -v deepspeed >/dev/null 2>&1 || ! "${PYTHON}" -c "import openrlhf" >/dev/null 2>&1; then
    echo "OpenRLHF is not installed. From the repo root run: bash scripts/setup_env.sh" >&2
    exit 1
  fi
}

model_tag() {
  case "$1" in
    qwen|qwen2.5|qwen2.5-7b) echo qwen ;;
    llama|llama3|llama3-8b) echo llama ;;
    mistral) echo mistral ;;
    *) echo "Unknown model: $1 (expected qwen|llama)" >&2; return 1 ;;
  esac
}

model_name_dir() {
  case "$(model_tag "$1")" in
    qwen) echo qwen2.5-7b ;;
    llama) echo llama3-8b ;;
    mistral) echo mistral-7b ;;
  esac
}

sft_model_path() {
  case "$(model_tag "$1")" in
    qwen) echo "${MODEL_QWEN}" ;;
    llama) echo "${MODEL_LLAMA}" ;;
  esac
}

saves_for() {
  echo "${SAVES_DIR}/$(model_name_dir "$1")/full"
}

weights_ok() {
  local dir="$1"
  [ -f "${dir}/config.json" ] || return 1
  [ -f "${dir}/model.safetensors" ] || [ -f "${dir}/model.safetensors.index.json" ] || [ -f "${dir}/pytorch_model.bin" ]
}

# OpenRLHF DPO (Appendix G). Dataset is prompt/chosen/rejected JSONL.
orhf_dpo() {
  local pretrain="$1" dataset="$2" save_path="$3"
  require_orhf
  mkdir -p "${save_path}"
  log "OpenRLHF DPO  pretrain=${pretrain}"
  log "  dataset=${dataset}"
  log "  save=${save_path}  lr=${LR}  beta=${BETA}  attn=${ATTN_IMPL}  gpus=${GPUS}"
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  export TOKENIZERS_PARALLELISM=false
  deepspeed --module openrlhf.cli.train_dpo \
    --save_path "${save_path}" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --pretrain "${pretrain}" \
    --dataset "${dataset}" \
    --dataset_split train \
    --apply_chat_template \
    --prompt_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --max_len "${MAX_LEN}" \
    --train_batch_size "${GLOBAL_BATCH}" \
    --micro_train_batch_size "${MICRO_BATCH}" \
    --max_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --lr_warmup_ratio 0.1 \
    --beta "${BETA}" \
    --bf16 \
    --zero_stage "${ZERO_STAGE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --gradient_checkpointing \
    --load_checkpoint \
    --seed 42
}
