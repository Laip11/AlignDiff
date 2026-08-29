#!/usr/bin/env bash
# Install AlignDiff + the vendored OpenRLHF trainer.
#
#   # 1. create env (once)
#   conda create -n aligndiff python=3.10 -y
#   conda activate aligndiff
#   pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA
#
#   # 2. install this repo
#   bash scripts/setup_env.sh
#
# Optional: FLASH_ATTN=1 bash scripts/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
FLASH_ATTN="${FLASH_ATTN:-0}"

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  echo "Install PyTorch for your CUDA first, then re-run." >&2
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124" >&2
  exit 1
fi

echo "==> AlignDiff (convert / filter / margin scoring)"
"${PYTHON}" -m pip install -e "${ROOT}"

REQ="${ROOT}/OpenRLHF/requirements.txt"
TMP="$(mktemp)"
grep -vE '^[[:space:]]*flash-attn' "${REQ}" > "${TMP}"
echo "==> OpenRLHF dependencies (flash-attn skipped; default attn is sdpa)"
"${PYTHON}" -m pip install -r "${TMP}"
rm -f "${TMP}"

echo "==> OpenRLHF from ${ROOT}/OpenRLHF"
"${PYTHON}" -m pip install -e "${ROOT}/OpenRLHF" --no-deps

if [ "${FLASH_ATTN}" = "1" ]; then
  echo "==> flash-attn (optional)"
  "${PYTHON}" -m pip install flash-attn==2.8.3 --no-build-isolation
fi

"${PYTHON}" -c "import openrlhf, deepspeed, aligndiff; print('ok: openrlhf + deepspeed + aligndiff')"
echo "Done. Copy configs/local.env.example to configs/local.env and run:"
echo "  bash scripts/run_pipeline.sh qwen"
