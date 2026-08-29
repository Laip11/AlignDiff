# OpenRLHF (training backend)

This directory is the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) snapshot used for AlignDiff DPO (paper Appendix G). Install it from here rather than PyPI so the trainer matches this repo.

```bash
# from the repo root, after PyTorch is installed
bash scripts/setup_env.sh
```

Or by hand:

```bash
pip install -e .                 # AlignDiff (data + filter)
pip install -e ./OpenRLHF        # DPO trainer (pulls DeepSpeed, etc.)
```

`flash-attn` is optional. The numbered scripts default to `--attn_implementation sdpa`. To use FlashAttention:

```bash
FLASH_ATTN=1 bash scripts/setup_env.sh
ATTN_IMPL=flash_attention_2 bash scripts/02_train_seed_dpo.sh qwen
```

Run DPO with paper hyperparameters via `train_dpo.sh` (same recipe as `scripts/02` / `scripts/05`):

```bash
PRETRAIN=glorgao/Qwen2.5-7B-SFT \
  DATASET=../data/ultrafeedback_binarized.jsonl \
  SAVE_PATH=../saves/qwen2.5-7b/full/dpo \
  bash train_dpo.sh
```

Upstream license: Apache-2.0 (`LICENSE`).
