<div align="center">

# AlignDiff

### Exploiting Model-Intrinsic Information for Better Preference Data Selection

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Findings-191970.svg)](https://2026.emnlp.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/code-Laip11%2FAlignDiff-black.svg)](https://github.com/Laip11/AlignDiff)

**Official implementation** of our EMNLP 2026 Findings paper.

[Overview](#overview) • [Installation](#installation) • [Reproduction](#reproduction) • [Results](#results) • [Citation](#citation)

</div>

---

> AlignDiff filters preference data using **only model-intrinsic signals**: Alignment Discrepancy (RAD) for polarity, then Average Negative Log-Likelihood Gap (ANG) for difficulty. DPO on the top-30k subset outperforms training on the full UltraFeedback set and seven strong filters.

## News

- **[2026.08]** AlignDiff is accepted to **EMNLP 2026 Findings** 🎉

## Overview

Existing preference filters either trust an external LLM-as-judge or keep only the easiest implicit-reward pairs. AlignDiff separates two orthogonal questions:

1. **Does this pair have a clear preference?** Stage 1 trains a positive DPO policy $\pi_\theta^{\mathrm{pos}}$ and an inverse DPO policy $\pi_\theta^{\mathrm{inv}}$, then keeps pairs with $|R_{\mathrm{AD}}| > \tau$ and **flips** labels when $R_{\mathrm{AD}} < -\tau$.
2. **Is the pair informative to learn from?** Stage 2 ranks the polarity-clear pairs by ANG under the SFT reference and keeps the **hard** top-$K$ (30k), which mitigates the DPO squeezing effect.

```
π_sft  (UltraChat SFT: LLaMA-3-8B / Qwen2.5-7B)
   │
   ├──── DPO on UltraFeedback ──────────► π_θ^pos
   └──── DPO on flipped UltraFeedback ──► π_θ^inv
                    │
                    ▼
         R_AD = M_im^pos − M_im^inv          (Eq. 19)
         keep |R_AD| > τ; flip if R_AD < −τ
         rank by ANG(π_sft), take top 30k    (Eq. 11)
                    │
                    ▼
         DPO on the AlignDiff subset ───────► π_AlignDiff
```

This repository implements the **method and training pipeline**. Benchmark evaluation (AlpacaEval 2.0, Arena-Hard, MT-Bench) follows the official protocols in the paper (Appendix F) and is not bundled here.

## Installation

```bash
git clone https://github.com/Laip11/AlignDiff.git
cd AlignDiff
conda create -n aligndiff python=3.10 -y
conda activate aligndiff
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build you need
bash scripts/setup_env.sh
cp configs/local.env.example configs/local.env   # set GPUs
```

`scripts/setup_env.sh` installs this package and the vendored [OpenRLHF](OpenRLHF/README.md) trainer (Appendix G). FlashAttention is optional: `FLASH_ATTN=1 bash scripts/setup_env.sh`.

Filter and margin scoring only need `pip install -e .`.

## Data and Models

We **do not** train SFT from scratch. DPO is initialized from the public UltraChat SFT checkpoints used in the paper:

| Role            | Checkpoint                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| LLaMA-3-8B-SFT  | [`princeton-nlp/Llama-3-Base-8B-SFT`](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT)                                    |
| Qwen2.5-7B-SFT  | [`glorgao/Qwen2.5-7B-SFT`](https://huggingface.co/glorgao/Qwen2.5-7B-SFT)                                                          |
| Preference data | [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (`train_prefs`) |

```bash
bash scripts/01_prepare_data.sh
# writes data/ultrafeedback_binarized.jsonl
#         data/ultrafeedback_binarized_flipped.jsonl
```

JSONL is OpenRLHF format: `prompt`, `chosen`, `rejected`.

## Reproduction

End-to-end (Qwen2.5-7B-SFT; swap `qwen` for `llama`):

```bash
bash scripts/run_pipeline.sh qwen
```

| Step | Script                                     | Description                                           |
| :--: | ------------------------------------------ | ----------------------------------------------------- |
|  1  | `scripts/01_prepare_data.sh`             | Convert UltraFeedback to prompt/chosen/rejected JSONL |
|  2  | `scripts/02_train_seed_dpo.sh qwen`      | OpenRLHF DPO:`π_θ^pos` and inverse `π_θ^inv`  |
|  3  | `scripts/03_compute_margins.sh qwen`     | IM (`pos` vs SFT) and RAD (`pos` vs `inv`)      |
|  4  | `scripts/04_filter_aligndiff.sh qwen`    | RAD + τ, then ANG top-30k                            |
|  5  | `scripts/05_train_aligndiff_dpo.sh qwen` | OpenRLHF DPO on the filtered subset                   |

Main paper setting: **τ = 20**, **K = 30k**. Ablate τ with:

```bash
TAUS=10,20,30 bash scripts/04_filter_aligndiff.sh qwen
```

Optional easy-to-hard curriculum (Table 2):

```bash
CURRICULUM=easy-to-hard bash scripts/04_filter_aligndiff.sh qwen
```

Ad-hoc DPO (same recipe, from `OpenRLHF/`):

```bash
PRETRAIN=glorgao/Qwen2.5-7B-SFT \
  DATASET=data/filtered/qwen/aligndiff_tau20_30k.jsonl \
  SAVE_PATH=saves/qwen2.5-7b/full/dpo_aligndiff_tau20_30k \
  bash OpenRLHF/train_dpo.sh
```

### Hyperparameters (Appendix G)

All DPO runs use OpenRLHF, global batch 128, AdamW, cosine schedule, 10% warmup, **1 epoch**.

|                   | Value    |
| ----------------- | -------- |
| Learning rate     | `5e-7` |
| DPO β            | `0.01` |
| Max length        | 2048     |
| τ (LLaMA / Qwen) | 20       |
| τ (Mistral)      | 80       |
| Top-K             | 30,000   |

Steps 1 and 4 already wrap convert / filter. Re-run the filter by hand only if you already have margin files:

```bash
python -m aligndiff.filter \
    --im-margins data/margins/qwen_im_margins.jsonl \
    --rad-margins data/margins/qwen_rad_margins.jsonl \
    --out-dir data/filtered/qwen \
    --model-key qwen --top-k 30000 --tau 20
```

Baseline selectors (`im`, `em`, `lcpp`, `pplgap`, `map`, `im_em`, `rip`) are available via `--methods` and are **not** required for AlignDiff. EM scoring uses Qwen2.5-72B-Instruct (Appendix H).

## Results

AlpacaEval 2.0 / Arena-Hard / MT-Bench from the paper (Table 1). AlignDiff uses 30k pairs vs. the full UltraFeedback set.

| Method              | LLaMA-3-8B-SFT LC |             WR |     Arena-Hard |      MT-Bench | Qwen2.5-7B-SFT LC |             WR |     Arena-Hard |      MT-Bench |
| ------------------- | ----------------: | -------------: | -------------: | ------------: | ----------------: | -------------: | -------------: | ------------: |
| Full data           |              13.7 |           16.8 |           36.2 |           6.5 |              21.3 |           18.9 |           46.9 |           6.8 |
| IM                  |              19.1 |           19.8 |           38.9 |           7.1 |              27.8 |           27.0 |           57.0 |           7.3 |
| SDPO                |              20.1 |           21.3 |           45.3 |           6.9 |              30.2 |           28.8 |           55.3 |           7.2 |
| **AlignDiff** |    **26.4** | **29.3** | **47.0** | **7.2** |    **33.4** | **33.8** | **58.7** | **7.3** |

On AlpacaEval 2.0, AlignDiff improves length-controlled win rate over SDPO by **+6.3** (LLaMA) and **+3.2** (Qwen). Training LLaMA-3-8B-SFT on the AlignDiff subset nearly **doubles** LC vs. the full set (13.7 → 26.4).

## Repository Structure

```
.
├── OpenRLHF/                 # DPO trainer (Appendix G) + launch scripts
├── configs/local.env.example
├── environment.yml           # conda: python 3.10
├── scripts/
│   ├── setup_env.sh          # pip install AlignDiff + OpenRLHF
│   ├── 01_prepare_data.sh
│   ├── 02_train_seed_dpo.sh
│   ├── 03_compute_margins.sh
│   ├── 04_filter_aligndiff.sh
│   ├── 05_train_aligndiff_dpo.sh
│   └── run_pipeline.sh
└── src/aligndiff/            # convert, margin scoring, RAD/ANG filter
```

## Acknowledgements

Our training recipe follows [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Experiments use [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) and the SFT checkpoints released by [SimPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT) and [SelectiveDPO](https://huggingface.co/glorgao/Qwen2.5-7B-SFT).

## Citation

If you find this repository or the paper useful, please cite:

```bibtex
@inproceedings{lai2026aligndiff,
  title     = {{AlignDiff}: Exploiting Model-Intrinsic Information for Better Preference Data Selection},
  author    = {Lai, Peng and Zhu, He and Ruan, Zhiwen and Zhang, Dongdong and Chen, Yun and Li, Peng and Wei, Furu and Liu, Yang and Chen, Guanhua},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## Bugs or Questions?

Please open a [GitHub issue](https://github.com/Laip11/AlignDiff/issues). If you have questions about the paper, contact the corresponding author listed in the manuscript.
