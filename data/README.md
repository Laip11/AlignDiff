# Data layout (AlignDiff / UltraFeedback)

Generated files are gitignored. Preference JSONL is OpenRLHF format: `prompt`, `chosen`, `rejected`.

```
data/
  ultrafeedback_binarized.jsonl              # seed DPO + margin scoring
  ultrafeedback_binarized_flipped.jsonl      # inverse DPO
  margins/
    qwen_im_margins.jsonl                    # π_θ^pos vs π_sft
    qwen_rad_margins.jsonl                   # π_θ^pos vs π_θ^inv  (Eq. 19)
  filtered/
    qwen/aligndiff_tau20_30k.jsonl           # AlignDiff subset
```

SFT initialization uses the public checkpoints in the paper, not a dataset in this folder:

- `princeton-nlp/Llama-3-Base-8B-SFT`
- `glorgao/Qwen2.5-7B-SFT`
