#!/usr/bin/env python3
"""Build AlignDiff (and optional baseline) preference subsets.

Paper Algorithm (Fig. 8):
  Stage 1: train π_θ^pos / π_θ^inv, keep |RAD| > τ, flip if RAD < -τ.
  Stage 2: rank remaining pairs by ANG(π_ref=SFT) descending, take top-K (30k).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aligndiff.features import (
    add_ref_logprob_features,
    get_len_columns,
    merge_em_scores,
    merge_im_and_rad,
    select_aligndiff,
    select_aligndiff_stage2,
    select_em,
    select_im,
    select_im_and_em,
    select_lcpp,
    select_map,
    select_pplgap,
    select_rip,
    tau_for_model,
    tokenizer_for_model,
    write_outputs,
)

DEFAULT_METHODS = ("aligndiff",)
ALL_METHODS = ("aligndiff", "im", "em", "lcpp", "pplgap", "map", "im_em", "rip")
EM_METHODS = ("em", "map", "im_em", "rip")
ALIGNDIFF_ALIASES = {"aligndiff"}


def parse_methods(raw: str) -> list[str]:
    key = raw.strip().lower()
    if key in {"default", "aligndiff"}:
        return ["aligndiff"]
    if key == "all":
        return ["aligndiff", "im", "em", "lcpp", "pplgap", "map", "im_em", "rip"]
    chosen = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in chosen if m not in ALL_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Valid: {ALL_METHODS}")
    return chosen


def apply_curriculum(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Table 2: easy-to-hard (small ANG first) beats random / hard-to-easy."""
    if mode in {"none", "", "random"}:
        return df
    if "ang" not in df.columns:
        raise KeyError("curriculum ordering needs an 'ang' column")
    if mode == "easy-to-hard":
        return df.sort_values("ang", ascending=True)
    if mode == "hard-to-easy":
        return df.sort_values("ang", ascending=False)
    raise ValueError(f"Unknown curriculum: {mode}")


def prepare_base_df(
    im_margins: Path,
    rad_margins: Path | None,
    em_scores: Path | None,
    model_key: str,
    tokenizer: str | None,
    need_rad: bool,
) -> pd.DataFrame:
    im_df = pd.read_json(im_margins, lines=True).dropna(subset=["prompt", "chosen", "rejected"])
    print(f"Loaded IM margins: {len(im_df)} rows from {im_margins}")

    if need_rad and rad_margins and rad_margins.is_file():
        rad_df = pd.read_json(rad_margins, lines=True).dropna(subset=["prompt", "chosen", "rejected"])
        print(f"Loaded RAD margins: {len(rad_df)} rows from {rad_margins}")
        df = merge_im_and_rad(im_df, rad_df)
    else:
        if need_rad:
            raise FileNotFoundError(f"RAD margins required for AlignDiff stage-1: {rad_margins}")
        df = im_df

    if em_scores and em_scores.is_file():
        em_df = pd.read_json(em_scores, lines=True).dropna(subset=["prompt", "chosen", "rejected"])
        print(f"Loaded EM scores: {len(em_df)} rows from {em_scores}")
        df = merge_em_scores(df, em_df)

    tok = tokenizer_for_model(model_key, tokenizer)
    df = get_len_columns(df, tok)
    df = add_ref_logprob_features(df)
    print(f"After feature prep: {len(df)} rows")
    return df


def build_subsets(
    df,
    methods: list[str],
    out_dir: Path,
    top_k: int,
    model_key: str,
    ang_descending: bool,
    stage1: bool,
    tau_override: float | None,
    curriculum: str,
) -> None:
    top_k = min(top_k, len(df))
    tau = tau_override if tau_override is not None else tau_for_model(model_key)
    print(
        f"Building subsets top_k={top_k}, methods={methods}, "
        f"stage1={stage1}, tau={tau}, curriculum={curriculum}"
    )

    need_em = any(m in EM_METHODS for m in methods)
    if need_em and "em_margin" not in df.columns and "em" not in df.columns:
        raise RuntimeError("EM methods requested but em/em_margin missing.")

    if stage1 and any(m in ALIGNDIFF_ALIASES for m in methods):
        if "rad" not in df.columns and "margin" not in df.columns:
            raise RuntimeError("AlignDiff stage-1 needs RAD = M_im^pos - M_im^inv.")

    def aligndiff_selector():
        if stage1:
            return select_aligndiff(
                df,
                top_k,
                tau=tau,
                rad_col="rad" if "rad" in df.columns else "margin",
                ang_descending=ang_descending,
            )
        return select_aligndiff_stage2(df, top_k, ang_descending=ang_descending)

    selectors = {
        "im": lambda: select_im(df, top_k),
        "em": lambda: select_em(df, top_k),
        "lcpp": lambda: select_lcpp(df, top_k),
        "pplgap": lambda: select_pplgap(df, top_k, use_ppl=True),
        "map": lambda: select_map(df, top_k),
        "im_em": lambda: select_im_and_em(df, top_k),
        "rip": lambda: select_rip(df, top_k),
        "aligndiff": aligndiff_selector,
    }

    for method in methods:
        subset = selectors[method]()
        subset = apply_curriculum(subset, curriculum)
        if method in ALIGNDIFF_ALIASES and stage1:
            tau_tag = str(int(tau)) if float(tau).is_integer() else str(tau).replace(".", "p")
            prefix = f"aligndiff_tau{tau_tag}"
        elif method in ALIGNDIFF_ALIASES:
            prefix = "aligndiff"
        else:
            prefix = method
        write_outputs(subset, out_dir, prefix, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter preference data with AlignDiff.")
    parser.add_argument("--im-margins", type=Path, required=True, help="DPO vs SFT (IM + SFT logprobs)")
    parser.add_argument("--rad-margins", type=Path, default=None, help="DPO vs inverse-DPO (RAD)")
    parser.add_argument("--em-scores", type=Path, default=None, help="Optional LLM-judge scores")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30000, help="Paper: 30k, about the top half of UltraFeedback")
    parser.add_argument("--model-key", choices=["qwen", "llama", "mistral"], required=True)
    parser.add_argument("--methods", type=str, default="aligndiff")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--stage1",
        dest="stage1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RAD + τ keep/flip before ANG ranking (paper default: on).",
    )
    parser.add_argument("--tau", dest="tau", type=float, default=None)
    parser.add_argument(
        "--ang-descending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage-2: largest ANG first (paper Eq. 11).",
    )
    parser.add_argument(
        "--curriculum",
        choices=["none", "random", "easy-to-hard", "hard-to-easy"],
        default="none",
        help="Optional DPO example order after filtering (Table 2: easy-to-hard).",
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    need_rad = args.stage1 and any(m in ALIGNDIFF_ALIASES for m in methods)
    df = prepare_base_df(
        args.im_margins,
        args.rad_margins,
        args.em_scores,
        args.model_key,
        args.tokenizer,
        need_rad=need_rad,
    )
    build_subsets(
        df,
        methods,
        args.out_dir,
        args.top_k,
        args.model_key,
        args.ang_descending,
        args.stage1,
        args.tau,
        args.curriculum,
    )


if __name__ == "__main__":
    main()
