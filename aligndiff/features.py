"""Shared helpers for AlignDiff / baseline preference-data filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL_TOKENIZER = {
    "qwen": "glorgao/Qwen2.5-7B-SFT",
    "llama": "princeton-nlp/Llama-3-Base-8B-SFT",
    "mistral": "HuggingFaceH4/mistral-7b-sft-beta",
}

MODEL_TAU = {
    "qwen": 20.0,
    "llama": 20.0,
    "mistral": 80.0,
}

EM_ASPECTS = ("helpfulness", "honesty", "instruction_following", "truthfulness")
IM_EM_M1 = -2.0
IM_EM_M2 = 30.0
RIP_EM_THRESHOLD = 0.126


def tokenizer_for_model(model_key: str, override: str | None = None) -> str:
    if override:
        return override
    return MODEL_TOKENIZER.get(model_key, MODEL_TOKENIZER["qwen"])


def tau_for_model(model_key: str) -> float:
    return MODEL_TAU.get(model_key, 20.0)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    sample = out.iloc[0]["chosen"]
    if not isinstance(sample, str):
        out["chosen"] = out["chosen"].apply(lambda x: x if isinstance(x, str) else str(x))
        out["rejected"] = out["rejected"].apply(lambda x: x if isinstance(x, str) else str(x))
    return out


def get_len_columns(df: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    out = normalize_text_columns(df)
    out["chosen_len"] = out["chosen"].apply(lambda x: len(tokenizer.encode(x)))
    out["rejected_len"] = out["rejected"].apply(lambda x: len(tokenizer.encode(x)))
    out["len_gap"] = out["chosen_len"] - out["rejected_len"]
    return out


def add_ref_logprob_features(df: pd.DataFrame) -> pd.DataFrame:
    """ANG / PPL features from SFT reference logprobs (π_ref)."""
    out = df.copy()
    out["original_chosen_logprobs"] = out["ref_chosen_logprob"]
    out["original_rejected_logprobs"] = out["ref_rejected_logprob"]
    avg_chosen = out["ref_chosen_logprob"] / out["chosen_len"]
    avg_rejected = out["ref_rejected_logprob"] / out["rejected_len"]
    out["original_avg_chosen_logprobs"] = avg_chosen
    out["original_avg_rejected_logprobs"] = avg_rejected
    out["original_chosen_ppl"] = np.exp(-avg_chosen)
    out["original_rejected_ppl"] = np.exp(-avg_rejected)
    nll_chosen = -avg_chosen
    nll_rejected = -avg_rejected
    # Paper Eq. 11: ANG = AvgNLL(y_w) - AvgNLL(y_l). Larger = harder chosen / easier rejected.
    out["ang"] = nll_chosen - nll_rejected
    out["original_avg_logprobs_gap"] = avg_chosen - avg_rejected
    out["original_ppl_gap"] = out["original_chosen_ppl"] - out["original_rejected_ppl"]
    return out[(out["chosen_len"] > 0) & (out["rejected_len"] > 0)]


def merge_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["prompt"].astype(str)
        + "\x00"
        + df["chosen"].astype(str)
        + "\x00"
        + df["rejected"].astype(str)
    )


def merge_im_and_rad(im_df: pd.DataFrame, rad_df: pd.DataFrame) -> pd.DataFrame:
    """Join IM (DPO vs SFT) with RAD (DPO vs inverse DPO).

    IM file keeps SFT logprobs for ANG. RAD file contributes Eq. 19:
    RAD = M_im^pos - M_im^inv.
    """
    im = im_df.copy()
    rad = rad_df.copy()
    if "im" not in im.columns and "margin" in im.columns:
        im["im"] = im["margin"]
    if "rad" not in rad.columns:
        if "margin" not in rad.columns:
            raise KeyError("RAD file needs a 'rad' or 'margin' column (M_im^pos - M_im^inv).")
        rad["rad"] = rad["margin"]
    im["_key"] = merge_key(im)
    rad["_key"] = merge_key(rad)
    keep = ["_key", "rad"]
    if "ref_chosen_logprob" in rad.columns:
        rad = rad.rename(
            columns={
                "ref_chosen_logprob": "inv_chosen_logprob",
                "ref_rejected_logprob": "inv_rejected_logprob",
            }
        )
        keep.extend(["inv_chosen_logprob", "inv_rejected_logprob"])
    merged = im.merge(rad[keep], on="_key", how="left")
    if merged["rad"].isna().any():
        print(f"WARN: {int(merged['rad'].isna().sum())} rows missing RAD after merge")
    return merged.drop(columns=["_key"])


def merge_em_scores(df: pd.DataFrame, em_df: pd.DataFrame) -> pd.DataFrame:
    em = em_df.copy()
    base = df.copy()
    base["_key"] = merge_key(base)
    em["_key"] = merge_key(em)
    em_cols = [c for c in em.columns if c not in {"prompt", "chosen", "rejected", "_key"}]
    merged = base.merge(em[["_key"] + em_cols], on="_key", how="left")
    if "em_margin" not in merged.columns and "em" in merged.columns:
        merged["em_margin"] = merged["em"]
    return merged.drop(columns=["_key"])


def ensure_im_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "im" not in out.columns and "margin" in out.columns:
        out["im"] = out["margin"]
    return out


def select_im(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    col = "im" if "im" in df.columns else "margin"
    return df.nlargest(top_k, col)


def select_em(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    col = "em_margin" if "em_margin" in df.columns else "em"
    return df.nlargest(top_k, col)


def select_lcpp(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df.nlargest(top_k, "chosen_len")


def select_pplgap(df: pd.DataFrame, top_k: int, use_ppl: bool = True) -> pd.DataFrame:
    col = "original_ppl_gap" if use_ppl else "original_avg_logprobs_gap"
    return df.nlargest(top_k, col)


def normalize_margin(m: np.ndarray, m1: float, m2: float) -> np.ndarray:
    clipped = np.clip(m, m1, m2)
    return (clipped - m1) / (m2 - m1)


def select_im_and_em(
    df: pd.DataFrame,
    top_k: int,
    m1: float = IM_EM_M1,
    m2: float = IM_EM_M2,
) -> pd.DataFrame:
    work = ensure_im_column(df)
    em_col = "em_margin" if "em_margin" in work.columns else "em"
    p_ex = normalize_margin(work[em_col].values, m1, m2)
    p_im = normalize_margin(work["im"].values, m1, m2)
    numerator = p_ex * p_im
    denominator = numerator + (1 - p_ex) * (1 - p_im)
    work = work.copy()
    work["joint_prob"] = numerator / denominator
    return work.nlargest(top_k, "joint_prob")


def select_map(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    work = ensure_im_column(df)
    em_col = "em_margin" if "em_margin" in work.columns else "em"
    work = work.copy()
    work["map"] = np.abs(work[em_col]) - np.abs(work["im"])
    return work.nlargest(top_k, "map")


def select_rip(df: pd.DataFrame, top_k: int, threshold: float = RIP_EM_THRESHOLD) -> pd.DataFrame:
    work = ensure_im_column(df)
    em_col = "em_margin" if "em_margin" in work.columns else "em"
    filtered = work[work[em_col] > threshold]
    return filtered.nlargest(min(top_k, len(filtered)), "rejected_len")


def select_aligndiff_stage2(
    df: pd.DataFrame,
    top_k: int,
    ang_col: str = "ang",
    ang_descending: bool = True,
) -> pd.DataFrame:
    """AlignDiff stage-2 only: rank by ANG (π_ref = SFT), no RAD / τ filtering."""
    work = df.copy()
    if ang_col not in work.columns:
        raise KeyError(f"ANG column missing: expected '{ang_col}'")
    work = work.sort_values(by=ang_col, ascending=not ang_descending)
    return work.head(top_k)


def flip_preference_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Swap chosen/rejected (and related features) for reverse-preference pairs."""
    out = df.copy()
    out["chosen"], out["rejected"] = out["rejected"], out["chosen"]
    swap_pairs = (
        ("chosen_len", "rejected_len"),
        ("ref_chosen_logprob", "ref_rejected_logprob"),
        ("original_chosen_logprobs", "original_rejected_logprobs"),
        ("original_avg_chosen_logprobs", "original_avg_rejected_logprobs"),
        ("original_chosen_ppl", "original_rejected_ppl"),
        ("policy_chosen_logprob", "policy_rejected_logprob"),
    )
    for a, b in swap_pairs:
        if a in out.columns and b in out.columns:
            out[a], out[b] = out[b].copy(), out[a].copy()
    for col in ("ang", "original_avg_logprobs_gap", "original_ppl_gap", "len_gap"):
        if col in out.columns:
            out[col] = -out[col]
    return out


def select_aligndiff(
    df: pd.DataFrame,
    top_k: int,
    tau: float,
    rad_col: str = "rad",
    ang_col: str = "ang",
    ang_descending: bool = True,
) -> pd.DataFrame:
    """AlignDiff: stage-1 RAD + τ (keep / flip), then stage-2 top-k by ANG."""
    work = df.copy()
    if rad_col not in work.columns:
        if "margin" in work.columns:
            rad_col = "margin"
        else:
            raise KeyError(f"RAD column missing: expected '{rad_col}' or 'margin'")

    signal = work[rad_col]
    correct = work[signal > tau].copy()
    reverse = flip_preference_pair(work[signal < -tau])
    combined = pd.concat([correct, reverse], ignore_index=True)
    return select_aligndiff_stage2(combined, top_k, ang_col=ang_col, ang_descending=ang_descending)


def subset_suffix(top_k: int) -> str:
    if top_k % 1000 == 0:
        return f"{top_k // 1000}k"
    return str(top_k)


def write_outputs(df: pd.DataFrame, out_dir: Path, prefix: str, top_k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = subset_suffix(top_k)
    path = out_dir / f"{prefix}_{suffix}.jsonl"
    slim = df[["prompt", "chosen", "rejected"]].copy()
    slim.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"  {prefix}: {len(slim)} -> {path}")
