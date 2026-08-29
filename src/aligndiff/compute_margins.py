#!/usr/bin/env python3
"""Compute preference margins with multi-GPU data-parallel sharding.

IM  (policy=DPO, ref=SFT):
    margin = (log π_θ(y_w) - log π_θ(y_l)) - (log π_sft(y_w) - log π_sft(y_l))

RAD (policy=π_θ^pos, ref=π_θ^inv, --save-rad):
    rad = margin = (log π_pos(y_w) - log π_pos(y_l))
                 - (log π_inv(y_w) - log π_inv(y_l))
    which equals M_im^pos - M_im^inv after π_ref cancels (paper Eq. 19).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_scoring_tokenizer(primary: str, fallback: str | None = None) -> AutoTokenizer:
    """Load tokenizer for chat-template tokenization (policy first, ref as fallback)."""
    tokenizer = AutoTokenizer.from_pretrained(primary, trust_remote_code=True)
    template_source = primary
    if getattr(tokenizer, "chat_template", None) is None and fallback:
        fb = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        if getattr(fb, "chat_template", None) is not None:
            tokenizer.chat_template = fb.chat_template
            template_source = fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"No chat_template for margin scoring. policy={primary}, ref={fallback}. "
            "Use an instruct/SFT checkpoint with chat_template for --policy-model."
        )
    tokenizer.template_source = template_source  # type: ignore[attr-defined]
    return tokenizer


def preflight_margin_scoring(policy_model: str, ref_model: str) -> None:
    """Fail fast before spawning GPU workers if chat templates or paths are invalid."""
    for label, path in [("policy", policy_model), ("ref", ref_model)]:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Preflight failed: {label} model path not found: {path}")

    tokenizer = load_scoring_tokenizer(policy_model, ref_model)
    sample = {"prompt": "Hello", "chosen": "Hi there.", "rejected": "Not helpful."}
    formatted = apply_chat_template(sample, tokenizer)
    if not formatted["formatted_prompt"] or not formatted["chosen_response_only"]:
        raise RuntimeError("Preflight failed: chat template produced empty prompt/response spans.")

    print(
        f"Preflight OK | policy={policy_model} ref={ref_model} "
        f"tokenizer_template_from={tokenizer.template_source}"
    )


def apply_chat_template(example: dict, tokenizer: AutoTokenizer) -> dict:
    prompt_messages = [{"role": "user", "content": example["prompt"]}]
    chosen_messages = prompt_messages + [{"role": "assistant", "content": example["chosen"]}]
    rejected_messages = prompt_messages + [{"role": "assistant", "content": example["rejected"]}]

    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    full_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    prompt_len_char = len(formatted_prompt)
    return {
        "formatted_prompt": formatted_prompt,
        "chosen_response_only": full_chosen[prompt_len_char:],
        "rejected_response_only": full_rejected[prompt_len_char:],
    }


def pad_ids(input_ids: list[int], max_length: int, pad_id: int) -> tuple[list[int], list[int]]:
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        return input_ids, [1] * max_length
    pad_len = max_length - len(input_ids)
    return input_ids + [pad_id] * pad_len, [1] * len(input_ids) + [0] * pad_len


def tokenize_row(row: dict, tokenizer: AutoTokenizer, max_length: int, pad_id: int) -> dict:
    formatted = apply_chat_template(row, tokenizer)
    prompt_tokens = tokenizer(formatted["formatted_prompt"], add_special_tokens=False)
    chosen_tokens = tokenizer(
        formatted["chosen_response_only"] + tokenizer.eos_token, add_special_tokens=False
    )
    rejected_tokens = tokenizer(
        formatted["rejected_response_only"] + tokenizer.eos_token, add_special_tokens=False
    )
    p_len = len(prompt_tokens["input_ids"])
    c_len = len(chosen_tokens["input_ids"])
    r_len = len(rejected_tokens["input_ids"])
    c_ids, c_mask = pad_ids(prompt_tokens["input_ids"] + chosen_tokens["input_ids"], max_length, pad_id)
    r_ids, r_mask = pad_ids(prompt_tokens["input_ids"] + rejected_tokens["input_ids"], max_length, pad_id)
    return {
        "prompt_len": p_len,
        "chosen_len": c_len,
        "rejected_len": r_len,
        "chosen_ids": c_ids,
        "chosen_mask": c_mask,
        "rejected_ids": r_ids,
        "rejected_mask": r_mask,
    }


def pre_tokenize(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, workers: int) -> list[dict]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = [row.to_dict() for _, row in df.iterrows()]

    def _one(row: dict) -> dict:
        return tokenize_row(row, tokenizer, max_length, pad_id)

    if workers <= 1:
        return [_one(r) for r in rows]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, rows))


@torch.no_grad()
def response_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: List[int],
    response_lengths: List[int],
    pad_id: int,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    token_nll = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)
    token_logp = -token_nll

    totals = []
    for i in range(input_ids.size(0)):
        p_len = prompt_lengths[i]
        r_len = response_lengths[i]
        start = max(p_len - 1, 0)
        end = min(p_len + r_len - 1, token_logp.size(1))
        if end <= start:
            totals.append(0.0)
            continue
        vals = token_logp[i, start:end]
        labels = shift_labels[i, start:end]
        mask = (labels != pad_id).float()
        totals.append((vals * mask).sum().item())
    return torch.tensor(totals)


def load_model(model_path: str, dtype: torch.dtype, device: torch.device) -> AutoModelForCausalLM:
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn,
    )
    model.to(device)
    return model.eval()


def score_tokenized(
    model: AutoModelForCausalLM,
    device: torch.device,
    tokenized: list[dict],
    batch_size: int,
    pad_id: int,
    micro_batch_size: int = 8,
) -> tuple[list[float], list[float]]:
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []

    for start in tqdm(range(0, len(tokenized), batch_size), desc=f"scoring@{device.index}", leave=False):
        batch = tokenized[start : start + batch_size]
        batch_ids: list[list[int]] = []
        batch_mask: list[list[int]] = []
        prompt_lengths: list[int] = []
        chosen_lens: list[int] = []
        rejected_lens: list[int] = []

        for item in batch:
            batch_ids.extend([item["chosen_ids"], item["rejected_ids"]])
            batch_mask.extend([item["chosen_mask"], item["rejected_mask"]])
            prompt_lengths.extend([item["prompt_len"], item["prompt_len"]])
            chosen_lens.append(item["chosen_len"])
            rejected_lens.append(item["rejected_len"])

        resp_lens = []
        for i in range(len(chosen_lens)):
            resp_lens.extend([chosen_lens[i], rejected_lens[i]])

        all_logps: list[float] = []
        for mb_start in range(0, len(batch_ids), micro_batch_size):
            mb_end = mb_start + micro_batch_size
            input_ids = torch.tensor(batch_ids[mb_start:mb_end], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_mask[mb_start:mb_end], dtype=torch.long, device=device)
            mb_prompt_lengths = prompt_lengths[mb_start:mb_end]
            mb_resp_lens = resp_lens[mb_start:mb_end]
            logps = response_logprobs(
                model, input_ids, attention_mask, mb_prompt_lengths, mb_resp_lens, pad_id
            )
            all_logps.extend(logps.tolist())

        for i in range(len(chosen_lens)):
            chosen_scores.append(all_logps[i * 2])
            rejected_scores.append(all_logps[i * 2 + 1])

    return chosen_scores, rejected_scores


def shard_df(df: pd.DataFrame, shard_id: int, num_shards: int) -> pd.DataFrame:
    return df.iloc[shard_id : len(df) : num_shards].copy()


def score_model_shard(
    model_path: str,
    df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    token_workers: int,
    gpu_id: int,
    tokenized: list[dict] | None = None,
    tokenizer: AutoTokenizer | None = None,
    pad_id: int | None = None,
    micro_batch_size: int = 8,
) -> tuple[list[float], list[float]]:
    device = torch.device(f"cuda:{gpu_id}")
    if tokenized is None:
        tokenizer = load_scoring_tokenizer(model_path)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        print(f"[gpu {gpu_id}] tokenizing {len(df)} rows (workers={token_workers})...")
        tokenized = pre_tokenize(df, tokenizer, max_length, token_workers)

    assert tokenizer is not None and pad_id is not None
    print(f"[gpu {gpu_id}] loading {model_path}")
    model = load_model(model_path, torch.bfloat16, device)
    scores = score_tokenized(model, device, tokenized, batch_size, pad_id, micro_batch_size)
    del model
    torch.cuda.empty_cache()
    return scores


def merge_partials(output_path: Path, num_shards: int) -> int:
    rows = []
    for sid in range(num_shards):
        part = output_path.with_suffix(f".part{sid}.jsonl")
        if not part.is_file():
            raise FileNotFoundError(f"Missing shard output: {part}")
        with part.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for sid in range(num_shards):
        output_path.with_suffix(f".part{sid}.jsonl").unlink(missing_ok=True)
    return len(rows)


def run_sharded_scoring(
    model_path: str,
    df: pd.DataFrame,
    partial_path: Path,
    batch_size: int,
    max_length: int,
    num_shards: int,
    token_workers: int,
    gpus: list[int],
) -> tuple[list[float], list[float]]:
    if num_shards != len(gpus):
        raise ValueError(f"num_shards ({num_shards}) must match len(gpus) ({len(gpus)})")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for sid, gpu_id in enumerate(gpus):
        shard_df_slice = shard_df(df, sid, num_shards)
        part_out = partial_path.with_suffix(f".part{sid}.jsonl")
        p = ctx.Process(
            target=_worker_score,
            args=(model_path, shard_df_slice, part_out, batch_size, max_length, token_workers, gpu_id),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    chosen_all: list[float] = [0.0] * len(df)
    rejected_all: list[float] = [0.0] * len(df)
    for sid in range(num_shards):
        part = partial_path.with_suffix(f".part{sid}.jsonl")
        scores_c: list[float] = []
        scores_r: list[float] = []
        with part.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                scores_c.append(rec["chosen_logprob"])
                scores_r.append(rec["rejected_logprob"])
        shard_indices = list(range(sid, len(df), num_shards))
        for local_i, global_idx in enumerate(shard_indices):
            chosen_all[global_idx] = scores_c[local_i]
            rejected_all[global_idx] = scores_r[local_i]
    return chosen_all, rejected_all


def run_sharded_margin_scoring(
    policy_model: str,
    ref_model: str,
    df: pd.DataFrame,
    partial_path: Path,
    batch_size: int,
    max_length: int,
    num_shards: int,
    token_workers: int,
    gpus: list[int],
    micro_batch_size: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if num_shards != len(gpus):
        raise ValueError(f"num_shards ({num_shards}) must match len(gpus) ({len(gpus)})")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for sid, gpu_id in enumerate(gpus):
        shard_df_slice = shard_df(df, sid, num_shards)
        part_out = partial_path.with_suffix(f".part{sid}.jsonl")
        p = ctx.Process(
            target=_worker_score_both,
            args=(
                policy_model,
                ref_model,
                shard_df_slice,
                part_out,
                batch_size,
                max_length,
                token_workers,
                gpu_id,
                micro_batch_size,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    ref_c: list[float] = [0.0] * len(df)
    ref_r: list[float] = [0.0] * len(df)
    pol_c: list[float] = [0.0] * len(df)
    pol_r: list[float] = [0.0] * len(df)
    for sid in range(num_shards):
        part = partial_path.with_suffix(f".part{sid}.jsonl")
        shard_indices = list(range(sid, len(df), num_shards))
        with part.open(encoding="utf-8") as f:
            for local_i, line in enumerate(f):
                rec = json.loads(line)
                global_idx = shard_indices[local_i]
                ref_c[global_idx] = rec["ref_chosen_logprob"]
                ref_r[global_idx] = rec["ref_rejected_logprob"]
                pol_c[global_idx] = rec["policy_chosen_logprob"]
                pol_r[global_idx] = rec["policy_rejected_logprob"]
    return ref_c, ref_r, pol_c, pol_r


def _worker_score_both(
    policy_model: str,
    ref_model: str,
    df: pd.DataFrame,
    part_out: Path,
    batch_size: int,
    max_length: int,
    token_workers: int,
    gpu_id: int,
    micro_batch_size: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = load_scoring_tokenizer(policy_model, ref_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    print(f"[gpu {gpu_id}] tokenizing {len(df)} rows (workers={token_workers})...")
    tokenized = pre_tokenize(df, tokenizer, max_length, token_workers)

    ref_c, ref_r = score_model_shard(
        ref_model, df, batch_size, max_length, token_workers, 0,
        tokenized=tokenized, tokenizer=tokenizer, pad_id=pad_id,
        micro_batch_size=micro_batch_size,
    )
    pol_c, pol_r = score_model_shard(
        policy_model, df, batch_size, max_length, token_workers, 0,
        tokenized=tokenized, tokenizer=tokenizer, pad_id=pad_id,
        micro_batch_size=micro_batch_size,
    )

    part_out.parent.mkdir(parents=True, exist_ok=True)
    with part_out.open("w", encoding="utf-8") as fout:
        for rc, rr, pc, pr in zip(ref_c, ref_r, pol_c, pol_r):
            fout.write(
                json.dumps(
                    {
                        "ref_chosen_logprob": rc,
                        "ref_rejected_logprob": rr,
                        "policy_chosen_logprob": pc,
                        "policy_rejected_logprob": pr,
                    }
                )
                + "\n"
            )


def _worker_score(
    model_path: str,
    df: pd.DataFrame,
    part_out: Path,
    batch_size: int,
    max_length: int,
    token_workers: int,
    gpu_id: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    chosen, rejected = score_model_shard(
        model_path, df, batch_size, max_length, token_workers, 0
    )
    part_out.parent.mkdir(parents=True, exist_ok=True)
    with part_out.open("w", encoding="utf-8") as fout:
        for c, r in zip(chosen, rejected):
            fout.write(json.dumps({"chosen_logprob": c, "rejected_logprob": r}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--policy-model", type=str, required=True)
    parser.add_argument("--ref-model", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--token-workers", type=int, default=8)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only validate model paths and chat templates; exit without scoring.",
    )
    parser.add_argument(
        "--save-rad",
        action="store_true",
        help="Store rad=margin. Use when policy=π_θ^pos and ref=π_θ^inv (paper Eq. 19).",
    )
    args = parser.parse_args()

    preflight_margin_scoring(args.policy_model, args.ref_model)
    if args.preflight_only:
        return

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpus) != args.num_shards:
        args.num_shards = len(gpus)

    df = pd.read_json(args.data_path, lines=True).dropna(subset=["prompt", "chosen", "rejected"])
    print(
        f"Loaded {len(df)} examples | batch_size={args.batch_size} "
        f"micro_batch={args.micro_batch_size} shards={args.num_shards} gpus={gpus}"
    )

    partial = args.output_path.with_suffix(".partial.jsonl")

    print(f"Scoring REF+POLICY shards on gpus={gpus}")
    ref_c, ref_r, pol_c, pol_r = run_sharded_margin_scoring(
        args.policy_model,
        args.ref_model,
        df,
        partial,
        args.batch_size,
        args.max_length,
        args.num_shards,
        args.token_workers,
        gpus,
        args.micro_batch_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fout:
        for idx in range(len(df)):
            row = df.iloc[idx]
            margin = (pol_c[idx] - pol_r[idx]) - (ref_c[idx] - ref_r[idx])
            rec = {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "policy_chosen_logprob": pol_c[idx],
                "policy_rejected_logprob": pol_r[idx],
                "ref_chosen_logprob": ref_c[idx],
                "ref_rejected_logprob": ref_r[idx],
                "margin": margin,
            }
            if args.save_rad:
                rec["rad"] = margin
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    partial.unlink(missing_ok=True)
    for sid in range(args.num_shards):
        partial.with_suffix(f".part{sid}.jsonl").unlink(missing_ok=True)

    print(f"Saved {len(df)} rows -> {args.output_path}")


if __name__ == "__main__":
    main()
