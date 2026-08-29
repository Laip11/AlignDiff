#!/usr/bin/env python3
"""Convert preference datasets to OpenRLHF JSONL: prompt / chosen / rejected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _text(message: dict) -> str:
    return message.get("content") or message.get("value") or ""


def _role(message: dict) -> str:
    return (message.get("role") or message.get("from") or "").lower()


def _user_text(example: dict) -> str:
    instruction = example.get("instruction") or example.get("prompt")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    for key in ("chosen", "rejected"):
        messages = example.get(key) or []
        if isinstance(messages, list):
            for message in messages:
                if _role(message) in {"user", "human"}:
                    return _text(message)
        elif isinstance(messages, dict) and _role(messages) in {"user", "human"}:
            return _text(messages)
    return ""


def _assistant_text(field) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        return _text(field)
    if isinstance(field, list) and field:
        for message in reversed(field):
            if _role(message) in {"assistant", "gpt", ""}:
                return _text(message)
        return _text(field[-1])
    return ""


def to_preference_row(example: dict, flip: bool = False) -> dict | None:
    prompt = _user_text(example)
    chosen = _assistant_text(example.get("chosen"))
    rejected = _assistant_text(example.get("rejected"))
    if flip:
        chosen, rejected = rejected, chosen
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def sharegpt_to_flat(example: dict) -> dict | None:
    convs = example.get("conversations") or []
    prompt = ""
    if convs and isinstance(convs[0], dict):
        prompt = (convs[0].get("value") or convs[0].get("content") or "").strip()
    if not prompt:
        prompt = str(example.get("prompt") or "").strip()

    chosen_field = example.get("chosen")
    rejected_field = example.get("rejected")
    if isinstance(chosen_field, dict):
        chosen = (chosen_field.get("value") or chosen_field.get("content") or "").strip()
    else:
        chosen = str(chosen_field or "").strip()
    if isinstance(rejected_field, dict):
        rejected = (rejected_field.get("value") or rejected_field.get("content") or "").strip()
    else:
        rejected = str(rejected_field or "").strip()

    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def cmd_hf_prefs(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    output = Path(args.output)
    dataset = load_dataset(args.repo, split=args.split)
    kept, skipped = 0, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for example in dataset:
            converted = to_preference_row(example, flip=args.flip)
            if converted is None:
                skipped += 1
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} preference rows to {output} (skipped {skipped})")


def cmd_flip(args: argparse.Namespace) -> None:
    kept, skipped = 0, 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            example = json.loads(line)
            row = to_preference_row(example) or sharegpt_to_flat(example)
            if row is None:
                skipped += 1
                continue
            row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"flipped {kept} rows to {output} (skipped {skipped})")


def cmd_to_flat(args: argparse.Namespace) -> None:
    kept, skipped = 0, 0
    rows = []
    for example in _iter_jsonl(Path(args.input)):
        flat = to_preference_row(example) or sharegpt_to_flat(example)
        if flat is None:
            skipped += 1
            continue
        rows.append(flat)
        kept += 1
    _write_jsonl(Path(args.output), rows)
    print(f"flat: kept={kept} skipped={skipped} -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignDiff data format conversion.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_uf = sub.add_parser(
        "ultrafeedback",
        help="HF UltraFeedback_Binarized -> prompt/chosen/rejected JSONL",
    )
    p_uf.add_argument("--repo", default="HuggingFaceH4/ultrafeedback_binarized")
    p_uf.add_argument("--split", default="train_prefs")
    p_uf.add_argument("--output", required=True)
    p_uf.add_argument("--flip", action="store_true")
    p_uf.set_defaults(func=cmd_hf_prefs)

    p_flip = sub.add_parser("flip", help="Swap chosen/rejected in an existing JSONL")
    p_flip.add_argument("--input", required=True)
    p_flip.add_argument("--output", required=True)
    p_flip.set_defaults(func=cmd_flip)

    p_flat = sub.add_parser("to-flat", help="sharegpt or mixed JSONL -> prompt/chosen/rejected")
    p_flat.add_argument("--input", required=True)
    p_flat.add_argument("--output", required=True)
    p_flat.set_defaults(func=cmd_to_flat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
