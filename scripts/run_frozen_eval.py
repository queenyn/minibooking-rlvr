#!/usr/bin/env python3
"""Run frozen greedy evaluation on ``test_id`` and ``test_comp`` and render dual-view tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibooking_rlvr.data import load_jsonl, save_jsonl
from minibooking_rlvr.eval import evaluate_model_outputs, render_eval_table
from minibooking_rlvr.eval.inference import run_inference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Frozen greedy evaluation for MiniBooking-RLVR.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--rl-adapter", type=Path, default=None)
    parser.add_argument("--name", default="model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--splits", nargs="+", default=["test_id", "test_comp"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, str(args.sft_adapter)).merge_and_unload()
    if args.rl_adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.rl_adapter))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        recs = load_jsonl(args.data_dir / f"{split}.jsonl")
        gens = run_inference(model, tokenizer, recs, max_new_tokens=256, batch_size=args.batch_size)
        save_jsonl(gens, args.out_dir / f"{args.name}__{split}.raw.jsonl")
        res = evaluate_model_outputs(recs, [g["raw"] for g in gens], split=split)
        (args.out_dir / f"{args.name}__{split}.metrics.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(render_eval_table({args.name: res}, split))


if __name__ == "__main__":
    main()
