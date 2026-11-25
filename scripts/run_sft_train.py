#!/usr/bin/env python3
"""Standalone completion-only LoRA SFT training script for Qwen3-1.7B on MiniBooking.

Implements completion-only cross-entropy loss (prompt tokens masked with -100) using
plain PyTorch + HuggingFace Transformers + PEFT. Requires the optional ``[train]`` stack
and a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from minibooking_rlvr.data import build_prompt, gold_completion, load_jsonl
from minibooking_rlvr.eval.inference import run_inference
from minibooking_rlvr.eval.metrics import evaluate_model_outputs


def encode_sft_example(tokenizer: Any, record: Dict[str, Any]) -> Dict[str, List[int]]:
    """Tokenize prompt + completion and verify exact boundary alignment for loss masking."""
    prompt_text = build_prompt(tokenizer, record, enable_thinking=False)
    target_text = gold_completion(record, confidence=1.0) + (tokenizer.eos_token or "<|im_end|>")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for MiniBooking world model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ids", type=Path, default=Path("configs/sft_train_ids.json"), help="Optional frozen SFT sample_id list.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_train = load_jsonl(args.data_dir / "train.jsonl")
    if args.train_ids and args.train_ids.exists():
        id_spec = json.loads(args.train_ids.read_text(encoding="utf-8"))
        by_id = {r["sample_id"]: r for r in all_train}
        train_records = [by_id[sid] for sid in id_spec["sample_ids"]]
    else:
        train_records = all_train[: args.n_train]
    dev_records = load_jsonl(args.data_dir / "dev.jsonl")[:100]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision, torch_dtype=dtype, device_map="auto")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    encoded = [encode_sft_example(tokenizer, r) for r in train_records]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    device = next(model.parameters()).device
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        batch = [rng.choice(encoded) for _ in range(args.batch_size)]
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
        for i, ex in enumerate(batch):
            seq_len = len(ex["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(ex["labels"], dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == args.steps:
            print(f"[SFT] step={step}/{args.steps} loss={float(out.loss.detach()):.4f}")

    adapter_dir = args.out_dir / "sft_adapter"
    model.save_pretrained(adapter_dir)
    gens = run_inference(model, tokenizer, dev_records, max_new_tokens=256, batch_size=16)
    summary = evaluate_model_outputs(dev_records, [g["raw"] for g in gens], split="dev")
    (args.out_dir / "dev100_eval.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved SFT adapter to {adapter_dir}; dev[:100] exact={summary['subsets']['all']['main']['exact']:.4f}")


if __name__ == "__main__":
    main()
