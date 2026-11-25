#!/usr/bin/env python3
"""Pure-PyTorch GRPO + LoRA trainer (G1 / G2 / G3) with H/D/S/C diagnostics and EntropyController.

Replaces notebook-only training cells with a clean, reproducible CLI entry point backed by
``minibooking_rlvr.rl.{grpo, diagnostics, controller}``. Requires the ``[train]`` stack and CUDA.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from minibooking_rlvr.data import build_prompt, load_jsonl
from minibooking_rlvr.eval.inference import eos_token_ids
from minibooking_rlvr.rl.controller import EntropyController
from minibooking_rlvr.rl.diagnostics import batch_stats, group_stats, make_sample_row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pure-PyTorch GRPO trainer for MiniBooking-RLVR.")
    parser.add_argument("--group", choices=("G1", "G2", "G3"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_formal_config.json"))
    parser.add_argument("--train-order", type=Path, default=Path("configs/grpo_train_order.json"), help="Optional frozen 800-slot GRPO prompt schedule.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sft-adapter", type=Path, required=True, help="Path to frozen LoRA SFT adapter.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Override total optimizer steps (default from config).")
    parser.add_argument("--ent-ratio-every", type=int, default=10, help="Measure gradient norm ratio every N steps.")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minibooking_rlvr.rl.grpo import (
        compute_group_advantages,
        compute_grpo_objective,
        entropy_summary,
        masked_token_entropy,
        token_logprobs_from_logits,
    )

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    gcfg = cfg["grpo"]
    group_spec = cfg["groups"][args.group]
    total_steps = args.steps or int(gcfg["steps"])
    seed = int(gcfg["seed"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"]["model_id"], revision=cfg["base_model"]["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["model_id"],
        revision=cfg["base_model"]["revision"],
        torch_dtype=dtype,
        device_map="auto",
    )
    # Merge frozen SFT adapter in memory so policy start == reference (disable_adapter() gives reference logprobs).
    sft_merged = PeftModel.from_pretrained(base, str(args.sft_adapter)).merge_and_unload()
    lora_spec = cfg["sft"]["lora"]
    model = get_peft_model(
        sft_merged,
        LoraConfig(
            r=lora_spec["r"],
            lora_alpha=lora_spec["alpha"],
            lora_dropout=lora_spec["dropout"],
            target_modules=lora_spec["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(gcfg["lr"]),
        betas=tuple(gcfg["betas"]),
        eps=float(gcfg["adam_eps"]),
        weight_decay=float(gcfg["weight_decay"]),
    )
    warmup = int(gcfg["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / warmup))

    controller = EntropyController(**group_spec["controller"]) if group_spec["entropy_controller"] else None
    with_reg = bool(group_spec["entropy_regularization"])

    all_train = load_jsonl(args.data_dir / "train.jsonl")
    if args.train_order and args.train_order.exists():
        import hashlib

        order_spec = json.loads(args.train_order.read_text(encoding="utf-8"))
        order_ids = order_spec["sample_ids"]
        got_sha = hashlib.sha256(",".join(order_ids).encode("utf-8")).hexdigest()[:16]
        if got_sha != order_spec["train_order_sha"]:
            raise ValueError(f"train_order_sha mismatch: {got_sha} != {order_spec['train_order_sha']}")
        by_id = {r["sample_id"]: r for r in all_train}
        records = [by_id[sid] for sid in order_ids]
    else:
        records = all_train
    k_size = int(gcfg["group_size_K"])
    n_prompts = int(gcfg["prompts_per_step"])
    eos_ids = eos_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    eos_t = torch.tensor(eos_ids, device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_log = args.out_dir / "steps.jsonl"

    for step in range(1, total_steps + 1):
        start = ((step - 1) * n_prompts) % len(records)
        batch_recs = [records[(start + i) % len(records)] for i in range(n_prompts)]
        prompts = [build_prompt(tokenizer, r, enable_thinking=False) for r in batch_recs]

        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            gen_out = model.generate(
                **enc,
                do_sample=True,
                temperature=float(gcfg["temperature"]),
                top_p=float(gcfg["top_p"]),
                top_k=int(gcfg["top_k"]),
                num_return_sequences=k_size,
                max_new_tokens=int(gcfg["max_new_tokens"]),
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                use_cache=True,
            )

        p_len = enc["input_ids"].shape[1]
        comp = gen_out[:, p_len:]
        lc = comp.shape[1]
        is_eos = torch.isin(comp, eos_t)
        hit = is_eos.any(dim=1)
        first = torch.where(hit, is_eos.float().argmax(dim=1), torch.full_like(hit, lc, dtype=torch.long))
        n_tok = torch.where(hit, first + 1, first)
        comp_mask = torch.arange(lc, device=device)[None] < n_tok[:, None]

        rows: List[Dict[str, Any]] = []
        for idx in range(gen_out.shape[0]):
            rec = batch_recs[idx // k_size]
            raw = tokenizer.decode(comp[idx, : first[idx]].tolist(), skip_special_tokens=False)
            rows.append(make_sample_row(raw, rec["gold"], args.group))

        rewards_t = torch.tensor([r["reward"] for r in rows], dtype=torch.float32, device=device)
        adv_all, _, _ = compute_group_advantages(rewards_t, group_size=k_size, eps=float(gcfg["adv_eps"]))

        lam = controller.lam() if controller is not None else 0.0
        model.train()
        p_ids = enc["input_ids"].repeat_interleave(k_size, dim=0)
        p_att = enc["attention_mask"].repeat_interleave(k_size, dim=0)
        full_ids = torch.cat([p_ids, comp], dim=1)
        full_att = torch.cat([p_att, comp_mask.long()], dim=1)
        pos_ids = (full_att.cumsum(dim=-1) - 1).clamp(min=0)

        with torch.no_grad(), model.disable_adapter(), torch.autocast(device.type, dtype=dtype):
            ref_logits = model(input_ids=full_ids, attention_mask=full_att, position_ids=pos_ids, logits_to_keep=lc + 1, use_cache=False).logits[:, :-1]
            ref_lp = token_logprobs_from_logits(ref_logits, comp, temperature=float(gcfg["logprob_temperature"]))

        with torch.autocast(device.type, dtype=dtype):
            logits = model(input_ids=full_ids, attention_mask=full_att, position_ids=pos_ids, logits_to_keep=lc + 1, use_cache=False).logits[:, :-1]

        measure_ratio = args.ent_ratio_every > 0 and (step % args.ent_ratio_every == 0)
        loss, obj_info = compute_grpo_objective(
            logits,
            comp,
            ref_lp,
            adv_all,
            comp_mask,
            beta_kl=float(gcfg["beta_kl"]),
            lambda_entropy=lam,
            with_entropy_reg=with_reg,
            clip_eps=float(gcfg["clip_eps"]),
            logprob_temperature=float(gcfg["logprob_temperature"]),
            reg_temperature=float(group_spec.get("reg_temperature", 1.0)),
            params_for_ratio=trainable if measure_ratio else None,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(gcfg["max_grad_norm"])))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        ent_tok = masked_token_entropy(logits, comp_mask, temperature=float(gcfg["entropy_log_temperature"]))
        gs = [group_stats(rows[g * k_size : (g + 1) * k_size]) for g in range(n_prompts)]
        bs = batch_stats(gs)
        eff_mask = torch.tensor([not gs[i // k_size]["zero_var"] for i in range(len(rows))], device=device)[:, None] & comp_mask
        h_eff = entropy_summary(ent_tok, eff_mask)["mean"]
        exact_rate = float(np.mean([r["exact"] for r in rows]))
        ctrl_ev = controller.observe(step, h_eff, exact_rate) if controller is not None else None

        rec_out = {
            "step": step,
            "group": args.group,
            "lr": scheduler.get_last_lr()[0],
            "lambda_entropy": lam,
            "grad_norm": grad_norm,
            "H_eff_mean": h_eff,
            "exact_rate": exact_rate,
            **{k: v for k, v in obj_info.items() if k != "seq_kl"},
            **bs,
            "ctrl": ctrl_ev,
        }
        with open(step_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        print(
            f"[{args.group}] step={step:03d} loss={obj_info['loss']:.4f} kl={obj_info['kl']:.4f} "
            f"exact={exact_rate:.3f} eff_groups={bs['n_effective']}/{n_prompts} lam={lam:.5f}"
        )

    final_dir = args.out_dir / "final_adapter"
    model.save_pretrained(final_dir)
    if controller is not None:
        controller.save(final_dir / "controller_state.json")
    print(f"Finished {args.group} GRPO training; saved adapter to {final_dir}.")


if __name__ == "__main__":
    main()
