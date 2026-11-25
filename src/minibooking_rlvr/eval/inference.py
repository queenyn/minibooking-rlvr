"""Batched greedy HuggingFace inference wrapper with strict schema parsing."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from minibooking_rlvr.data.prompt import PROMPT_VERSION, build_prompt
from minibooking_rlvr.rewards import PredictionError, parse_prediction


def eos_token_ids(model: Any, tokenizer: Any) -> List[int]:
    """Collect EOS token IDs including chat-template ``<|im_end|>`` when present."""
    ids = set()
    for val in (getattr(getattr(model, "generation_config", None), "eos_token_id", None), tokenizer.eos_token_id):
        if val is None:
            continue
        ids.update(val if isinstance(val, (list, tuple)) else [val])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(ids)


def run_inference(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """Run greedy decoding and strict parsing on a sequence of MiniBooking records."""
    import torch

    eos = eos_token_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    device = next(model.parameters()).device
    old_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    model.eval()
    results: List[Dict[str, Any]] = []
    try:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            prompts = [build_prompt(tokenizer, r, enable_thinking=enable_thinking) for r in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
            gen = out[:, enc["input_ids"].shape[1] :].tolist()
            for j, (rec, ids) in enumerate(zip(chunk, gen)):
                stop = next((i for i, tok_id in enumerate(ids) if tok_id in eos), None)
                body = ids if stop is None else ids[:stop]
                raw = tokenizer.decode(body, skip_special_tokens=False)
                parsed, valid, error = None, False, None
                try:
                    parsed = parse_prediction(raw)
                    valid = True
                except PredictionError as exc:
                    error = f"PredictionError: {exc}"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    {
                        "sample_id": rec.get("sample_id"),
                        "prompt_version": PROMPT_VERSION,
                        "prompt_tokens": int(enc["attention_mask"][j].sum()),
                        "max_new_tokens": max_new_tokens,
                        "raw": raw,
                        "n_new_tokens": len(body) + (0 if stop is None else 1),
                        "hit_eos": stop is not None,
                        "truncated": stop is None and len(ids) >= max_new_tokens,
                        "has_think": "<think>" in raw,
                        "valid": valid,
                        "parsed": parsed,
                        "error": error,
                    }
                )
    finally:
        tokenizer.padding_side = old_side
    return results
