#!/usr/bin/env python3
"""Recompute frozen evaluation tables directly from raw model completions on pure CPU.

Reads ``data/{test_id,test_comp}.jsonl`` and ``results/predictions/{SFT,G1,G2,G3}__{split}.jsonl``
and runs ``minibooking_rlvr.eval.evaluate_model_outputs`` from scratch—zero GPU required.

Usage:
    python -X utf8 scripts/reproduce_eval_tables.py
    python -X utf8 scripts/reproduce_eval_tables.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minibooking_rlvr.data import load_jsonl
from minibooking_rlvr.eval import evaluate_model_outputs, render_eval_table

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("SFT", "G1", "G2", "G3")
SPLITS = ("test_id", "test_comp")


def recompute_all(data_dir: Path, pred_dir: Path) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    for split in SPLITS:
        records = load_jsonl(data_dir / f"{split}.jsonl")
        rec_ids = [r["sample_id"] for r in records]
        results[split] = {}
        for model in MODELS:
            preds = load_jsonl(pred_dir / f"{model}__{split}.jsonl")
            pred_ids = [p["sample_id"] for p in preds]
            if pred_ids != rec_ids:
                raise ValueError(f"sample_id alignment mismatch in {model}__{split}.jsonl")
            raws = [p["raw"] for p in preds]
            results[split][model] = evaluate_model_outputs(records, raws, split=split)
    return results


def verify_against_summary(recomputed: dict, summary: dict, tol: float = 1e-4) -> list[str]:
    mismatches: list[str] = []
    field_map = {
        "json_valid": lambda a: a["json_valid_rate"],
        "parse_coverage": lambda a: a["parse_coverage"],
        "exact": lambda a: a["main"]["exact"],
        "delta_f1": lambda a: a["main"]["delta_f1"],
        "full_exact": lambda a: a["main"]["full_exact"],
        "rev_macro_f1": lambda a: a["main"]["reversible_macro_f1"],
        "rev_acc": lambda a: a["main"]["reversible_acc"],
        "cost_mae": lambda a: a["main"]["cost_mae"],
        "brier": lambda a: a["main"]["brier"],
        "frac_conf_eq_1": lambda a: a["confidence_dist_valid"]["frac_eq_1"],
    }
    for split in SPLITS:
        for model in MODELS:
            agg = recomputed[split][model]["subsets"]["all"]
            expected = summary["frozen_eval"][split][model]
            for key, getter in field_map.items():
                got = float(getter(agg))
                want = float(expected[key])
                if abs(got - want) > tol:
                    mismatches.append(f"{split}/{model}/{key}: got {got:.6f} != summary {want:.6f}")

    # Verify compositional allow_change=True subset as well
    for model in MODELS:
        sub = recomputed["test_comp"][model]["subsets"]["allow_change=True"]
        exp = summary["frozen_eval"]["test_comp_allow_change_true"][model]
        for k, src_k in (("exact", "exact"), ("full_exact", "full_exact"), ("rev_macro_f1", "reversible_macro_f1")):
            got = float(sub["main"][src_k])
            want = float(exp[k])
            if abs(got - want) > tol:
                mismatches.append(f"test_comp[allow_change=True]/{model}/{k}: got {got:.6f} != summary {want:.6f}")

    # Verify inspect regression == 1.0 across test_id and test_comp
    for split in SPLITS:
        for model in MODELS:
            got = float(recomputed[split][model]["subsets"]["action=inspect"]["main"]["exact"])
            if abs(got - 1.0) > 1e-12:
                mismatches.append(f"{split}/{model}/inspect_exact: got {got:.6f} != 1.0")

    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit-recompute frozen evaluation metrics from raw predictions on CPU.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--pred-dir", type=Path, default=REPO_ROOT / "results" / "predictions")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results" / "summary_metrics.json")
    parser.add_argument("--verify", action="store_true", help="Assert exact agreement with results/summary_metrics.json.")
    args = parser.parse_args(argv)

    recomputed = recompute_all(args.data_dir, args.pred_dir)
    for split in SPLITS:
        print(render_eval_table(recomputed[split], split))

    if args.verify:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        mismatches = verify_against_summary(recomputed, summary)
        if mismatches:
            print("AUDIT VERIFICATION FAILED:")
            for msg in mismatches:
                print("  -", msg)
            return 1
        print("AUDIT VERIFICATION PASSED: all 2,400 raw completions match results/summary_metrics.json to 4 decimal places.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
