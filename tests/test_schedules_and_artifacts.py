"""Integrity tests for frozen training schedules, dynamics logs, and curated trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from minibooking_rlvr.data import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_sft_and_grpo_schedules_are_disjoint_and_verified():
    train_ids = {r["sample_id"] for r in load_jsonl(REPO_ROOT / "data" / "train.jsonl")}
    sft_spec = json.loads((REPO_ROOT / "configs" / "sft_train_ids.json").read_text(encoding="utf-8"))
    grpo_spec = json.loads((REPO_ROOT / "configs" / "grpo_train_order.json").read_text(encoding="utf-8"))

    sft_ids = sft_spec["sample_ids"]
    grpo_ids = grpo_spec["sample_ids"]

    assert len(sft_ids) == 600 and len(set(sft_ids)) == 600
    assert set(sft_ids).issubset(train_ids)

    assert len(grpo_ids) == 800 and len(set(grpo_ids)) == 702
    assert set(grpo_ids).issubset(train_ids)
    # Zero overlap between SFT subset and GRPO training pool
    assert set(sft_ids).isdisjoint(set(grpo_ids))

    got_sha = hashlib.sha256(",".join(grpo_ids).encode("utf-8")).hexdigest()[:16]
    assert got_sha == grpo_spec["train_order_sha"]


def test_training_dynamics_and_curated_trajectories_schema():
    dyn = json.loads((REPO_ROOT / "results" / "training_dynamics.json").read_text(encoding="utf-8"))
    for g in ("G1", "G2", "G3"):
        rows = dyn["groups"][g]
        assert len(rows) == 200
        assert [r["step"] for r in rows] == list(range(1, 201))
    assert dyn["lr_sweep"]["lr_1e-5"]["stable"] is True
    assert dyn["lr_sweep"]["lr_2e-5"]["stable"] is False

    curated = json.loads((REPO_ROOT / "results" / "sample_trajectories.json").read_text(encoding="utf-8"))
    assert len(curated["cases"]) >= 6
    for case in curated["cases"]:
        assert set(case["predictors"]) == {"SFT", "G1", "G2", "G3", "oracle", "null"}

    viewer_html = (REPO_ROOT / "demos" / "trajectory_viewer.html").read_text(encoding="utf-8")
    assert "MiniBooking-RLVR" in viewer_html and "test_comp-000" in viewer_html

    dev_grid = json.loads((REPO_ROOT / "results" / "planner_dev_grid.json").read_text(encoding="utf-8"))
    assert len(dev_grid["configs"]) == 18
    assert all(set(c["per_model"]) == {"SFT", "G1", "G2", "G3"} for c in dev_grid["configs"])


def test_archived_raw_predictions_reproduce_summary_metrics():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "reproduce_eval_tables.py"
    spec = importlib.util.spec_from_file_location("reproduce_eval_tables", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    recomputed = mod.recompute_all(REPO_ROOT / "data", REPO_ROOT / "results" / "predictions")
    summary = json.loads((REPO_ROOT / "results" / "summary_metrics.json").read_text(encoding="utf-8"))
    mismatches = mod.verify_against_summary(recomputed, summary)
    assert mismatches == [], f"Audit mismatches: {mismatches}"
