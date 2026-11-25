"""Unit tests for the risk-aware 1-step MPC planner, candidate isolation, baselines, and BFS tasks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from minibooking_rlvr.planner import (
    END_GOAL,
    END_MAX_STEPS,
    END_NO_VALID,
    END_THRESHOLD,
    MAX_CANDIDATES,
    bfs_label,
    breakdown_episodes,
    decide_step,
    env_step,
    null_predict_batch,
    oracle_predict_batch,
    progress,
    propose_candidates,
    render_planner_table,
    run_tasks_batched,
    score_components,
    should_stop,
    summarize_episodes,
)

RULES = {
    "price_cents": 100,
    "refund_num": 1,
    "refund_den": 2,
    "allow_change": True,
    "change_fee_cents": 20,
    "allowed_days": [1, 2, 3],
}


def _sample_tasks():
    t_sol = {
        "task_id": "t-000",
        "split": "test_id",
        "category": "create_confirm",
        "scenario_id": "sc-1",
        "rules": RULES,
        "start_state": {"status": "empty", "day": None, "balance_cents": 500, "paid_cents": 0, "loss_cents": 0},
        "goal": {"conditions": [{"field": "status", "op": "eq", "value": "confirmed"}, {"field": "day", "op": "eq", "value": 2}]},
        "solvable": True,
        "tags": {"source": "test_id", "allow_change": True, "full_refund_fee20": False, "ref_contains_cancel": False},
    }
    t_unsol = {
        "task_id": "t-001",
        "split": "test_comp",
        "category": "unsolvable",
        "scenario_id": "sc-2",
        "rules": RULES,
        "start_state": {"status": "empty", "day": None, "balance_cents": 50, "paid_cents": 0, "loss_cents": 0},
        "goal": {"conditions": [{"field": "status", "op": "eq", "value": "confirmed"}, {"field": "balance_cents", "op": "ge", "value": 51}]},
        "solvable": False,
        "tags": {"source": "test_comp", "allow_change": True, "full_refund_fee20": False, "ref_contains_cancel": False},
    }
    return [t_sol, t_unsol]


def test_candidates_module_does_not_import_env_internals():
    cand_path = Path(__file__).resolve().parents[1] / "src" / "minibooking_rlvr" / "planner" / "candidates.py"
    tree = ast.parse(cand_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "env" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "env" not in node.module


def test_propose_candidates_bounded_and_deterministic():
    tasks = _sample_tasks()
    c1 = propose_candidates(tasks[0]["start_state"], tasks[0]["goal"], RULES)
    c2 = propose_candidates(tasks[0]["start_state"], tasks[0]["goal"], RULES)
    assert c1 == c2 and 1 <= len(c1) <= MAX_CANDIDATES
    assert all(a["name"] != "inspect" for a in c1)


def test_progress_score_and_stop_precedence():
    st = {"status": "draft", "day": 2, "balance_cents": 500, "paid_cents": 0, "loss_cents": 20}
    goal = {
        "conditions": [{"field": "status", "op": "eq", "value": "confirmed"}, {"field": "day", "op": "eq", "value": 2}],
        "loss_limit_cents": 10,
    }
    assert progress(st, goal) == 0.0  # loss exceeds limit
    goal["loss_limit_cents"] = 50
    assert progress(st, goal) == 0.5

    weights = {"w_risk": 0.5, "w_irr": 0.25, "w_conf": 0.1}
    invalid_comp = score_components({"valid": False}, st, goal, weights)
    assert invalid_comp["score"] == float("-inf")

    assert should_stop(0.0, 0.0, 0.0, is_goal_met=True, step_index=5) == (True, END_GOAL)
    assert should_stop(1.0, 0.0, 0.0, is_goal_met=False, step_index=5, max_steps=5) == (True, END_MAX_STEPS)
    assert should_stop(float("-inf"), 0.0, 0.0, is_goal_met=False, step_index=0) == (True, END_NO_VALID)
    assert should_stop(0.5, 0.5, 0.0, is_goal_met=False, step_index=0) == (True, END_THRESHOLD)


def test_decide_step_tie_breaks_by_candidate_order():
    st = {"status": "empty", "day": None, "balance_cents": 500, "paid_cents": 0, "loss_cents": 0}
    goal = {"conditions": [{"field": "status", "op": "eq", "value": "draft"}]}
    cands = [{"name": "create", "day": 1}, {"name": "create", "day": 2}]
    preds = [
        {"valid": True, "changes": [{"field": "status", "value": "draft"}], "cost": 0.0, "reversible": True, "confidence": 1.0},
        {"valid": True, "changes": [{"field": "status", "value": "draft"}], "cost": 0.0, "reversible": True, "confidence": 1.0},
    ]
    d = decide_step(st, goal, cands, preds, {"w_risk": 0.0, "w_irr": 0.0, "w_conf": 0.1}, tau=0.0, step_index=0)
    assert d["best_idx"] == 0 and d["stop"] is False


def test_oracle_and_null_closed_loop_and_resume_consistency():
    tasks = _sample_tasks()
    weights = {"w_risk": 0.0, "w_irr": 0.0, "w_conf": 0.1}

    # Null baseline always stops at step 0 with 0 solvable success and 1.0 unsolvable refusal
    out_null = run_tasks_batched(tasks, propose_candidates, null_predict_batch, env_step, weights, tau=0.0, predictor_kind="cpu")
    summ_null = summarize_episodes(out_null["episodes"], tasks, out_null["batches"])
    assert summ_null["solvable_success_rate"] == 0.0
    assert summ_null["unsolvable_refusal_rate"] == 1.0
    assert summ_null["avg_steps"] == 0.0

    # Oracle solves create->confirm in 2 steps and refuses unsolvable task
    snapshots = []
    out_oracle = run_tasks_batched(
        tasks,
        propose_candidates,
        oracle_predict_batch,
        env_step,
        weights,
        tau=0.0,
        predictor_kind="cpu",
        on_round=lambda snap: snapshots.append(snap),
    )
    summ_oracle = summarize_episodes(out_oracle["episodes"], tasks, out_oracle["batches"])
    assert summ_oracle["solvable_success_rate"] == 1.0
    assert summ_oracle["unsolvable_refusal_rate"] == 1.0

    # Resume from round-1 snapshot matches uninterrupted run
    resumed = run_tasks_batched(
        tasks,
        propose_candidates,
        oracle_predict_batch,
        env_step,
        weights,
        tau=0.0,
        predictor_kind="cpu",
        resume_state=snapshots[0],
    )
    assert [e["final_state"] for e in resumed["episodes"]] == [e["final_state"] for e in out_oracle["episodes"]]
    assert [e["end_reason"] for e in resumed["episodes"]] == [e["end_reason"] for e in out_oracle["episodes"]]

    bd = breakdown_episodes(out_oracle["episodes"], tasks)
    assert "category" in bd and "comp_allow_change" in bd
    md = render_planner_table({"oracle": summ_oracle, "null": summ_null}, ["oracle", "null"])
    assert "| oracle |" in md and "| null |" in md


def test_bfs_solvability_labeling():
    tasks = _sample_tasks()
    lab_sol = bfs_label(tasks[0]["start_state"], tasks[0]["goal"], RULES, max_steps=5)
    assert lab_sol["solvable"] is True and lab_sol["shortest_steps"] == 2
    lab_unsol = bfs_label(tasks[1]["start_state"], tasks[1]["goal"], RULES, max_steps=5)
    assert lab_unsol["solvable"] is False and lab_unsol["shortest_steps"] is None
