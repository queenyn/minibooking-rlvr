"""Unit tests for online H/D/S/C group/batch diagnostics, zero-var splitting, and offline filtering."""

from __future__ import annotations

import pytest

from minibooking_rlvr.rewards import INVALID_REWARD, compute_reward
from minibooking_rlvr.rl import (
    TASK_TYPES,
    batch_stats,
    classify_zero_var_group,
    filter_all_correct_prompts,
    group_stats,
    make_sample_row,
    summarize_zero_var_groups,
)

GOLD_T = {
    "changes": [
        {"field": "loss_cents", "value": 200},
        {"field": "paid_cents", "value": 0},
        {"field": "status", "value": "draft"},
    ],
    "reversible": False,
    "cost": 1.0,
}
OK1 = '{"changes":[{"field":"loss_cents","value":200},{"field":"paid_cents","value":0},{"field":"status","value":"draft"}],"reversible":false,"cost":1.0,"confidence":1.0}'
OK2 = '{"confidence": 0.8, "cost": 1.0, "reversible": false, "changes": [{"value": "draft", "field": "status"}, {"field": "paid_cents", "value": 0}, {"field": "loss_cents", "value": 200}]}'
OK3 = '{"changes":[{"field":"status","value":"draft"},{"field":"loss_cents","value":200},{"field":"paid_cents","value":0}],"reversible":true,"cost":0.0,"confidence":1.0}'
W_A = '{"changes":[{"field":"balance_cents","value":300},{"field":"loss_cents","value":200},{"field":"paid_cents","value":0},{"field":"status","value":"draft"}],"reversible":false,"cost":1.0,"confidence":1.0}'
W_B = '{"changes":[{"field":"status","value":"confirmed"}],"reversible":true,"cost":0.0,"confidence":1.0}'
W_C = '{"changes":[],"reversible":true,"cost":0.0,"confidence":0.9}'
BAD = ["not json", "```json\n" + OK1 + "\n```", OK1[:60], OK1.replace('"confidence":1.0', '"confidence":7')]


def _g(raws):
    return group_stats([make_sample_row(r, GOLD_T, "G1", "G2") for r in raws])


def test_group_diagnostics_all_cases():
    s_ac = _g([OK1, OK1, OK2, OK3])
    assert s_ac["task_type"] == "all_correct" and s_ac["zero_var"] is True
    assert s_ac["D"] == 0.25 and s_ac["C"] == 1.0 and s_ac["mode_correct"] is True
    # Composite reward (G2) distinguishes differing risk/confidence even when all DeltaF1 == 1
    assert s_ac["zero_var_alt"] is False

    s_aw_same = _g([W_A, W_A, W_A, W_A])
    assert s_aw_same["task_type"] == "all_wrong" and s_aw_same["zero_var"] is True
    assert s_aw_same["D"] == 0.25 and s_aw_same["C"] == 1.0 and s_aw_same["mode_correct"] is False

    s_aw_div = _g([W_A, W_A, W_B, W_C])
    assert s_aw_div["task_type"] == "all_wrong" and s_aw_div["zero_var"] is False and s_aw_div["n_exact"] == 0

    s_mix = _g([OK1, OK2, W_A, W_B])
    assert s_mix["task_type"] == "mixed" and s_mix["zero_var"] is False
    assert s_mix["D"] == 0.75 and s_mix["C"] == 0.5 and s_mix["mode_correct"] is True and s_mix["mode_tie"] is False

    s_bad = _g(BAD)
    assert s_bad["task_type"] == "parse_fail" and s_bad["n_valid"] == 0
    assert s_bad["D"] is None and s_bad["C"] is None and s_bad["mode_correct"] is None
    assert s_bad["zero_var"] is True and s_bad["reward_mean"] == INVALID_REWARD

    s_part = _g([BAD[0], BAD[1], OK1, OK2])
    assert s_part["task_type"] == "mixed" and s_part["valid_parse_rate"] == 0.5
    assert s_part["D"] == 0.5 and s_part["C"] == 1.0


def test_batch_stats_aggregation():
    gs = [_g([OK1, OK1, OK2, OK3]), _g([W_A, W_A, W_A, W_A]), _g([OK1, OK2, W_A, W_B]), _g(BAD)]
    b = batch_stats(gs)
    assert b["S_reward"] == 0.75 and b["n_effective"] == 1
    assert b["zero_var_all_correct"] == 1 and b["zero_var_all_wrong"] == 1
    assert b["zero_var_parse_fail"] == 1 and b["zero_var_mixed"] == 0
    assert all(b[f"S_task_{t}"] == 0.25 for t in TASK_TYPES)
    assert sum(b[f"S_task_{t}"] for t in TASK_TYPES) == pytest.approx(1.0)
    assert b["dc_coverage"] == 0.75
    assert b["D_mean"] == pytest.approx((0.25 + 0.25 + 0.75) / 3)
    assert b["C_mean"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert b["mode_correct_rate"] == pytest.approx(2 / 3)


def test_g2_and_g3_rewards_identical_across_all_samples():
    for raw in [OK1, OK2, OK3, W_A, W_B, W_C, *BAD]:
        assert compute_reward(raw, GOLD_T, "G2").reward == compute_reward(raw, GOLD_T, "G3").reward


def test_zero_var_classification_and_summary():
    assert classify_zero_var_group([1.0, 1.0, 1.0, 1.0]) == "all_correct"
    assert classify_zero_var_group([0.0, 0.0, 0.0, 0.0]) == "all_wrong"
    assert classify_zero_var_group([0.75, 0.75, 0.75, 0.75]) == "other_same"
    assert classify_zero_var_group([1.0, 0.75, 0.0, 0.0]) == "effective"

    summary = summarize_zero_var_groups(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]]
    )
    assert summary["n_groups"] == 4 and summary["n_effective"] == 1 and summary["n_zero_var"] == 3
    assert summary["n_all_correct"] == 1 and summary["n_all_wrong"] == 1 and summary["n_other_same"] == 1


def test_offline_filter_removes_only_all_correct():
    recs = [
        {"sample_id": "s1", "gold": GOLD_T, "metadata": {"action_type": "inspect"}},
        {"sample_id": "s2", "gold": GOLD_T, "metadata": {"action_type": "cancel"}},
        {"sample_id": "s3", "gold": GOLD_T, "metadata": {"action_type": "cancel"}},
    ]
    samples = {
        "s1": [OK1, OK1, OK2, OK3],  # all_correct -> removed
        "s2": [W_A, W_A, W_A, W_A],  # all_wrong -> kept
        "s3": [OK1, W_A, W_B, BAD[0]],  # mixed -> kept
    }
    kept, report = filter_all_correct_prompts(recs, samples, reward_group="G1")
    assert [r["sample_id"] for r in kept] == ["s2", "s3"]
    assert report["pool_before"] == 3 and report["pool_after"] == 2 and report["removed_all_correct"] == 1
    assert report["action_type_after"] == {"cancel": 2}
