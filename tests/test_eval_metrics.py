"""Unit tests for dual-view frozen evaluation metrics and compositional subset slicing."""

from __future__ import annotations

import json

import pytest

from minibooking_rlvr.env import Action, State, label_transition
from minibooking_rlvr.eval import (
    confidence_distribution,
    evaluate_model_outputs,
    macro_f1_reversible,
    render_eval_table,
    score_raw,
)
from conftest import make_rules


def _rec(sample_id, state, action, rules):
    r = label_transition(state, action, rules)
    r["sample_id"] = sample_id
    return r


def test_score_raw_primary_vs_valid_only_policy():
    rec = _rec("s0", State("confirmed", 1, 900, 100, 0), Action("cancel"), make_rules())
    good = json.dumps({**rec["gold"], "confidence": 1.0})
    s_ok = score_raw(rec, good)
    assert s_ok["valid"] is True and s_ok["metrics"]["exact"] == 1 and s_ok["metrics"]["full_exact"] == 1
    assert s_ok["metrics_valid_only"] == s_ok["metrics"]

    s_bad = score_raw(rec, "```json\n" + good + "\n```")
    assert s_bad["valid"] is False and s_bad["json_valid"] is False
    assert s_bad["metrics"]["exact"] == 0 and s_bad["metrics"]["cost_abs_err"] == 1.0
    assert s_bad["metrics"]["brier"] == 1.0 and s_bad["metrics"]["reversible_ok"] == 0
    assert s_bad["metrics"]["reversible_pred"] is True  # not gold(False)
    assert s_bad["metrics_valid_only"] is None


def test_macro_f1_reversible_and_confidence_collapse():
    res = macro_f1_reversible([True, True, False, False], [True, False, False, True])
    assert res["macro_f1"] == pytest.approx(0.5)
    cd = confidence_distribution([1.0, 1.0, 1.0])
    assert cd["frac_eq_1"] == 1.0 and cd["counts"][-1] == 3


def test_evaluate_model_outputs_comp_slices_and_table():
    r_true = _rec("c0", State("confirmed", 1, 900, 100, 0), Action("cancel"), make_rules(allow_change=True, fee=0))
    r_false = _rec("c1", State("empty", None, 500, 0, 0), Action("inspect"), make_rules(allow_change=False, fee=0))
    raw0 = json.dumps({**r_true["gold"], "confidence": 1.0})
    raw1 = "invalid output"
    res = evaluate_model_outputs([r_true, r_false], [raw0, raw1], split="test_comp")
    assert "allow_change=True" in res["subsets"]
    assert "allow_change=False" in res["subsets"]
    assert res["subsets"]["all"]["parse_coverage"] == 0.5
    assert res["subsets"]["all"]["main"]["exact"] == 0.5
    assert res["subsets"]["all"]["valid_only"]["exact"] == 1.0
    md = render_eval_table({"SFT": res}, "test_comp")
    assert "| SFT | 2 |" in md
