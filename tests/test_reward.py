"""Unit tests for strict JSON schema parsing and composite rewards (G1 / G2 / G3)."""

import copy
import json

import pytest

from minibooking_rlvr.rewards import (
    PredictionError,
    brier_score,
    canonical_changes,
    compute_reward,
    delta_scores,
    parse_prediction,
    risk_score,
)

GOLD = {
    "changes": [
        {"field": "balance_cents", "value": 950},
        {"field": "loss_cents", "value": 50},
        {"field": "paid_cents", "value": 0},
        {"field": "status", "value": "draft"},
    ],
    "reversible": False,
    "cost": 0.5,
}


def pred(confidence=0.8, **override):
    p = {**copy.deepcopy(GOLD), "confidence": confidence}
    p.update(override)
    return p


def raw(p):
    return json.dumps(p, ensure_ascii=False)


def test_composite_reward_correct_example():
    r = compute_reward(raw(pred()), GOLD, "G2")
    assert r.valid and r.f1 == 1.0 and r.exact == 1
    assert r.risk == pytest.approx(1.0) and r.brier == pytest.approx(0.04)
    assert r.reward == pytest.approx(0.994)


def test_composite_reward_one_wrong_field():
    p = pred()
    p["changes"][-1]["value"] = "empty"
    r = compute_reward(raw(p), GOLD, "G2")
    assert r.f1 == pytest.approx(0.75) and r.exact == 0
    assert r.brier == pytest.approx(0.64) and r.reward == pytest.approx(0.754)
    assert r.reward < compute_reward(raw(pred()), GOLD, "G2").reward


def test_missing_one_change():
    f1, exact = delta_scores(GOLD["changes"][:3], GOLD["changes"])
    assert f1 == pytest.approx(6 / 7) and exact == 0


def test_extra_irrelevant_change():
    extra = GOLD["changes"] + [{"field": "day", "value": 2}]
    f1, exact = delta_scores(extra, GOLD["changes"])
    assert f1 == pytest.approx(8 / 9) and exact == 0


def test_wrong_value():
    wrong = copy.deepcopy(GOLD["changes"])
    wrong[0]["value"] = 900
    assert delta_scores(wrong, GOLD["changes"]) == (pytest.approx(0.75), 0)


def test_order_invariance():
    assert delta_scores(list(reversed(GOLD["changes"])), GOLD["changes"]) == (1.0, 1)
    p = {"confidence": 0.8, "cost": 0.5, "reversible": False, "changes": list(reversed(GOLD["changes"]))}
    assert compute_reward(raw(p), GOLD).exact == 1


def test_empty_vs_empty_and_empty_vs_nonempty():
    assert delta_scores([], []) == (1.0, 1)
    assert delta_scores([], GOLD["changes"]) == (0.0, 0)
    assert delta_scores(GOLD["changes"], []) == (0.0, 0)


def test_false_and_zero_are_different_values():
    a = canonical_changes([{"field": "day", "value": False}])
    b = canonical_changes([{"field": "day", "value": 0}])
    assert a != b
    assert delta_scores([{"field": "day", "value": False}], [{"field": "day", "value": 0}]) == (0.0, 0)


def test_none_day_is_a_real_value():
    assert delta_scores([{"field": "day", "value": None}], [{"field": "day", "value": None}]) == (1.0, 1)


def test_risk_and_brier_components():
    assert risk_score(False, False, 0.5, 0.5) == pytest.approx(1.0)
    assert risk_score(True, False, 0.5, 0.5) == pytest.approx(0.5)
    assert risk_score(False, False, 0.0, 0.5) == pytest.approx(0.75)
    assert brier_score(0.8, 1) == pytest.approx(0.04)
    assert brier_score(0.8, 0) == pytest.approx(0.64)


def test_correct_but_low_confidence():
    r = compute_reward(raw(pred(confidence=0.1)), GOLD)
    assert r.exact == 1 and r.brier == pytest.approx(0.81)
    assert r.reward == pytest.approx(0.6 + 0.25 + 0.15 * 0.19)


def test_wrong_but_high_confidence():
    p = pred(confidence=1.0)
    p["changes"] = p["changes"][:2]
    r = compute_reward(raw(p), GOLD)
    assert r.exact == 0 and r.brier == pytest.approx(1.0)
    assert r.reward == pytest.approx(0.6 * (4 / 6) + 0.25)


def test_g1_uses_only_delta_f1():
    p = pred(confidence=0.0, reversible=True, cost=1.0)
    p["changes"] = p["changes"][:3]
    r = compute_reward(raw(p), GOLD, "G1")
    assert r.reward == pytest.approx(6 / 7)


def test_g2_equals_g3():
    assert compute_reward(raw(pred()), GOLD, "G2") == compute_reward(raw(pred()), GOLD, "G3")


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        compute_reward(raw(pred()), GOLD, "G4")


def _dup_field():
    p = pred()
    p["changes"].append(dict(p["changes"][0]))
    return raw(p)


def _with_change(field, value):
    p = pred()
    p["changes"] = [{"field": field, "value": value}]
    return raw(p)


INVALID_RAWS = {
    "not_json": "not json",
    "top_level_list": "[]",
    "markdown_fence": "```json\n" + raw(pred()) + "\n```",
    "duplicate_field": _dup_field(),
    "duplicate_json_key": '{"changes":[],"changes":[],"reversible":true,"cost":0,"confidence":1}',
    "missing_confidence": raw({k: v for k, v in pred().items() if k != "confidence"}),
    "extra_top_key": raw({**pred(), "legal": True}),
    "cost_gt_1": raw(pred(cost=1.5)),
    "cost_lt_0": raw(pred(cost=-0.1)),
    "confidence_gt_1": raw(pred(confidence=7)),
    "nan_confidence": raw(pred(confidence=float("nan"))),
    "infinity_cost": '{"changes":[],"reversible":true,"cost":Infinity,"confidence":1}',
    "string_false": raw(pred(reversible="false")),
    "int_zero_reversible": raw(pred(reversible=0)),
    "bool_confidence": raw(pred(confidence=True)),
    "bool_balance": _with_change("balance_cents", False),
    "float_balance": _with_change("balance_cents", 950.0),
    "negative_loss": _with_change("loss_cents", -5),
    "unknown_field": _with_change("price_cents", 100),
    "unknown_status": _with_change("status", "paid"),
    "day_out_of_range": _with_change("day", 4),
    "day_as_string": _with_change("day", "1"),
    "changes_not_list": raw(pred(changes={})),
    "change_missing_value": raw(pred(changes=[{"field": "status"}])),
}


@pytest.mark.parametrize("name", list(INVALID_RAWS))
@pytest.mark.parametrize("group", ["G1", "G2"])
def test_invalid_outputs_get_minus_one(name, group):
    r = compute_reward(INVALID_RAWS[name], GOLD, group)
    assert not r.valid and r.reward == -1.0, name


@pytest.mark.parametrize("name", list(INVALID_RAWS))
def test_parser_raises_prediction_error(name):
    with pytest.raises(PredictionError):
        parse_prediction(INVALID_RAWS[name])


def test_valid_day_null_accepted():
    parse_prediction(_with_change("day", None))


def test_integer_cost_and_confidence_accepted():
    parse_prediction(raw(pred(cost=0, confidence=1)))


def test_bad_tendencies_score_lower_than_truth():
    truth = compute_reward(raw(pred(confidence=1.0)), GOLD).reward
    always_empty = compute_reward(raw(pred(changes=[], confidence=1.0)), GOLD).reward
    always_reversible = compute_reward(raw(pred(reversible=True, confidence=1.0)), GOLD).reward
    assert always_empty < truth - 0.5
    assert always_reversible < truth


def test_reward_is_pure_and_does_not_mutate_gold():
    g = copy.deepcopy(GOLD)
    a = compute_reward(raw(pred()), g)
    b = compute_reward(raw(pred()), g)
    assert a == b and g == GOLD
