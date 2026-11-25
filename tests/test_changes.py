"""Unit tests for delta extraction, immediate cost, and transition labeling."""

import pytest

from minibooking_rlvr.env import Action, State, apply_changes, compute_cost, extract_changes, label_transition, step_raw
from conftest import all_cases, make_rules

CANONICAL_CANCEL_CHANGES = [
    {"field": "balance_cents", "value": 950},
    {"field": "loss_cents", "value": 50},
    {"field": "paid_cents", "value": 0},
    {"field": "status", "value": "draft"},
]


def test_cancel_changes_canonical(half_rules):
    before = State("confirmed", 1, 900, 100, 0)
    after = step_raw(before, Action("cancel"), half_rules).state
    assert extract_changes(before, after) == CANONICAL_CANCEL_CHANGES


def test_no_change_gives_empty_list():
    s = State("draft", 1, 900, 0, 0)
    assert extract_changes(s, s.copy()) == []


def test_changes_sorted_and_only_changed_fields():
    before = State("confirmed", 1, 900, 100, 0)
    after = State("confirmed", 2, 880, 100, 20)
    out = extract_changes(before, after)
    assert out == [
        {"field": "balance_cents", "value": 880},
        {"field": "day", "value": 2},
        {"field": "loss_cents", "value": 20},
    ]
    assert [c["field"] for c in out] == sorted(c["field"] for c in out)


def test_clear_writes_day_none():
    before = State("draft", 2, 900, 0, 0)
    after = State("empty", None, 900, 0, 0)
    assert extract_changes(before, after) == [
        {"field": "day", "value": None},
        {"field": "status", "value": "empty"},
    ]


def test_extract_does_not_mutate_inputs():
    before = State("confirmed", 1, 900, 100, 0)
    after = State("draft", 1, 950, 0, 50)
    b0, a0 = before.copy(), after.copy()
    extract_changes(before, after)
    assert before == b0 and after == a0


@pytest.mark.parametrize(
    "rules_kw,before,action,expected",
    [
        (dict(), State("confirmed", 1, 900, 100, 0), Action("cancel"), 0.5),
        (dict(refund=(0, 1)), State("confirmed", 1, 900, 100, 0), Action("cancel"), 1.0),
        (dict(refund=(1, 1)), State("confirmed", 1, 900, 100, 0), Action("cancel"), 0.0),
        (dict(), State("confirmed", 1, 900, 100, 0), Action("change_day", 2), 0.2),
        (dict(price=200), State("confirmed", 1, 900, 200, 0), Action("change_day", 2), 0.1),
        (dict(), State("draft", 1, 900, 0, 0), Action("confirm"), 0.0),
        (dict(), State("draft", 1, 900, 0, 0), Action("inspect"), 0.0),
    ],
)
def test_cost(rules_kw, before, action, expected):
    rules = make_rules(**rules_kw)
    after = step_raw(before, action, rules).state
    assert compute_cost(before, after, rules) == pytest.approx(expected)


def test_cost_in_unit_range_and_roundtrip():
    for state, action, rules in all_cases():
        after = step_raw(state, action, rules).state
        cost = compute_cost(state, after, rules)
        assert 0.0 <= cost <= 1.0 and isinstance(cost, float)
        assert apply_changes(state, extract_changes(state, after)) == after


def test_inspect_and_illegal_both_empty_but_distinguishable(half_rules):
    s = State("empty", None, 900, 0, 0)
    inspect = label_transition(s, Action("inspect"), half_rules, with_reversible=False)
    illegal = label_transition(s, Action("confirm"), half_rules, with_reversible=False)
    assert inspect["gold"]["changes"] == [] and illegal["gold"]["changes"] == []
    assert inspect["metadata"]["legal"] is True
    assert illegal["metadata"]["legal"] is False
    assert inspect["metadata"]["action_type"] != illegal["metadata"]["action_type"]
