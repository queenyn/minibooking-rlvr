"""Unit tests for bounded-depth BFS reversibility search."""

import pytest

from minibooking_rlvr.env import Action, State, candidate_actions, is_reversible, label_transition, step_raw
from conftest import make_rules


def rev(before, action, rules, **kw):
    after = step_raw(before, action, rules).state
    return is_reversible(before, after, rules, **kw)


def test_candidate_actions_cover_six_types_and_three_days(half_rules):
    acts = candidate_actions(half_rules)
    assert len(acts) == 10
    assert {a.name for a in acts} == {"create", "confirm", "cancel", "change_day", "clear", "inspect"}


@pytest.mark.parametrize(
    "before,action",
    [
        (State("confirmed", 1, 900, 100, 0), Action("inspect")),
        (State("empty", None, 900, 0, 0), Action("confirm")),
        (State("draft", 1, 900, 0, 0), Action("change_day", 1)),
    ],
)
def test_no_change_is_reversible_at_depth_zero(before, action, half_rules):
    stats = {}
    assert rev(before, action, half_rules, stats=stats) is True
    assert stats["nodes"] == 1


@pytest.mark.parametrize(
    "before,action,rules_kw,expected",
    [
        (State("empty", None, 900, 0, 0), Action("create", 2), dict(), True),
        (State("draft", 2, 900, 0, 0), Action("clear"), dict(), True),
        (State("draft", 1, 900, 0, 0), Action("change_day", 3), dict(), True),
        (State("draft", 1, 900, 0, 0), Action("confirm"), dict(refund=(1, 1)), True),
        (State("draft", 1, 900, 0, 0), Action("confirm"), dict(refund=(1, 2)), False),
        (State("confirmed", 1, 900, 100, 0), Action("cancel"), dict(refund=(1, 1)), True),
        (State("confirmed", 1, 900, 100, 0), Action("cancel"), dict(refund=(1, 2)), False),
        (State("confirmed", 1, 900, 100, 0), Action("change_day", 2), dict(fee=0), True),
        (State("confirmed", 1, 900, 100, 0), Action("change_day", 2), dict(fee=20), False),
    ],
)
def test_reversibility_labels(before, action, rules_kw, expected):
    assert rev(before, action, make_rules(**rules_kw)) is expected


def test_depth_limit_matters():
    rules = make_rules(refund=(1, 1))
    before = State("confirmed", 1, 900, 100, 0)
    after = State("empty", None, 1000, 0, 0)
    assert is_reversible(before, after, rules, max_depth=1) is False
    assert is_reversible(before, after, rules, max_depth=2) is True


def test_search_is_bounded(half_rules):
    stats = {}
    rev(State("draft", 1, 900, 0, 0), Action("confirm"), half_rules, stats=stats)
    assert 1 < stats["nodes"] < 200


def test_full_label_matches_spec(half_rules):
    rec = label_transition(State("confirmed", 1, 900, 100, 0), Action("cancel"), half_rules)
    assert rec["gold"]["reversible"] is False and rec["gold"]["cost"] == 0.5
    assert rec["metadata"]["legal"] is True
