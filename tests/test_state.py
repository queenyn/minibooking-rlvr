"""Unit tests for State, Rules, and Action schema invariants."""

import pytest

from minibooking_rlvr.env import STATE_FIELDS, Action, State
from conftest import make_rules


def test_state_fields_sorted():
    assert list(STATE_FIELDS) == sorted(STATE_FIELDS)


@pytest.mark.parametrize(
    "bad",
    [
        State("unknown", None, 0, 0, 0),
        State("empty", 1, 0, 0, 0),
        State("draft", None, 0, 0, 0),
        State("draft", 4, 0, 0, 0),
        State("draft", 1, -1, 0, 0),
        State("draft", 1, 100, 100, 0),
        State("draft", 1, True, 0, 0),
        State("confirmed", 1, 1.5, 100, 0),
    ],
)
def test_invalid_state_rejected(bad):
    with pytest.raises(ValueError):
        bad.validate()


def test_valid_states():
    State("empty", None, 0, 0, 0).validate()
    State("draft", 2, 100, 0, 30).validate()
    State("confirmed", 3, 0, 200, 0).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(price=150),
        dict(refund=(1, 3)),
        dict(fee=10),
        dict(allow_change=1),
    ],
)
def test_invalid_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        make_rules(**kwargs).validate()


@pytest.mark.parametrize("bad", [Action("pay"), Action("create"), Action("confirm", 1), Action("change_day", True)])
def test_malformed_action_rejected(bad):
    with pytest.raises(ValueError):
        bad.validate_shape()


def test_key_and_roundtrip():
    s = State("confirmed", 1, 900, 100, 0)
    assert s.key() == (900, 1, 0, 100, "confirmed")
    assert State.from_dict(s.to_dict()) == s
    assert s.copy() == s and s.copy() is not s
