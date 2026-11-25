"""Unit tests for deterministic MiniBooking transitions and conservation invariants."""

import copy

import pytest

from minibooking_rlvr.env import (
    REASON_CHANGE_NOT_ALLOWED,
    REASON_INSUFFICIENT_BALANCE,
    REASON_INVALID_DAY,
    REASON_WRONG_STATUS,
    Action,
    State,
    step_raw,
)
from conftest import all_cases, make_rules


def run(state, action, rules):
    return step_raw(state, action, rules)


def test_create_from_empty(half_rules):
    r = run(State("empty", None, 500, 0, 0), Action("create", 2), half_rules)
    assert r.legal and r.state == State("draft", 2, 500, 0, 0)


def test_create_invalid_day_is_illegal(half_rules):
    s = State("empty", None, 500, 0, 0)
    r = run(s, Action("create", 4), half_rules)
    assert not r.legal and r.reason == REASON_INVALID_DAY and r.state == s


@pytest.mark.parametrize("s", [State("draft", 1, 500, 0, 0), State("confirmed", 1, 400, 100, 0)])
def test_create_when_not_empty_is_illegal(s, half_rules):
    r = run(s, Action("create", 2), half_rules)
    assert not r.legal and r.reason == REASON_WRONG_STATUS and r.state == s


def test_confirm_ok(half_rules):
    r = run(State("draft", 1, 1000, 0, 0), Action("confirm"), half_rules)
    assert r.legal and r.state == State("confirmed", 1, 900, 100, 0)


def test_confirm_exact_balance(half_rules):
    r = run(State("draft", 1, 100, 0, 0), Action("confirm"), half_rules)
    assert r.legal and r.state.balance_cents == 0 and r.state.paid_cents == 100


def test_confirm_insufficient_balance(half_rules):
    s = State("draft", 1, 99, 0, 0)
    r = run(s, Action("confirm"), half_rules)
    assert not r.legal and r.reason == REASON_INSUFFICIENT_BALANCE
    assert r.state == s and r.state.balance_cents >= 0


def test_confirm_price_200():
    r = run(State("draft", 3, 250, 0, 0), Action("confirm"), make_rules(price=200))
    assert r.state == State("confirmed", 3, 50, 200, 0)


@pytest.mark.parametrize("s", [State("empty", None, 500, 0, 0), State("confirmed", 1, 400, 100, 0)])
def test_confirm_wrong_status(s, half_rules):
    r = run(s, Action("confirm"), half_rules)
    assert not r.legal and r.state == s


def test_cancel_half_refund(half_rules):
    r = run(State("confirmed", 1, 900, 100, 0), Action("cancel"), half_rules)
    assert r.legal and r.state == State("draft", 1, 950, 0, 50)


@pytest.mark.parametrize("refund,balance,loss", [((1, 1), 1000, 0), ((0, 1), 900, 100), ((1, 2), 950, 50)])
def test_cancel_refund_ratios(refund, balance, loss):
    r = run(State("confirmed", 2, 900, 100, 0), Action("cancel"), make_rules(refund=refund))
    assert r.state == State("draft", 2, balance, 0, loss)


def test_cancel_keeps_existing_loss():
    r = run(State("confirmed", 1, 0, 200, 30), Action("cancel"), make_rules(price=200))
    assert r.state == State("draft", 1, 100, 0, 130)


@pytest.mark.parametrize("s", [State("empty", None, 500, 0, 0), State("draft", 1, 500, 0, 0)])
def test_cancel_wrong_status(s, half_rules):
    r = run(s, Action("cancel"), half_rules)
    assert not r.legal and r.reason == REASON_WRONG_STATUS and r.state == s


def test_duplicate_cancel_cannot_increase_balance(half_rules):
    first = run(State("confirmed", 1, 900, 100, 0), Action("cancel"), half_rules)
    second = run(first.state, Action("cancel"), half_rules)
    assert first.legal and not second.legal
    assert second.state.balance_cents == first.state.balance_cents
    assert second.state.loss_cents == first.state.loss_cents


@pytest.mark.parametrize("refund", [(0, 1), (1, 2), (1, 1)])
def test_repeated_cancel_confirm_conserves_money(refund):
    rules = make_rules(refund=refund)
    s = State("draft", 1, 500, 0, 0)
    for _ in range(5):
        for a in (Action("confirm"), Action("cancel"), Action("cancel")):
            s = run(s, a, rules).state
            assert s.balance_cents + s.paid_cents <= 500
            assert s.balance_cents + s.paid_cents + s.loss_cents == 500


def test_change_day_draft_is_free(half_rules):
    r = run(State("draft", 1, 900, 0, 0), Action("change_day", 3), half_rules)
    assert r.legal and r.state == State("draft", 3, 900, 0, 0)


def test_change_day_confirmed_charges_fee(half_rules):
    r = run(State("confirmed", 1, 900, 100, 0), Action("change_day", 2), half_rules)
    assert r.legal and r.state == State("confirmed", 2, 880, 100, 20)


def test_change_day_confirmed_zero_fee():
    r = run(State("confirmed", 1, 900, 100, 0), Action("change_day", 2), make_rules(fee=0))
    assert r.legal and r.state == State("confirmed", 2, 900, 100, 0)


@pytest.mark.parametrize("s", [State("draft", 2, 900, 0, 0), State("confirmed", 2, 900, 100, 0)])
def test_change_day_same_day_is_legal_noop(s, half_rules):
    r = run(s, Action("change_day", 2), half_rules)
    assert r.legal and r.state == s


def test_change_day_same_day_free_even_if_balance_low(half_rules):
    s = State("confirmed", 2, 0, 100, 0)
    r = run(s, Action("change_day", 2), half_rules)
    assert r.legal and r.state == s


def test_change_day_insufficient_balance_for_fee(half_rules):
    s = State("confirmed", 1, 19, 100, 0)
    r = run(s, Action("change_day", 2), half_rules)
    assert not r.legal and r.reason == REASON_INSUFFICIENT_BALANCE and r.state == s


def test_change_day_exact_fee_balance(half_rules):
    r = run(State("confirmed", 1, 20, 100, 0), Action("change_day", 3), half_rules)
    assert r.legal and r.state == State("confirmed", 3, 0, 100, 20)


def test_change_day_not_allowed():
    s = State("draft", 1, 900, 0, 0)
    r = run(s, Action("change_day", 2), make_rules(allow_change=False))
    assert not r.legal and r.reason == REASON_CHANGE_NOT_ALLOWED and r.state == s


def test_change_day_not_allowed_same_day_is_illegal():
    s = State("draft", 1, 900, 0, 0)
    r = run(s, Action("change_day", 1), make_rules(allow_change=False))
    assert not r.legal and r.state == s


def test_change_day_invalid_day(half_rules):
    s = State("draft", 1, 900, 0, 0)
    r = run(s, Action("change_day", 0), half_rules)
    assert not r.legal and r.reason == REASON_INVALID_DAY and r.state == s


def test_change_day_from_empty(half_rules):
    s = State("empty", None, 900, 0, 0)
    r = run(s, Action("change_day", 1), half_rules)
    assert not r.legal and r.state == s


def test_clear_draft(half_rules):
    r = run(State("draft", 2, 900, 0, 10), Action("clear"), half_rules)
    assert r.legal and r.state == State("empty", None, 900, 0, 10)


@pytest.mark.parametrize("s", [State("empty", None, 900, 0, 0), State("confirmed", 1, 800, 100, 0)])
def test_clear_wrong_status(s, half_rules):
    r = run(s, Action("clear"), half_rules)
    assert not r.legal and r.state == s


@pytest.mark.parametrize(
    "s",
    [State("empty", None, 0, 0, 0), State("draft", 1, 5, 0, 0), State("confirmed", 3, 0, 200, 70)],
)
def test_inspect_always_legal_no_change(s, half_rules):
    r = run(s, Action("inspect"), half_rules)
    assert r.legal and r.state == s


def test_determinism_two_identical_calls():
    for state, action, rules in all_cases():
        assert run(state, action, rules) == run(state, action, rules)


def test_input_state_not_mutated():
    for state, action, rules in all_cases():
        snapshot = copy.deepcopy(state)
        result = run(state, action, rules)
        assert state == snapshot
        assert result.state is not state


def test_global_invariants():
    for state, action, rules in all_cases():
        r = run(state, action, rules)
        s = r.state
        assert s.balance_cents >= 0
        assert s.loss_cents >= state.loss_cents
        assert (s.status == "confirmed") or s.paid_cents == 0
        assert s.balance_cents + s.paid_cents + s.loss_cents == state.balance_cents + state.paid_cents + state.loss_cents
        if not r.legal:
            assert s == state
