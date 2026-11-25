"""Shared pytest fixtures and exhaustive domain grids."""

from __future__ import annotations

import itertools
from typing import Iterator, List

import pytest

from minibooking_rlvr.env import (
    ALLOWED_DAYS,
    CHANGE_FEE_CHOICES,
    PRICE_CHOICES,
    REFUND_CHOICES,
    Action,
    Rules,
    State,
)


def make_rules(price: int = 100, refund=(1, 2), allow_change: bool = True, fee: int = 20) -> Rules:
    return Rules(
        price_cents=price,
        refund_num=refund[0],
        refund_den=refund[1],
        allow_change=allow_change,
        change_fee_cents=fee,
    )


def all_rules() -> List[Rules]:
    return [
        make_rules(p, r, a, f)
        for p, r, a, f in itertools.product(PRICE_CHOICES, REFUND_CHOICES, (True, False), CHANGE_FEE_CHOICES)
    ]


def all_actions() -> List[Action]:
    acts = [Action("create", d) for d in (*ALLOWED_DAYS, 4)]
    acts += [Action("change_day", d) for d in (*ALLOWED_DAYS, 0)]
    acts += [Action(n) for n in ("confirm", "cancel", "clear", "inspect")]
    return acts


def sample_states(rules: Rules) -> List[State]:
    p = rules.price_cents
    states = [State("empty", None, b, 0, loss) for b in (0, p, 1000) for loss in (0, 50)]
    states += [State("draft", d, b, 0, loss) for d in ALLOWED_DAYS for b in (0, p - 1, p, 1000) for loss in (0, 50)]
    states += [State("confirmed", d, b, p, loss) for d in ALLOWED_DAYS for b in (0, 10, 20, 900) for loss in (0, 50)]
    return states


def all_cases() -> Iterator:
    for rules in all_rules():
        for state in sample_states(rules):
            for action in all_actions():
                yield state, action, rules


@pytest.fixture
def half_rules() -> Rules:
    """Default rules: price=100, refund=1/2, allow_change=True, fee=20."""
    return make_rules()
