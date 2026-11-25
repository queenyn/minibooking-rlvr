"""Deterministic, goal-driven candidate action proposer (isolated from env internals).

Isolation invariant:
- Reads only ``(state, goal, rules)`` and action schemas.
- NEVER imports ``minibooking_rlvr.env`` transitions, reversibility search, or labels.
- Performs no legality filtering (illegal actions may be proposed and scored by the world model).
- Deterministic fixed ordering with at most ``MAX_CANDIDATES = 4`` actions per step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

TEMPLATES = {
    "confirmed": ("create", "confirm", "change_day", "cancel"),
    "draft": ("create", "change_day", "cancel", "clear"),
    "empty": ("cancel", "clear", "change_day", "create"),
    None: ("change_day", "cancel", "confirm", "create"),
}
ACTION_NAMES = ("create", "confirm", "cancel", "change_day", "clear", "inspect")
ACTIONS_WITH_DAY = ("create", "change_day")
MAX_CANDIDATES = 4


def _goal_value(goal: Dict[str, Any], field: str) -> Optional[Any]:
    for cond in goal.get("conditions", []):
        if cond.get("field") == field and cond.get("op") == "eq":
            return cond.get("value")
    return None


def target_day(state: Dict[str, Any], goal: Dict[str, Any], rules: Dict[str, Any]) -> int:
    """Resolve target day: goal eq day -> current state day -> first allowed day."""
    days = list(rules.get("allowed_days", [1, 2, 3]))
    d = _goal_value(goal, "day")
    if d in days:
        return d
    if state.get("day") in days:
        return state["day"]
    return days[0]


def propose_candidates(
    state: Dict[str, Any],
    goal: Dict[str, Any],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return a deterministic list of at most 4 candidate actions for ``(state, goal, rules)``."""
    st = _goal_value(goal, "status")
    names = TEMPLATES.get(st, TEMPLATES[None])
    d = target_day(state, goal, rules)
    return [{"name": n, "day": d} if n in ACTIONS_WITH_DAY else {"name": n} for n in names]
