"""Environment execution adapter and diagnostic predictor baselines (``oracle`` and ``null``).

Note: ``oracle_predict_batch`` supplies ground-truth ``(changes, cost, reversible)`` under the
same 1-step lookahead planner, candidate templates, and stopping rule; it is a diagnostic
control rather than a theoretical upper bound over multi-step planners.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from minibooking_rlvr.env import (
    Action,
    Rules,
    State,
    compute_cost,
    extract_changes,
    is_reversible,
    step_raw,
)


def to_state(d: Dict[str, Any]) -> State:
    return State.from_dict(d)


def to_rules(d: Dict[str, Any]) -> Rules:
    return Rules.from_dict(d)


def to_action(d: Dict[str, Any]) -> Action:
    return Action.from_dict(d)


def true_transition(state: Dict[str, Any], action: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    s, r = to_state(state), to_rules(rules)
    res = step_raw(s, to_action(action), r)
    after = res.state
    changes = extract_changes(s, after)
    return {
        "state": after.to_dict(),
        "legal": res.legal,
        "reason": res.reason,
        "changes": changes,
        "changed": len(changes) > 0,
        "cost_true": compute_cost(s, after, r),
        "reversible_true": is_reversible(s, after, r),
    }


def env_step(state: Dict[str, Any], action: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one real environment step and return the resulting state plus ground-truth risk logs."""
    out = true_transition(state, action, rules)
    out.pop("changes")
    return out


def oracle_predict_batch(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ground-truth predictor control: true changes/cost/reversibility with ``confidence=1.0``."""
    out: List[Dict[str, Any]] = []
    for item in items:
        t = true_transition(item["state_before"], item["action"], item["rules"])
        out.append(
            {
                "valid": True,
                "changes": t["changes"],
                "cost": t["cost_true"],
                "reversible": t["reversible_true"],
                "confidence": 1.0,
            }
        )
    return out


def null_predict_batch(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Zero-delta baseline predictor: always predicts no state change, zero cost, reversible, conf=1.0."""
    return [
        {"valid": True, "changes": [], "cost": 0.0, "reversible": True, "confidence": 1.0}
        for _ in items
    ]


def from_inference_output(gen: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a single ``run_inference`` record into the planner prediction contract."""
    parsed = gen.get("parsed") if gen.get("valid") else None
    pred: Dict[str, Any] = {
        "valid": bool(gen.get("valid")) and parsed is not None,
        "changes": parsed.get("changes") if parsed else None,
        "cost": parsed.get("cost") if parsed else None,
        "reversible": parsed.get("reversible") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
    }
    for key in ("raw", "error", "prompt_tokens", "n_new_tokens", "truncated", "hit_eos"):
        pred[key] = gen.get(key)
    return pred
