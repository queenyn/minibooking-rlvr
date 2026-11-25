"""Risk-aware 1-step lookahead MPC planner with real-state replanning.

Scoring and stopping rules:
- ``progress(state, goal)``: fraction of satisfied goal conditions in ``[0, 1]``; ``0.0`` if
  ``state['loss_cents'] > goal['loss_limit_cents']``.
- ``score(a) = progress(s_hat_a) - (w_risk * c_hat + w_irr * I(not r_hat) + w_conf * (1 - q_hat))``
  where invalid world-model outputs receive ``-inf``.
- ``should_stop`` priority: goal reached -> max steps -> no valid candidate ->
  ``max_a score(a) - progress(s) <= tau``.
- Replanning: after executing the winning candidate in the true environment, the next step
  always starts from the true observed state (preventing imagined-state drift).
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

STATE_FIELDS: Tuple[str, ...] = ("balance_cents", "day", "loss_cents", "paid_cents", "status")
GOAL_OPS: Tuple[str, ...] = ("eq", "ge", "le")

MAX_CANDIDATES = 4
MAX_STEPS = 5
BATCH_SIZE = 16

END_GOAL = "goal_reached"
END_THRESHOLD = "score_threshold"
END_NO_VALID = "no_valid_candidate"
END_MAX_STEPS = "max_steps"
END_REASONS = (END_GOAL, END_THRESHOLD, END_NO_VALID, END_MAX_STEPS)

ProposeFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]
PredictBatchFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
EnvStepFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def apply_state_changes(state: Dict[str, Any], changes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a new state dict with ``changes`` applied; never mutates ``state``."""
    new_state = dict(state)
    for item in changes:
        if item["field"] not in STATE_FIELDS:
            raise ValueError(f"Unknown state field: {item['field']!r}")
        new_state[item["field"]] = item["value"]
    return new_state


def _check_goal(goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    conds = goal.get("conditions")
    if not conds:
        raise ValueError("goal.conditions must be non-empty")
    for cond in conds:
        if cond.get("field") not in STATE_FIELDS:
            raise ValueError(f"Unknown goal field: {cond.get('field')!r}")
        if cond.get("op") not in GOAL_OPS:
            raise ValueError(f"Unknown goal operator: {cond.get('op')!r}")
    return conds


def condition_met(state: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    """Evaluate a single goal condition; ``ge``/``le`` against ``None`` evaluates to ``False``."""
    x, op, v = state[cond["field"]], cond["op"], cond["value"]
    if op == "eq":
        return x == v
    if x is None or v is None:
        return False
    return x >= v if op == "ge" else x <= v


def progress(state: Dict[str, Any], goal: Dict[str, Any]) -> float:
    """Heuristic fraction of satisfied conditions in ``[0, 1]``; ``0.0`` when loss exceeds limit."""
    conds = _check_goal(goal)
    limit = goal.get("loss_limit_cents")
    if limit is not None and state["loss_cents"] > limit:
        return 0.0
    return sum(condition_met(state, c) for c in conds) / len(conds)


def goal_met(state: Dict[str, Any], goal: Dict[str, Any]) -> bool:
    """Return True iff all conditions hold and the optional loss limit is respected."""
    return progress(state, goal) == 1.0


def score_components(
    prediction: Dict[str, Any],
    current_state: Dict[str, Any],
    goal: Dict[str, Any],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Compute decomposed candidate score from a world-model prediction (never calls true env)."""
    if not prediction.get("valid"):
        return {
            "valid": False,
            "predicted_state": None,
            "predicted_progress": None,
            "risk_penalty": None,
            "irreversible_penalty": None,
            "confidence_penalty": None,
            "score": -math.inf,
        }
    pred_state = apply_state_changes(current_state, prediction["changes"] or [])
    pg = progress(pred_state, goal)
    risk_pen = weights["w_risk"] * float(prediction["cost"])
    irr_pen = weights["w_irr"] * (0.0 if prediction["reversible"] else 1.0)
    conf_pen = weights["w_conf"] * (1.0 - float(prediction["confidence"]))
    return {
        "valid": True,
        "predicted_state": pred_state,
        "predicted_progress": pg,
        "risk_penalty": risk_pen,
        "irreversible_penalty": irr_pen,
        "confidence_penalty": conf_pen,
        "score": pg - (risk_pen + irr_pen + conf_pen),
    }


def score(
    prediction: Dict[str, Any],
    current_state: Dict[str, Any],
    goal: Dict[str, Any],
    weights: Dict[str, float],
) -> float:
    """Scalar score of a single candidate prediction (``-inf`` when invalid)."""
    return score_components(prediction, current_state, goal, weights)["score"]


def should_stop(
    best_score: float,
    current_progress: float,
    tau: float,
    *,
    is_goal_met: bool,
    step_index: int,
    max_steps: int = MAX_STEPS,
) -> Tuple[bool, Optional[str]]:
    """Stopping decision with strict precedence: goal -> max_steps -> no_valid -> threshold."""
    if is_goal_met:
        return True, END_GOAL
    if step_index >= max_steps:
        return True, END_MAX_STEPS
    if best_score == -math.inf:
        return True, END_NO_VALID
    if best_score - current_progress <= tau:
        return True, END_THRESHOLD
    return False, None


def decide_step(
    state: Dict[str, Any],
    goal: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    tau: float,
    step_index: int,
    max_steps: int = MAX_STEPS,
) -> Dict[str, Any]:
    """Score all candidates, pick highest (first candidate on ties), and evaluate stopping."""
    cur = progress(state, goal)
    components = [score_components(p, state, goal, weights) for p in predictions]
    scores = [c["score"] for c in components]
    best_idx = None
    for idx, val in enumerate(scores):
        if val != -math.inf and (best_idx is None or val > scores[best_idx]):
            best_idx = idx
    best = scores[best_idx] if best_idx is not None else -math.inf
    stop, reason = should_stop(
        best,
        cur,
        tau,
        is_goal_met=goal_met(state, goal),
        step_index=step_index,
        max_steps=max_steps,
    )
    return {
        "progress": cur,
        "components": components,
        "scores": scores,
        "best_idx": best_idx,
        "best_score": best,
        "stop": stop,
        "end_reason": reason,
    }


def _jsonable_score(val: float) -> Optional[float]:
    return None if val == -math.inf else val


def run_tasks_batched(
    tasks: Sequence[Dict[str, Any]],
    propose: ProposeFn,
    predict_batch: PredictBatchFn,
    env_step: EnvStepFn,
    weights: Dict[str, float],
    tau: float,
    max_steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    max_candidates: int = MAX_CANDIDATES,
    *,
    predictor_kind: str = "cpu",
    resume_state: Optional[Dict[str, Any]] = None,
    on_round: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Advance a set of tasks synchronously with deterministic candidate batching and replanning."""
    if not 1 <= batch_size <= BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {BATCH_SIZE}]")
    if predictor_kind not in ("cpu", "model"):
        raise ValueError("predictor_kind must be 'cpu' or 'model'")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task_id encountered")

    live: Dict[str, Dict[str, Any]] = {}
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        live[task["task_id"]] = {
            "task": task,
            "state": dict(task["start_state"]),
            "steps": [],
            "end_reason": None,
            "model_calls": 0,
            "gen_tokens": 0,
            "prompt_tokens": 0,
            "invalid_outputs": 0,
            "wall_sec": 0.0,
            "candidate_predictions": 0,
            "cpu_predictions": 0,
            "model_candidates": 0,
        }
    batches: List[Dict[str, Any]] = []
    rnd = 0

    if resume_state is not None:
        snap = copy.deepcopy(resume_state)
        if set(snap["live"]) != set(ids) or snap["predictor_kind"] != predictor_kind:
            raise ValueError("Resume checkpoint task set or predictor_kind mismatch")
        for tid, ep in snap["live"].items():
            if ep["task"] != live[tid]["task"]:
                raise ValueError(f"Resume checkpoint task content mismatch for {tid}")
        live, batches, rnd = snap["live"], snap["batches"], snap["next_round"]

    while True:
        running = [tid for tid in sorted(live) if live[tid]["end_reason"] is None]
        if not running:
            break
        pending = []
        for tid in running:
            ep, task = live[tid], live[tid]["task"]
            step_idx = len(ep["steps"])
            if goal_met(ep["state"], task["goal"]) or step_idx >= max_steps:
                _, reason = should_stop(
                    -math.inf,
                    progress(ep["state"], task["goal"]),
                    tau,
                    is_goal_met=goal_met(ep["state"], task["goal"]),
                    step_index=step_idx,
                    max_steps=max_steps,
                )
                ep["end_reason"] = reason
                continue
            cands = list(propose(dict(ep["state"]), task["goal"], task["rules"]))
            if len(cands) > max_candidates:
                raise ValueError(f"{tid}: candidate count {len(cands)} exceeds max_candidates={max_candidates}")
            pending.append((tid, cands))

        items = [
            {
                "task_id": tid,
                "candidate_id": j,
                "rules": live[tid]["task"]["rules"],
                "state_before": dict(live[tid]["state"]),
                "action": action,
            }
            for tid, cands in pending
            for j, action in enumerate(cands)
        ]
        items.sort(key=lambda x: (x["task_id"], x["candidate_id"]))

        preds: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for b0 in range(0, len(items), batch_size):
            chunk = items[b0 : b0 + batch_size]
            t0 = time.time()
            out = predict_batch(chunk)
            dt = time.time() - t0
            if len(out) != len(chunk):
                raise ValueError(f"predict_batch returned {len(out)} items, expected {len(chunk)}")
            batches.append(
                {
                    "round": rnd,
                    "batch_index": len(batches),
                    "size": len(chunk),
                    "predictor_kind": predictor_kind,
                    "generate_calls": int(predictor_kind == "model"),
                    "members": [[x["task_id"], x["candidate_id"]] for x in chunk],
                    "real_mask": [1] * len(chunk),
                    "prompt_tokens": [p.get("prompt_tokens") for p in out],
                    "n_new_tokens": [p.get("n_new_tokens") for p in out],
                    "wall_sec": round(dt, 4),
                }
            )
            for item, pred in zip(chunk, out):
                preds[(item["task_id"], item["candidate_id"])] = pred
                ep = live[item["task_id"]]
                ep["candidate_predictions"] += 1
                ep["cpu_predictions"] += int(predictor_kind == "cpu")
                ep["model_candidates"] += int(predictor_kind == "model")
                ep["model_calls"] += int(predictor_kind == "model")
                ep["gen_tokens"] += int(pred.get("n_new_tokens") or 0)
                ep["prompt_tokens"] += int(pred.get("prompt_tokens") or 0)
                ep["invalid_outputs"] += 0 if pred.get("valid") else 1
                ep["wall_sec"] += dt / len(chunk)

        for tid, cands in pending:
            ep, task = live[tid], live[tid]["task"]
            step_idx = len(ep["steps"])
            plist = [preds[(tid, j)] for j in range(len(cands))]
            decision = decide_step(ep["state"], task["goal"], cands, plist, weights, tau, step_idx, max_steps)
            rec = {
                "step": step_idx,
                "round": rnd,
                "state": dict(ep["state"]),
                "progress": decision["progress"],
                "candidates": [
                    {
                        "candidate_id": j,
                        "action": a,
                        "prediction": p,
                        "score": _jsonable_score(comp["score"]),
                        "score_components": {**comp, "score": _jsonable_score(comp["score"])},
                    }
                    for j, (a, p, comp) in enumerate(zip(cands, plist, decision["components"]))
                ],
                "best_idx": decision["best_idx"],
                "best_score": _jsonable_score(decision["best_score"]),
            }
            if decision["stop"]:
                rec["executed"] = None
                ep["decisions"] = ep.get("decisions", []) + [rec]
                ep["end_reason"] = decision["end_reason"]
                continue
            action = cands[decision["best_idx"]]
            res = env_step(dict(ep["state"]), action, task["rules"])
            rec["executed"] = {"action": action, **res}
            ep["steps"].append(rec)
            ep["state"] = dict(res["state"])

        rnd += 1
        if on_round is not None:
            on_round(copy.deepcopy({"live": live, "batches": batches, "next_round": rnd, "predictor_kind": predictor_kind}))

    episodes = []
    for tid in sorted(live):
        ep, task = live[tid], live[tid]["task"]
        executed = [s["executed"] for s in ep["steps"]]
        limit = task["goal"].get("loss_limit_cents")
        episodes.append(
            {
                "task_id": tid,
                "end_reason": ep["end_reason"],
                "n_steps": len(executed),
                "final_state": ep["state"],
                "success": goal_met(ep["state"], task["goal"]),
                "changed_any": any(e.get("changed") for e in executed),
                "irreversible_executed": any(e.get("reversible_true") is False for e in executed),
                "loss_over_limit": limit is not None and ep["state"]["loss_cents"] > limit,
                "model_calls": ep["model_calls"],
                "model_candidates": ep["model_candidates"],
                "candidate_predictions": ep["candidate_predictions"],
                "cpu_predictions": ep["cpu_predictions"],
                "gen_tokens": ep["gen_tokens"],
                "prompt_tokens": ep["prompt_tokens"],
                "invalid_outputs": ep["invalid_outputs"],
                "wall_sec": round(ep["wall_sec"], 4),
                "steps": ep["steps"],
                "stop_decision": (ep.get("decisions") or [None])[-1],
            }
        )
    return {
        "episodes": episodes,
        "batches": batches,
        "generate_calls": sum(b["generate_calls"] for b in batches),
    }


def run_episode(
    task: Dict[str, Any],
    propose: ProposeFn,
    predict: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    env_step: EnvStepFn,
    weights: Dict[str, float],
    tau: float,
    max_steps: int = MAX_STEPS,
    *,
    predictor_kind: str = "cpu",
) -> Dict[str, Any]:
    """Convenience single-task wrapper reusing ``run_tasks_batched``."""

    def _pb(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [predict(x["state_before"], x["action"], x["rules"]) for x in items]

    return run_tasks_batched(
        [task],
        propose,
        _pb,
        env_step,
        weights,
        tau,
        max_steps,
        batch_size=MAX_CANDIDATES,
        predictor_kind=predictor_kind,
    )["episodes"][0]
