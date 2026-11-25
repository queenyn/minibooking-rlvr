"""Pure verifiable reward functions for MiniBooking-RLVR (G1 / G2 / G3).

Reward definitions:
- Invalid JSON or schema violation (all groups): ``R = -1.0``
- G1 (delta-only): ``R = DeltaF1``
- G2 / G3 (composite): ``R = 0.60 * DeltaF1 + 0.25 * RiskScore + 0.15 * (1 - Brier)``
  where:
    - ``RiskScore = 0.5 * I(pred_rev == gold_rev) + 0.5 * (1 - |pred_cost - gold_cost|)``
    - ``Brier = (confidence - exact) ** 2``

G2 and G3 share the exact same reward function; G3 adds adaptive entropy regularization
in the GRPO loss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .parser import PredictionError, parse_prediction, validate_gold

W_DELTA_F1 = 0.60
W_RISK = 0.25
W_CONFIDENCE = 0.15
INVALID_REWARD = -1.0
GROUPS = ("G1", "G2", "G3")


def canonical_changes(changes: List[Dict[str, Any]]) -> FrozenSet[Tuple[str, str, str]]:
    """Canonicalize a list of field deltas into an order-invariant, type-aware frozenset.

    Each element is ``(field, type(value).__name__, json.dumps(value))`` so ``0`` and
    ``False`` are never conflated.
    """
    return frozenset(
        (item["field"], type(item["value"]).__name__, json.dumps(item["value"], ensure_ascii=False))
        for item in changes
    )


def delta_scores(predicted: List[Dict[str, Any]], target: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Compute ``(DeltaF1, exact)`` over canonicalized delta sets."""
    pred_set = canonical_changes(predicted)
    target_set = canonical_changes(target)
    if not pred_set and not target_set:
        return 1.0, 1
    if not pred_set or not target_set:
        return 0.0, 0
    tp = len(pred_set & target_set)
    f1 = 2 * tp / (len(pred_set) + len(target_set))
    exact = 1 if pred_set == target_set else 0
    return f1, exact


def risk_score(pred_reversible: bool, gold_reversible: bool, pred_cost: float, gold_cost: float) -> float:
    """Compute ``RiskScore in [0, 1]`` combining reversibility accuracy and cost accuracy."""
    reversible_correct = 1 if pred_reversible == gold_reversible else 0
    return 0.5 * reversible_correct + 0.5 * (1.0 - abs(pred_cost - gold_cost))


def brier_score(confidence: float, exact: int) -> float:
    """Compute squared calibration error ``(confidence - exact) ** 2``."""
    return (confidence - exact) ** 2


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-completion reward and decomposed components for online diagnostics."""

    valid: bool
    reward: float
    f1: float = 0.0
    exact: int = 0
    risk: Optional[float] = None
    brier: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_reward(raw: str, gold: Dict[str, Any], group: str = "G2") -> RewardBreakdown:
    """Score a raw model completion against ``gold`` under reward group ``G1`` / ``G2`` / ``G3``."""
    if group not in GROUPS:
        raise ValueError(f"Unknown reward group: {group!r}; expected one of {GROUPS}")
    validate_gold(gold)
    try:
        pred = parse_prediction(raw)
    except PredictionError as exc:
        return RewardBreakdown(valid=False, reward=INVALID_REWARD, error=str(exc))

    f1, exact = delta_scores(pred["changes"], gold["changes"])
    risk = risk_score(pred["reversible"], gold["reversible"], pred["cost"], gold["cost"])
    brier = brier_score(pred["confidence"], exact)
    if group == "G1":
        reward = f1
    else:
        reward = W_DELTA_F1 * f1 + W_RISK * risk + W_CONFIDENCE * (1.0 - brier)
    return RewardBreakdown(valid=True, reward=reward, f1=f1, exact=exact, risk=risk, brier=brier)
