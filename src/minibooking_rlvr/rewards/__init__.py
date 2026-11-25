"""Strict JSON schema parsing and verifiable composite reward functions."""

from .parser import GOLD_KEYS, PREDICTION_KEYS, PredictionError, parse_json_object, parse_prediction, validate_gold, validate_prediction
from .reward import (
    GROUPS,
    INVALID_REWARD,
    W_CONFIDENCE,
    W_DELTA_F1,
    W_RISK,
    RewardBreakdown,
    brier_score,
    canonical_changes,
    compute_reward,
    delta_scores,
    risk_score,
)

__all__ = [
    "PREDICTION_KEYS",
    "GOLD_KEYS",
    "PredictionError",
    "parse_json_object",
    "validate_prediction",
    "parse_prediction",
    "validate_gold",
    "W_DELTA_F1",
    "W_RISK",
    "W_CONFIDENCE",
    "INVALID_REWARD",
    "GROUPS",
    "RewardBreakdown",
    "canonical_changes",
    "delta_scores",
    "risk_score",
    "brier_score",
    "compute_reward",
]
