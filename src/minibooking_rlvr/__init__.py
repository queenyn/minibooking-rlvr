"""MiniBooking-RLVR: Verifiable state-delta & reversibility benchmark, pure-PyTorch GRPO,
post-training diagnostics, and risk-aware 1-step MPC planning.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .env import Action, Rules, State, apply_changes, compute_cost, extract_changes, is_reversible, label_transition, step_raw
from .rewards import PredictionError, RewardBreakdown, compute_reward, parse_prediction

__all__ = [
    "__version__",
    "Action",
    "Rules",
    "State",
    "step_raw",
    "extract_changes",
    "compute_cost",
    "apply_changes",
    "is_reversible",
    "label_transition",
    "PredictionError",
    "RewardBreakdown",
    "parse_prediction",
    "compute_reward",
]
