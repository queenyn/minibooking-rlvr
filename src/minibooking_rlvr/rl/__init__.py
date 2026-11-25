"""Pure-PyTorch GRPO core, H/D/S/C online diagnostics, entropy feedback controller, and offline filtering."""

from .controller import CTRL_PARAM_KEYS, CTRL_STATE_REQUIRED, EntropyController, controller_fingerprint
from .diagnostics import (
    COMPONENTS,
    DEFAULT_S_TOL,
    TASK_TYPES,
    batch_stats,
    change_key,
    classify_zero_var_group,
    group_stats,
    make_sample_row,
    summarize_zero_var_groups,
)
from .filtering import filter_all_correct_prompts

__all__ = [
    "CTRL_STATE_REQUIRED",
    "CTRL_PARAM_KEYS",
    "EntropyController",
    "controller_fingerprint",
    "DEFAULT_S_TOL",
    "TASK_TYPES",
    "COMPONENTS",
    "change_key",
    "make_sample_row",
    "group_stats",
    "batch_stats",
    "classify_zero_var_group",
    "summarize_zero_var_groups",
    "filter_all_correct_prompts",
]
