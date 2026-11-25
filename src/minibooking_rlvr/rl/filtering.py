"""Offline K-sample difficulty filtering to remove saturated zero-variance prompts before GRPO."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from minibooking_rlvr.rewards import compute_reward


def filter_all_correct_prompts(
    records: Sequence[Dict[str, Any]],
    sampled_raws_by_id: Dict[str, Sequence[str]],
    *,
    reward_group: str = "G1",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter out prompts whose K sampled completions are all valid and exact.

    Keeping all-wrong, mixed, and parse-fail prompts eliminates saturated zero-variance
    groups before RL while preserving hard examples (e.g., non-empty ``cancel``).
    """
    kept: List[Dict[str, Any]] = []
    removed_ids: List[str] = []
    before_actions: Counter[str] = Counter()
    after_actions: Counter[str] = Counter()
    zv_before = 0
    zv_after = 0

    for rec in records:
        sid = rec["sample_id"]
        action_type = rec["metadata"]["action_type"]
        before_actions[action_type] += 1
        raws = sampled_raws_by_id[sid]
        bds = [compute_reward(raw, rec["gold"], reward_group) for raw in raws]
        rewards = [b.reward for b in bds]
        is_zv = (max(rewards) - min(rewards)) < 1e-6 if rewards else True
        if is_zv:
            zv_before += 1

        all_correct = len(bds) > 0 and all(b.valid and b.exact == 1 for b in bds)
        if all_correct:
            removed_ids.append(sid)
        else:
            kept.append(rec)
            after_actions[action_type] += 1
            if is_zv:
                zv_after += 1

    n_before = len(records)
    n_after = len(kept)
    report = {
        "pool_before": n_before,
        "pool_after": n_after,
        "removed_all_correct": len(removed_ids),
        "zero_var_rate_before": (zv_before / n_before) if n_before else None,
        "zero_var_rate_after": (zv_after / n_after) if n_after else None,
        "action_type_before": dict(sorted(before_actions.items())),
        "action_type_after": dict(sorted(after_actions.items())),
    }
    return kept, report
