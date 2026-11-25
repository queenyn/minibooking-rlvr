"""Online H/D/S/C group/batch diagnostics and offline zero-variance group decomposition.

Metrics:
- **D (Diversity)**: ``n_unique_canonical_changes / n_valid`` within a K-sample group.
- **S (Signal / Zero-Variance)**: fraction of groups with population reward std ``< s_tol``
  (carrying zero GRPO policy gradient), decomposed into ``all_correct``, ``all_wrong``,
  ``mixed``, and ``parse_fail`` (and offline into ``all_correct / all_wrong / other_same``).
- **C (Concentration)**: mode frequency ``max_count / n_valid`` paired with ``mode_correct``
  to distinguish collapse onto the correct answer from collapse onto a wrong answer.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from minibooking_rlvr.rewards import canonical_changes, compute_reward

DEFAULT_S_TOL = 1e-6
TASK_TYPES = ("all_correct", "all_wrong", "mixed", "parse_fail")
COMPONENTS = ("f1", "exact", "risk", "brier")


def change_key(raw: str, valid: bool) -> Optional[str]:
    """Return a canonical JSON key for ``changes`` if valid, else ``None``."""
    if not valid:
        return None
    return json.dumps(sorted(canonical_changes(json.loads(raw)["changes"])), ensure_ascii=False)


def make_sample_row(
    raw: str,
    gold: Dict[str, Any],
    reward_group: str,
    alt_group: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one completion under ``reward_group`` and optionally an alternate tracking group."""
    if alt_group is None:
        alt_group = "G2" if reward_group == "G1" else "G1"
    bd = compute_reward(raw, gold, reward_group)
    bd_alt = compute_reward(raw, gold, alt_group)
    return {
        "raw": raw,
        "valid": bool(bd.valid),
        "error": bd.error,
        "reward": float(bd.reward),
        "reward_alt": float(bd_alt.reward),
        "f1": float(bd.f1),
        "exact": int(bd.exact),
        "risk": bd.risk,
        "brier": bd.brier,
        "change_key": change_key(raw, bd.valid),
    }


def _pstd(xs: Sequence[float]) -> Optional[float]:
    return float(np.std(np.asarray(xs, dtype=np.float64))) if len(xs) else None


def _pvar(xs: Sequence[Optional[float]]) -> Optional[float]:
    clean = [x for x in xs if x is not None]
    return float(np.var(np.asarray(clean, dtype=np.float64))) if clean else None


def _mean_or_none(xs: Sequence[Optional[float]]) -> Optional[float]:
    clean = [x for x in xs if x is not None]
    return float(np.mean(clean)) if clean else None


def group_stats(rows: Sequence[Dict[str, Any]], s_tol: float = DEFAULT_S_TOL) -> Dict[str, Any]:
    """Compute D/S/C diagnostics for a single K-sample prompt group."""
    k_size = len(rows)
    valid_rows = [r for r in rows if r["valid"]]
    n_valid = len(valid_rows)
    n_exact = sum(1 for r in valid_rows if r["exact"] == 1)

    if n_valid == 0:
        task_type = "parse_fail"
    elif n_exact == k_size:
        task_type = "all_correct"
    elif n_exact == 0:
        task_type = "all_wrong"
    else:
        task_type = "mixed"

    diversity = concentration = mode_correct = mode_tie = None
    n_unique = 0
    if n_valid:
        cnt = Counter(r["change_key"] for r in valid_rows)
        top = max(cnt.values())
        modes = [k for k, c in cnt.items() if c == top]
        correct_keys = {r["change_key"] for r in valid_rows if r["exact"] == 1}
        assert len(correct_keys) <= 1, "Exact completions must share one canonical changes key"
        n_unique = len(cnt)
        diversity = n_unique / n_valid
        concentration = top / n_valid
        mode_correct = any(m in correct_keys for m in modes)
        mode_tie = len(modes) > 1

    rs = [r["reward"] for r in rows]
    rs_alt = [r["reward_alt"] for r in rows]
    r_std = _pstd(rs)
    alt_std = _pstd(rs_alt)
    assert r_std is not None and alt_std is not None

    comp_var: Dict[str, Optional[float]] = {f"var_{c}": _pvar([r[c] for r in valid_rows]) for c in COMPONENTS}
    comp_var.update(
        var_valid=_pvar([float(r["valid"]) for r in rows]),
        var_reward=r_std**2,
        var_reward_alt=alt_std**2,
    )
    return {
        "K": k_size,
        "n_valid": n_valid,
        "valid_parse_rate": n_valid / k_size,
        "n_exact": n_exact,
        "task_type": task_type,
        "D": diversity,
        "C": concentration,
        "mode_correct": mode_correct,
        "mode_tie": mode_tie,
        "n_unique": n_unique,
        "reward_mean": float(np.mean(rs)),
        "reward_std": r_std,
        "zero_var": r_std < s_tol,
        "reward_alt_mean": float(np.mean(rs_alt)),
        "reward_alt_std": alt_std,
        "zero_var_alt": alt_std < s_tol,
        **comp_var,
    }


def batch_stats(groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate group-level D/S/C diagnostics into step-level scalars."""
    n = len(groups)
    cnt = Counter(g["task_type"] for g in groups)
    zero = [g for g in groups if g["zero_var"]]
    zc = Counter(g["task_type"] for g in zero)
    covered = [g for g in groups if g["D"] is not None]

    out: Dict[str, Any] = {
        "n_groups": n,
        "S_reward": len(zero) / n,
        "n_zero_var": len(zero),
        "n_effective": n - len(zero),
        "S_reward_alt": sum(g["zero_var_alt"] for g in groups) / n,
        "n_zero_var_alt": sum(g["zero_var_alt"] for g in groups),
    }
    for task_type in TASK_TYPES:
        out[f"n_{task_type}"] = cnt[task_type]
        out[f"S_task_{task_type}"] = cnt[task_type] / n
        out[f"zero_var_{task_type}"] = zc[task_type]
    out.update(
        D_mean=_mean_or_none([g["D"] for g in covered]),
        C_mean=_mean_or_none([g["C"] for g in covered]),
        mode_correct_rate=_mean_or_none([float(g["mode_correct"]) for g in covered]),
        dc_coverage=len(covered) / n,
        valid_parse_rate=sum(g["n_valid"] for g in groups) / sum(g["K"] for g in groups),
    )
    return out


def classify_zero_var_group(
    rewards: Sequence[float],
    *,
    s_tol: float = DEFAULT_S_TOL,
    r_max: float = 1.0,
    r_zero: float = 0.0,
    eq_tol: float = 1e-9,
) -> str:
    """Classify a K-sample reward vector into ``effective / all_correct / all_wrong / other_same``."""
    std = _pstd(rewards)
    if std is None or std >= s_tol:
        return "effective"
    v = rewards[0]
    if abs(v - r_max) < eq_tol:
        return "all_correct"
    if abs(v - r_zero) < eq_tol:
        return "all_wrong"
    return "other_same"


def summarize_zero_var_groups(reward_groups: Sequence[Sequence[float]], s_tol: float = DEFAULT_S_TOL) -> Dict[str, Any]:
    """Summarize zero-variance breakdown across a sequence of K-sample reward groups."""
    counts = Counter(classify_zero_var_group(rw, s_tol=s_tol) for rw in reward_groups)
    total = len(reward_groups)
    n_zv = counts["all_correct"] + counts["all_wrong"] + counts["other_same"]
    return {
        "n_groups": total,
        "n_effective": counts["effective"],
        "n_zero_var": n_zv,
        "n_all_correct": counts["all_correct"],
        "n_all_wrong": counts["all_wrong"],
        "n_other_same": counts["other_same"],
        "ratio_effective": (counts["effective"] / total) if total else None,
        "ratio_zero_var": (n_zv / total) if total else None,
        "ratio_all_correct": (counts["all_correct"] / total) if total else None,
        "ratio_all_wrong": (counts["all_wrong"] / total) if total else None,
        "ratio_other_same": (counts["other_same"] / total) if total else None,
    }
