"""Frozen evaluation metrics with explicit primary (all-sample) and valid-only views.

Invalid-output policy (frozen prior to test evaluation):
- ``valid``: ``parse_prediction(raw)`` succeeds (valid JSON + strict schema).
- ``exact``, ``delta_f1``, ``full_exact``: 0 for invalid outputs.
- ``reversible``: counted as incorrect (``predicted = not gold``) in macro-F1 and accuracy.
- ``cost_abs_err``: penalized at maximum 1.0 for invalid outputs.
- ``brier``: penalized at 1.0 (``confidence = 1.0``, ``exact = 0``).
- ``valid_only``: identical formulas evaluated over valid outputs only, reported alongside
  the primary view as a diagnostic without filtering test samples.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from minibooking_rlvr.rewards import (
    PredictionError,
    brier_score,
    delta_scores,
    parse_json_object,
    parse_prediction,
    risk_score,
)

ACTION_TYPES = ("cancel", "change_day", "clear", "confirm", "create", "inspect")
COST_TOL = 1e-6

INVALID_POLICY = {
    "valid_definition": "parse_prediction(raw) succeeds with strict JSON and schema checks.",
    "exact_and_delta_f1": "Invalid outputs score 0.",
    "full_exact": "Invalid outputs score 0.",
    "reversible": "Invalid outputs counted as incorrect (predicted = not gold).",
    "cost_abs_err": "Invalid outputs penalized at 1.0.",
    "brier": "Invalid outputs penalized at 1.0 (confidence=1.0, exact=0).",
    "valid_only_diagnostics": "Reported alongside primary all-sample metrics; never replaces primary metrics.",
}


def sample_tags(rec: Dict[str, Any]) -> Dict[str, Any]:
    g, m = rec["gold"], rec["metadata"]
    return {
        "action_type": m["action_type"],
        "legal": bool(m["legal"]),
        "reason": m["reason"],
        "allow_change": bool(rec["rules"]["allow_change"]),
        "gold_changes_empty": len(g["changes"]) == 0,
        "gold_reversible": bool(g["reversible"]),
        "gold_cost": g["cost"],
        "gold_cost_pos": g["cost"] > 0,
    }


def score_raw(rec: Dict[str, Any], raw: str) -> Dict[str, Any]:
    """Re-parse and score a single raw completion string against ``rec['gold']``."""
    g = rec["gold"]
    json_valid, json_error = True, None
    try:
        parse_json_object(raw)
    except PredictionError as exc:
        json_valid, json_error = False, str(exc)
    except Exception as exc:
        json_valid, json_error = False, f"{type(exc).__name__}: {exc}"

    parsed, valid, error = None, False, None
    try:
        parsed = parse_prediction(raw)
        valid = True
    except PredictionError as exc:
        error = f"PredictionError: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if valid:
        assert parsed is not None
        f1, ex = delta_scores(parsed["changes"], g["changes"])
        rev_ok = parsed["reversible"] == g["reversible"]
        cerr = abs(float(parsed["cost"]) - float(g["cost"]))
        conf = float(parsed["confidence"])
        metrics = {
            "delta_f1": float(f1),
            "exact": int(ex),
            "full_exact": int(bool(ex) and rev_ok and cerr < COST_TOL),
            "reversible_ok": int(rev_ok),
            "reversible_pred": bool(parsed["reversible"]),
            "cost_abs_err": float(cerr),
            "risk": float(risk_score(parsed["reversible"], g["reversible"], parsed["cost"], g["cost"])),
            "brier": float(brier_score(conf, ex)),
            "confidence": conf,
        }
        metrics_valid_only: Optional[Dict[str, Any]] = dict(metrics)
    else:
        metrics = {
            "delta_f1": 0.0,
            "exact": 0,
            "full_exact": 0,
            "reversible_ok": 0,
            "reversible_pred": (not g["reversible"]),
            "cost_abs_err": 1.0,
            "risk": 0.0,
            "brier": 1.0,
            "confidence": 1.0,
        }
        metrics_valid_only = None

    return {
        "json_valid": json_valid,
        "json_error": json_error,
        "valid": valid,
        "parsed": parsed,
        "error": error,
        "metrics": metrics,
        "metrics_valid_only": metrics_valid_only,
    }


def _mean(xs: Any) -> Optional[float]:
    vals = list(xs)
    return (sum(vals) / len(vals)) if vals else None


def macro_f1_reversible(gold: Sequence[bool], pred: Sequence[bool]) -> Dict[str, Any]:
    """Compute binary macro-F1 over present classes in ``{True, False}``."""
    out: Dict[str, Optional[float]] = {}
    for cls in (True, False):
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        out[str(cls).lower()] = None if (tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    present = [v for v in out.values() if v is not None]
    return {"macro_f1": (sum(present) / len(present)) if present else None, "per_class_f1": out}


def quantiles(
    xs: Sequence[float],
    qs: Sequence[float] = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0),
) -> Dict[str, Optional[float]]:
    """Linearly interpolated quantiles matching NumPy's default method (pure stdlib)."""
    if not xs:
        return {f"q{int(q * 100):02d}": None for q in qs}
    s = sorted(xs)
    res: Dict[str, Optional[float]] = {}
    for q in qs:
        pos = q * (len(s) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        res[f"q{int(q * 100):02d}"] = s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return res


def confidence_distribution(confs: Sequence[float]) -> Dict[str, Any]:
    """Summarize reported confidence values across 10 uniform bins in ``[0, 1]``."""
    edges = [i / 10 for i in range(11)]
    counts = [0] * 10
    for c in confs:
        bucket = min(int(c * 10), 9) if c < 1.0 else 9
        counts[bucket] += 1
    return {
        "n": len(confs),
        "bins": [f"[{edges[i]:.1f},{edges[i + 1]:.1f}{']' if i == 9 else ')'}" for i in range(10)],
        "counts": counts,
        "frac_eq_1": (sum(1 for c in confs if c == 1.0) / len(confs)) if confs else None,
        "mean": _mean(confs),
        "quantiles": quantiles(confs),
    }


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scored evaluation rows into primary, valid-only, and confidence distributions."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    m_all = [r["metrics"] for r in rows]
    v_rows = [r for r in rows if r["metrics_valid_only"] is not None]
    m_valid = [r["metrics_valid_only"] for r in v_rows]

    rev_main = macro_f1_reversible([r["tags"]["gold_reversible"] for r in rows], [m["reversible_pred"] for m in m_all])
    rev_valid = macro_f1_reversible(
        [r["tags"]["gold_reversible"] for r in v_rows],
        [m["reversible_pred"] for m in m_valid],
    )

    def _summarize(ms: Sequence[Dict[str, Any]], rev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n": len(ms),
            "delta_f1": _mean(m["delta_f1"] for m in ms),
            "exact": _mean(m["exact"] for m in ms),
            "full_exact": _mean(m["full_exact"] for m in ms),
            "reversible_acc": _mean(m["reversible_ok"] for m in ms),
            "reversible_macro_f1": rev["macro_f1"],
            "reversible_per_class_f1": rev["per_class_f1"],
            "cost_mae": _mean(m["cost_abs_err"] for m in ms),
            "cost_exact_acc": _mean(int(m["cost_abs_err"] < COST_TOL) for m in ms),
            "risk_mean": _mean(m["risk"] for m in ms),
            "brier": _mean(m["brier"] for m in ms),
            "confidence_mean": _mean(m["confidence"] for m in ms),
        }

    return {
        "n": n,
        "n_valid": len(m_valid),
        "n_json_valid": sum(bool(r["json_valid"]) for r in rows),
        "json_valid_rate": sum(bool(r["json_valid"]) for r in rows) / n,
        "parse_coverage": len(m_valid) / n,
        "n_truncated": sum(bool(r.get("truncated", False)) for r in rows),
        "truncation_rate": sum(bool(r.get("truncated", False)) for r in rows) / n,
        "mean_new_tokens": _mean(r.get("n_new_tokens", 0) for r in rows),
        "main": _summarize(m_all, rev_main),
        "valid_only": _summarize(m_valid, rev_valid),
        "confidence_dist_valid": confidence_distribution([m["confidence"] for m in m_valid]),
    }


def subset_defs(split: str) -> List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]]:
    """Return breakdown slices for ``split``, separating ``allow_change`` on ``test_comp``."""
    slices: List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]] = [("all", "All samples", lambda _t: True)]
    for act in ACTION_TYPES:
        slices.append((f"action={act}", f"Action: {act}", (lambda a_: (lambda t: t["action_type"] == a_))(act)))
    slices += [
        ("cancel_gold_nonempty", "cancel with non-empty gold changes", lambda t: t["action_type"] == "cancel" and not t["gold_changes_empty"]),
        ("cancel_gold_empty", "cancel with empty gold changes", lambda t: t["action_type"] == "cancel" and t["gold_changes_empty"]),
        ("gold_irreversible", "Gold irreversible (reversible=False)", lambda t: not t["gold_reversible"]),
        ("gold_cost_pos", "Gold immediate cost > 0", lambda t: t["gold_cost_pos"]),
        ("legal", "Legal actions", lambda t: t["legal"]),
        ("illegal", "Illegal actions", lambda t: not t["legal"]),
    ]
    if split == "test_comp":
        slices += [
            ("allow_change=True", "allow_change=True (true behavioral compositional shift)", lambda t: t["allow_change"]),
            ("allow_change=False", "allow_change=False (change fee inactive)", lambda t: not t["allow_change"]),
        ]
    return slices


def evaluate_model_outputs(
    records: Sequence[Dict[str, Any]],
    raw_outputs: Sequence[str],
    *,
    split: str = "test_id",
) -> Dict[str, Any]:
    """Score a sequence of raw completions aligned with ``records`` across all standard subsets."""
    if len(records) != len(raw_outputs):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(raw_outputs)} outputs")
    scored_rows = []
    for rec, raw in zip(records, raw_outputs):
        s = score_raw(rec, raw)
        scored_rows.append(
            {
                "sample_id": rec["sample_id"],
                "raw": raw,
                "json_valid": s["json_valid"],
                "valid": s["valid"],
                "error": s["error"],
                "tags": sample_tags(rec),
                "metrics": s["metrics"],
                "metrics_valid_only": s["metrics_valid_only"],
            }
        )
    subsets: Dict[str, Any] = OrderedDict()
    for name, label, pred in subset_defs(split):
        matched = [r for r in scored_rows if pred(r["tags"])]
        agg = aggregate_rows(matched)
        agg["label"] = label
        subsets[name] = agg
    return {"split": split, "n": len(records), "subsets": subsets}


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}" if isinstance(val, float) else str(val)


def render_eval_table(per_model_results: Dict[str, Dict[str, Any]], split: str) -> str:
    """Render a primary all-sample Markdown summary table for ``split``."""
    lines = [
        f"### Frozen Evaluation: `{split}` (Primary All-Sample View)",
        "",
        "| Model | n | Parse Cov. | Exact | DeltaF1 | Full Exact | Rev. Macro-F1 | Cost MAE | Brier | Mean Conf. |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model, res in per_model_results.items():
        agg = res["subsets"]["all"]
        m = agg["main"]
        lines.append(
            f"| {model} | {agg['n']} | {_fmt(agg['parse_coverage'])} | {_fmt(m['exact'])} | "
            f"{_fmt(m['delta_f1'])} | {_fmt(m['full_exact'])} | {_fmt(m['reversible_macro_f1'])} | "
            f"{_fmt(m['cost_mae'])} | {_fmt(m['brier'])} | {_fmt(m['confidence_mean'])} |"
        )
    return "\n".join(lines) + "\n"
