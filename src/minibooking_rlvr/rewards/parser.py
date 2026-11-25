"""Strict JSON parsing and schema validation for model outputs.

Protocol guarantees:
- Accepts strictly a single JSON object with no Markdown code fences or surrounding prose.
- Rejects duplicate JSON keys, NaN, and +/-Infinity.
- Never performs silent coercion (e.g., clipping out-of-range confidence or converting strings
  to booleans), preventing reward hacking across implementations.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, FrozenSet, List, Tuple

from minibooking_rlvr.env.state import ALLOWED_DAYS, STATE_FIELDS, STATUSES, is_strict_int

PREDICTION_KEYS = frozenset({"changes", "reversible", "cost", "confidence"})
GOLD_KEYS = frozenset({"changes", "reversible", "cost"})


class PredictionError(ValueError):
    """Raised when a model output fails strict JSON or schema validation (reward = -1)."""


def _reject_constant(value: str) -> Any:
    raise PredictionError(f"Non-finite JSON constant rejected: {value}")


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Strictly parse a raw completion string into a top-level JSON object."""
    if not isinstance(raw, str):
        raise PredictionError("Raw output must be a string")
    try:
        payload = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except PredictionError:
        raise
    except ValueError as exc:
        raise PredictionError(f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredictionError("Top-level JSON value must be an object")
    return payload


def validate_prediction(
    payload: Any,
    *,
    required_keys: FrozenSet[str] = PREDICTION_KEYS,
) -> Dict[str, Any]:
    """Validate a parsed object against the strict prediction or gold schema."""
    if not isinstance(payload, dict):
        raise PredictionError("Payload must be a dict")
    if set(payload) != required_keys:
        raise PredictionError(f"Payload keys must equal {sorted(required_keys)}, got {sorted(payload)}")
    if type(payload["reversible"]) is not bool:
        raise PredictionError("reversible must be a strict bool")

    numeric_keys = ["cost"]
    if "confidence" in required_keys:
        numeric_keys.append("confidence")
    for key in numeric_keys:
        value = payload[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise PredictionError(f"{key} must be a finite int or float (not bool)")
        if not 0 <= value <= 1:
            raise PredictionError(f"{key} must lie in [0, 1], got {value!r}")

    if type(payload["changes"]) is not list:
        raise PredictionError("changes must be a list")

    seen = set()
    for change in payload["changes"]:
        if not isinstance(change, dict):
            raise PredictionError("Each item in changes must be a dict")
        if set(change) != {"field", "value"}:
            raise PredictionError("Each change item must contain exact keys {'field', 'value'}")
        field = change["field"]
        value = change["value"]
        if field not in STATE_FIELDS:
            raise PredictionError(f"Unknown field in changes: {field!r}")
        if field in seen:
            raise PredictionError(f"Duplicate field in changes: {field!r}")
        seen.add(field)

        if field == "status":
            if value not in STATUSES:
                raise PredictionError(f"Invalid status value: {value!r}")
        elif field == "day":
            if not (value is None or (is_strict_int(value) and value in ALLOWED_DAYS)):
                raise PredictionError(f"day must be None or an int in {ALLOWED_DAYS}, got {value!r}")
        elif field in ("balance_cents", "paid_cents", "loss_cents"):
            if not is_strict_int(value) or value < 0:
                raise PredictionError(f"{field} must be a non-negative strict int, got {value!r}")

    return payload


def parse_prediction(raw: str) -> Dict[str, Any]:
    """Parse and schema-validate a raw model completion string."""
    return validate_prediction(parse_json_object(raw))


def validate_gold(gold: Any) -> Dict[str, Any]:
    """Validate ground-truth labels; raises ``ValueError`` (dataset bug) rather than ``PredictionError``."""
    try:
        return validate_prediction(gold, required_keys=GOLD_KEYS)
    except PredictionError as exc:
        raise ValueError(f"Invalid gold label (dataset bug): {exc}") from exc
