"""Accuracy-gated adaptive entropy feedback controller (``EntropyController``).

Controls the entropy regularization coefficient ``lambda_entropy`` in the GRPO loss:
    ``loss = policy_loss + beta_kl * KL - lambda_entropy * H_reg``

Note: ``lambda_entropy`` is distinct from the fixed KL penalty coefficient ``beta_kl``.

Update rule (evaluated every ``update_every`` optimizer steps):
- Maintain an exponential moving average ``H_ema`` of effective-group token entropy
  (initialized directly from the first valid observation).
- Trigger an increase iff BOTH conditions hold:
    1. ``H_ema < H_low`` (entropy has fallen below ``H_low_ratio * H_ref``), AND
    2. ``exact_window < exact_threshold`` (accuracy is not already saturated).
- Otherwise decay ``lambda_entropy`` toward ``lambda_min`` (0.0).
- Supports both additive (``mode="add"``) and multiplicative/logarithmic (``mode="log"``)
  schedules, clamped to ``[lambda_min, lambda_max]``, with strict state serialization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

CTRL_STATE_REQUIRED = (
    "H_ref",
    "H_ema",
    "lambda_entropy",
    "last_update_step",
    "trigger_count",
    "H_low",
    "exact_threshold",
)
CTRL_PARAM_KEYS = (
    "H_ref",
    "H_low",
    "H_low_ratio",
    "ema_decay",
    "update_every",
    "exact_threshold",
    "lambda_init",
    "lambda_min",
    "lambda_max",
    "mode",
    "add_step_up",
    "add_step_down",
    "log_factor_up",
    "log_factor_down",
    "log_floor",
)


def controller_fingerprint(state_dict: Dict[str, Any]) -> str:
    """Return a deterministic 16-char SHA256 fingerprint of a controller state dict."""
    return hashlib.sha256(json.dumps(state_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class EntropyController:
    """Single-knob, accuracy-gated entropy coefficient controller with checkpoint verification."""

    VERSION = "day8_entropy_controller_v1"

    def __init__(
        self,
        H_ref: float,
        H_low: Optional[float] = None,
        H_low_ratio: float = 0.8,
        ema_decay: float = 0.9,
        update_every: int = 10,
        exact_threshold: float = 0.9,
        lambda_init: Optional[float] = None,
        lambda_min: float = 0.0,
        lambda_max: float = 0.01,
        mode: str = "add",
        add_step_up: float = 1e-4,
        add_step_down: float = 5e-5,
        log_factor_up: float = 2.0,
        log_factor_down: float = 0.8,
        log_floor: float = 1e-4,
    ) -> None:
        if H_ref is None or not math.isfinite(H_ref) or H_ref <= 0:
            raise ValueError(f"H_ref must be a finite positive float, got {H_ref!r}")
        resolved_h_low = float(H_low) if H_low is not None else float(H_low_ratio) * float(H_ref)
        resolved_lam_init = float(lambda_min) if lambda_init is None else float(lambda_init)
        if not (0.0 <= lambda_min <= resolved_lam_init <= lambda_max):
            raise ValueError("Require 0 <= lambda_min <= lambda_init <= lambda_max")
        if not (0.0 < ema_decay < 1.0 and int(update_every) >= 1 and mode in ("add", "log")):
            raise ValueError("Invalid ema_decay, update_every, or mode")
        if not (add_step_up >= 0 and add_step_down >= 0 and log_factor_up > 1.0 and 0.0 < log_factor_down < 1.0 and log_floor > 0):
            raise ValueError("Invalid additive/logarithmic step parameters")

        self.cfg: Dict[str, Any] = {
            "H_ref": float(H_ref),
            "H_low": resolved_h_low,
            "H_low_ratio": float(H_low_ratio),
            "ema_decay": float(ema_decay),
            "update_every": int(update_every),
            "exact_threshold": float(exact_threshold),
            "lambda_init": resolved_lam_init,
            "lambda_min": float(lambda_min),
            "lambda_max": float(lambda_max),
            "mode": mode,
            "add_step_up": float(add_step_up),
            "add_step_down": float(add_step_down),
            "log_factor_up": float(log_factor_up),
            "log_factor_down": float(log_factor_down),
            "log_floor": float(log_floor),
        }
        self.state: Dict[str, Any] = {
            "version": self.VERSION,
            "H_ref": float(H_ref),
            "H_low": resolved_h_low,
            "exact_threshold": float(exact_threshold),
            "H_ema": None,
            "lambda_entropy": resolved_lam_init,
            "last_update_step": 0,
            "trigger_count": 0,
            "n_updates": 0,
            "n_decrease": 0,
            "n_obs": 0,
            "n_obs_skipped": 0,
            "last_step_seen": 0,
            "win_exact_sum": 0.0,
            "win_exact_n": 0,
            "last_action": None,
            "cfg": dict(self.cfg),
        }

    def lam(self) -> float:
        """Return the current ``lambda_entropy`` value for the next optimizer step."""
        return float(self.state["lambda_entropy"])

    def _up(self, lam: float) -> float:
        c = self.cfg
        if c["mode"] == "add":
            lam = lam + c["add_step_up"]
        else:
            lam = c["log_floor"] if lam < c["log_floor"] else lam * c["log_factor_up"]
        return min(c["lambda_max"], max(c["lambda_min"], lam))

    def _down(self, lam: float) -> float:
        c = self.cfg
        if c["mode"] == "add":
            lam = lam - c["add_step_down"]
        else:
            lam = lam * c["log_factor_down"]
            if lam < c["log_floor"]:
                lam = 0.0
        return min(c["lambda_max"], max(c["lambda_min"], lam))

    def observe(self, step: int, H_obs: Optional[float], exact: Optional[float]) -> Dict[str, Any]:
        """Record step-end diagnostics and update ``lambda_entropy`` every ``update_every`` steps.

        Step numbers must be strictly consecutive (``last_step_seen + 1``) so checkpoint
        resume mismatches fail loudly.
        """
        st, c = self.state, self.cfg
        if step != st["last_step_seen"] + 1:
            raise ValueError(f"Non-consecutive step: expected {st['last_step_seen'] + 1}, got {step}")

        lam_used = st["lambda_entropy"]
        if H_obs is not None and math.isfinite(H_obs):
            st["H_ema"] = (
                float(H_obs)
                if st["H_ema"] is None
                else c["ema_decay"] * st["H_ema"] + (1.0 - c["ema_decay"]) * float(H_obs)
            )
            st["n_obs"] += 1
        else:
            st["n_obs_skipped"] += 1

        if exact is not None and math.isfinite(exact):
            st["win_exact_sum"] += float(exact)
            st["win_exact_n"] += 1

        ev: Dict[str, Any] = {
            "step": step,
            "lambda_used": lam_used,
            "H_obs": H_obs,
            "exact_obs": exact,
            "update": False,
            "action": None,
            "exact_window": None,
            "H_low": st["H_low"],
            "H_ref": st["H_ref"],
        }

        if step % c["update_every"] == 0:
            ex_w = st["win_exact_sum"] / st["win_exact_n"] if st["win_exact_n"] else None
            low = st["H_ema"] is not None and st["H_ema"] < st["H_low"]
            err = ex_w is not None and ex_w < st["exact_threshold"]
            old = st["lambda_entropy"]
            if low and err:
                new = self._up(old)
                st["trigger_count"] += 1
                action = "increase"
            else:
                new = self._down(old)
                st["n_decrease"] += 1
                action = "decrease"
            reason = f"H_ema<H_low={low}, exact_window<{st['exact_threshold']}={err}"
            st.update(
                lambda_entropy=new,
                last_update_step=step,
                n_updates=st["n_updates"] + 1,
                win_exact_sum=0.0,
                win_exact_n=0,
                last_action={
                    "step": step,
                    "action": action,
                    "reason": reason,
                    "lambda_old": old,
                    "lambda_new": new,
                    "H_ema": st["H_ema"],
                    "exact_window": ex_w,
                },
            )
            ev.update(update=True, action=action, exact_window=ex_w, reason=reason)

        st["last_step_seen"] = step
        ev.update(
            H_ema=st["H_ema"],
            lambda_after=st["lambda_entropy"],
            trigger_count=st["trigger_count"],
            last_update_step=st["last_update_step"],
            n_updates=st["n_updates"],
        )
        return ev

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        missing = [k for k in CTRL_STATE_REQUIRED + ("version", "cfg", "last_step_seen") if k not in sd]
        if missing:
            raise ValueError(f"controller_state missing required keys: {missing}")
        if sd["version"] != self.VERSION:
            raise ValueError(f"Controller version mismatch: {sd['version']!r} != {self.VERSION!r}")
        if sd["cfg"] != self.cfg:
            raise ValueError(f"Controller config mismatch:\n  checkpoint={sd['cfg']}\n  current={self.cfg}")
        self.state = copy.deepcopy(sd)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
