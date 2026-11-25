"""Unit tests for the accuracy-gated EntropyController (add/log modes, gating, checkpointing)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from minibooking_rlvr.rl import CTRL_STATE_REQUIRED, EntropyController, controller_fingerprint


def _mk(**kw) -> EntropyController:
    base = dict(
        H_ref=0.02,
        H_low_ratio=0.8,
        ema_decay=0.9,
        update_every=10,
        exact_threshold=0.9,
        lambda_init=0.0,
        lambda_min=0.0,
        lambda_max=0.01,
        mode="add",
        add_step_up=1e-4,
        add_step_down=5e-5,
        log_factor_up=2.0,
        log_factor_down=0.8,
        log_floor=1e-4,
    )
    base.update(kw)
    return EntropyController(**base)


def _feed(ctrl: EntropyController, n: int, h_val, ex_val, start=None):
    evs = []
    s0 = ctrl.state["last_step_seen"] + 1 if start is None else start
    for s in range(s0, s0 + n):
        h = h_val(s) if callable(h_val) else h_val
        ex = ex_val(s) if callable(ex_val) else ex_val
        evs.append(ctrl.observe(s, h, ex))
    return evs


def test_ema_direct_init_and_decay():
    c = _mk()
    c.observe(1, 0.03, 0.5)
    assert c.state["H_ema"] == 0.03
    c.observe(2, 0.01, 0.5)
    assert c.state["H_ema"] == pytest.approx(0.9 * 0.03 + 0.1 * 0.01)


def test_update_schedule_and_one_step_lag():
    c = _mk()
    evs = _feed(c, 25, 0.001, 0.0)
    assert [e["step"] for e in evs if e["update"]] == [10, 20]
    assert all(e["lambda_after"] == 0.0 for e in evs[:9])
    assert evs[9]["lambda_used"] == 0.0
    assert evs[10]["lambda_used"] == evs[9]["lambda_after"] > 0


def test_high_accuracy_low_entropy_does_not_trigger():
    c = _mk(lambda_init=0.005)
    _feed(c, 200, 0.5 * 0.02, 0.95)
    assert c.state["trigger_count"] == 0
    assert c.lam() == pytest.approx(0.004)
    assert c.state["n_decrease"] == 20

    c2 = _mk(lambda_init=0.0003)
    _feed(c2, 200, 0.5 * 0.02, 0.95)
    assert c2.state["trigger_count"] == 0 and c2.lam() == 0.0


def test_low_entropy_with_errors_increases_bounded():
    c = _mk()
    evs = _feed(c, 200, 0.5 * 0.02, 0.0)
    lam_traj = [e["lambda_after"] for e in evs if e["update"]]
    assert c.state["trigger_count"] == 20
    assert c.lam() == pytest.approx(20 * 1e-4)
    assert all(b >= a for a, b in zip(lam_traj, lam_traj[1:]))

    c_cap = _mk()
    _feed(c_cap, 2000, 0.5 * 0.02, 0.0)
    assert c_cap.lam() == 0.01 and c_cap.state["trigger_count"] == 200


def test_high_entropy_with_errors_does_not_trigger():
    c = _mk()
    _feed(c, 200, 2.0 * 0.02, 0.0)
    assert c.state["trigger_count"] == 0 and c.lam() == 0.0


def test_log_mode_increase_and_floor_decay():
    c = _mk(mode="log")
    evs = _feed(c, 200, 0.5 * 0.02, 0.0)
    lam_log = [e["lambda_after"] for e in evs if e["update"]]
    assert lam_log[0] == 1e-4 and lam_log[1] == pytest.approx(2e-4) and c.lam() == 0.01

    c_dec = _mk(mode="log", lambda_init=2e-4)
    _feed(c_dec, 40, 0.5 * 0.02, 0.95)
    assert c_dec.lam() == 0.0


def test_none_entropy_observation_skips_ema_only():
    c = _mk()
    c.observe(1, None, 0.5)
    c.observe(2, 0.02, 0.5)
    assert c.state["H_ema"] == 0.02
    assert c.state["n_obs_skipped"] == 1
    assert c.state["win_exact_n"] == 2


def test_checkpoint_save_load_bit_identical(tmp_path: Path):
    h_seq = lambda s: 0.02 * (0.5 + 0.4 * math.sin(s))
    e_seq = lambda s: 0.5 + 0.3 * math.cos(s)

    c_full = _mk(mode="log")
    ev_full = _feed(c_full, 25, h_seq, e_seq)

    c_a = _mk(mode="log")
    _feed(c_a, 13, h_seq, e_seq)
    ckpt_file = tmp_path / "controller_state.json"
    c_a.save(ckpt_file)

    c_b = _mk(mode="log")
    default_sd = c_b.state_dict()
    c_b.load(ckpt_file)
    assert c_b.state_dict() == c_a.state_dict() != default_sd
    assert controller_fingerprint(c_b.state_dict()) == controller_fingerprint(c_a.state_dict())

    ev_b = _feed(c_b, 12, h_seq, e_seq)
    assert ev_b == ev_full[13:]
    assert c_b.state_dict() == c_full.state_dict()
    assert all(k in c_b.state_dict() for k in CTRL_STATE_REQUIRED)


def test_mismatched_config_and_nonconsecutive_steps_rejected():
    c_log = _mk(mode="log")
    with pytest.raises(ValueError, match="config mismatch"):
        _mk(mode="add").load_state_dict(c_log.state_dict())

    c = _mk()
    c.observe(1, 0.02, 0.5)
    with pytest.raises(ValueError, match="Non-consecutive step"):
        c.observe(3, 0.02, 0.5)


def test_formal_config_g3_reach_preview():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "grpo_formal_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ctrl = EntropyController(**cfg["groups"]["G3"]["controller"])
    assert ctrl.lam() == 0.0 and ctrl.state["H_ema"] is None
    updates = [e["lambda_after"] for e in _feed(ctrl, 200, 1e-3, 0.0) if e["update"]]
    expected_prefix = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2]
    assert len(updates) == 20
    for got, want in zip(updates[:12], expected_prefix):
        assert got == pytest.approx(want, rel=1e-12)
