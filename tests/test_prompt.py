"""Unit tests for frozen prompt formatting and strict no-label-leakage guarantees."""

from __future__ import annotations

import json

from minibooking_rlvr.data import INPUT_KEYS, PROMPT_VERSION, build_messages, gold_completion, select_boundary_samples
from minibooking_rlvr.env import Action, State, label_transition
from minibooking_rlvr.rewards import parse_prediction
from conftest import make_rules


def test_prompt_reads_only_input_keys():
    rec = label_transition(State("confirmed", 1, 900, 100, 0), Action("cancel"), make_rules())
    rec["sample_id"] = "test-00001"
    rec["secret_sentinel"] = "DO_NOT_LEAK_TOKEN"
    msgs = build_messages(rec)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    combined = msgs[0]["content"] + msgs[1]["content"]
    assert "DO_NOT_LEAK_TOKEN" not in combined
    assert "state_after" not in msgs[1]["content"]
    assert "gold" not in msgs[1]["content"]
    assert PROMPT_VERSION == "minibooking_prompt_v1"
    assert INPUT_KEYS == ("rules", "state_before", "action")


def test_gold_completion_passes_strict_parser():
    rec = label_transition(State("confirmed", 1, 900, 100, 0), Action("cancel"), make_rules())
    comp = gold_completion(rec)
    parsed = parse_prediction(comp)
    assert parsed["changes"] == rec["gold"]["changes"]
    assert parsed["reversible"] is False
    assert parsed["cost"] == 0.5
    assert parsed["confidence"] == 1.0


def test_select_boundary_samples_unique():
    rules = make_rules()
    raw_recs = [
        label_transition(State("empty", None, 500, 0, 0), Action("create", 1), rules),
        label_transition(State("draft", 1, 500, 0, 0), Action("confirm"), rules),
        label_transition(State("draft", 1, 500, 0, 0), Action("inspect"), rules),
        label_transition(State("empty", None, 500, 0, 0), Action("confirm"), rules),
        label_transition(State("confirmed", 1, 400, 100, 0), Action("cancel"), rules),
        label_transition(State("confirmed", 1, 400, 100, 0), Action("change_day", 2), rules),
    ]
    for i, r in enumerate(raw_recs):
        r["sample_id"] = f"s-{i:03d}"
    picked = select_boundary_samples(raw_recs, n_per_category=1)
    ids = [r["sample_id"] for _, r in picked]
    assert len(ids) == len(set(ids))
    assert len(picked) == 5
    # Verify JSON-serializable prompt payload
    assert json.loads(gold_completion(raw_recs[0]))["confidence"] == 1.0
