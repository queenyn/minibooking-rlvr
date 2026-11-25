"""Unit tests for leakage-free scenario-family dataset generation and byte-level determinism."""

from __future__ import annotations

from minibooking_rlvr.data import (
    FROZEN_DATA_SHA256,
    SPLITS,
    TARGET_SIZES,
    assign_families,
    cross_split_checks,
    generate_dataset,
    sha256_text,
)
from minibooking_rlvr.data.dataset import jsonl_text


def test_family_partition_totals():
    fams = assign_families()
    assert {s: len(v) for s, v in fams.items()} == {
        "train": 128,
        "dev": 16,
        "test_id": 16,
        "test_comp": 80,
    }


def test_dataset_matches_frozen_sha256_and_passes_leakage_checks():
    res = generate_dataset()
    assert res["scale"] >= 1.0
    for split in SPLITS:
        assert len(res["records"][split]) == TARGET_SIZES[split]
        text = jsonl_text(res["records"][split])
        assert sha256_text(text) == FROZEN_DATA_SHA256[f"{split}.jsonl"]

    checks = cross_split_checks(res["records"])
    assert checks["all_passed"] is True
    assert all(v == 0 for v in checks["cross_split_hash_intersection"].values())
    assert all(v == 0 for v in checks["cross_split_scenario_intersection"].values())
    assert checks["comp_pairs_absent_from_train_dev_test_id"] is True
    assert checks["comp_components_seen_in_train"] is True
