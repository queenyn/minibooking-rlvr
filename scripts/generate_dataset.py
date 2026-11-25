#!/usr/bin/env python3
"""Generate or byte-verify the frozen MiniBooking dataset splits.

Usage:
    python -X utf8 scripts/generate_dataset.py
    python -X utf8 scripts/generate_dataset.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minibooking_rlvr.data import FROZEN_DATA_SHA256, materialize_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or verify frozen MiniBooking JSONL splits.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument("--check", action="store_true", help="Verify byte-identical match with files on disk.")
    args = parser.parse_args(argv)

    res = materialize_dataset(args.data_dir, args.config_dir, check_only=args.check)
    if args.check:
        if not res["ok"]:
            print("Determinism check FAILED:", res["mismatched"])
            return 1
        for split in ("train", "dev", "test_id", "test_comp"):
            got = res["manifest"]["files"][f"data/{split}.jsonl"]["sha256"]
            want = FROZEN_DATA_SHA256[f"{split}.jsonl"]
            if got != want:
                print(f"Frozen SHA256 mismatch for {split}.jsonl: {got} != {want}")
                return 1
        print("Determinism & frozen SHA256 check PASSED (byte-identical across all 4 splits).")
        return 0

    for split in ("train", "dev", "test_id", "test_comp"):
        got = res["manifest"]["files"][f"data/{split}.jsonl"]["sha256"]
        want = FROZEN_DATA_SHA256[f"{split}.jsonl"]
        assert got == want, f"Frozen SHA256 mismatch for {split}.jsonl: {got} != {want}"
        print(f"Wrote data/{split}.jsonl  sha256={got}")
    print("Wrote data/data_manifest.json and configs/data_protocol.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
