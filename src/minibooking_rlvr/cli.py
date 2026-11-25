"""Command-line interface for MiniBooking-RLVR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from minibooking_rlvr.data import FROZEN_DATA_SHA256, materialize_dataset
from minibooking_rlvr.planner import (
    env_step,
    generate_planning_tasks,
    null_predict_batch,
    oracle_predict_batch,
    propose_candidates,
    render_planner_table,
    run_tasks_batched,
    summarize_episodes,
)


def _cmd_data(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    config_dir = Path(args.config_dir) if args.config_dir else None
    res = materialize_dataset(data_dir, config_dir, check_only=args.check)
    if args.check:
        if not res["ok"]:
            print("Determinism check FAILED:", res["mismatched"])
            return 1
        print("Determinism check PASSED: all generated files match disk byte-for-byte.")
        return 0
    for split in ("train", "dev", "test_id", "test_comp"):
        got = res["manifest"]["files"][f"data/{split}.jsonl"]["sha256"]
        want = FROZEN_DATA_SHA256[f"{split}.jsonl"]
        status = "MATCH" if got == want else "DIFF"
        print(f"  data/{split}.jsonl  sha256={got[:16]}...  frozen={status}")
    return 0


def _cmd_plan_cpu(args: argparse.Namespace) -> int:
    tasks_by_split, _ = generate_planning_tasks(args.data_dir)
    tasks = tasks_by_split[args.split]
    weights = {"w_risk": args.w_risk, "w_irr": args.w_irr, "w_conf": args.w_conf}
    predictors = {"oracle": oracle_predict_batch, "null": null_predict_batch}
    summaries = {}
    for name in args.predictors:
        out = run_tasks_batched(
            tasks,
            propose_candidates,
            predictors[name],
            env_step,
            weights=weights,
            tau=args.tau,
            predictor_kind="cpu",
        )
        summaries[name] = summarize_episodes(out["episodes"], tasks, out["batches"])
    print(render_planner_table(summaries, args.predictors))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minibooking-rlvr",
        description="MiniBooking-RLVR benchmark, diagnostics, and risk-aware MPC planning CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Generate or byte-verify the frozen dataset splits.")
    p_data.add_argument("--data-dir", default="data", help="Output directory for JSONL splits.")
    p_data.add_argument("--config-dir", default="configs", help="Output directory for data_protocol.json.")
    p_data.add_argument("--check", action="store_true", help="Verify byte-identical reproducibility without writing.")
    p_data.set_defaults(func=_cmd_data)

    p_plan = sub.add_parser("plan-cpu", help="Run CPU diagnostic planning baselines (oracle / null).")
    p_plan.add_argument("--data-dir", default="data", help="Frozen dataset directory.")
    p_plan.add_argument("--split", choices=("dev", "test"), default="test", help="Planning benchmark split.")
    p_plan.add_argument("--predictors", nargs="+", choices=("oracle", "null"), default=["oracle", "null"])
    p_plan.add_argument("--w-risk", type=float, default=0.0)
    p_plan.add_argument("--w-irr", type=float, default=0.0)
    p_plan.add_argument("--w-conf", type=float, default=0.1)
    p_plan.add_argument("--tau", type=float, default=0.0)
    p_plan.add_argument("--json-out", default=None, help="Optional JSON summary output path.")
    p_plan.set_defaults(func=_cmd_plan_cpu)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
