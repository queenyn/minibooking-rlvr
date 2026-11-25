#!/usr/bin/env python3
"""Run the risk-aware 1-step lookahead MPC planning closed loop (CPU baselines or GPU checkpoints).

Example (pure CPU, runs in <1 second):
    python -X utf8 scripts/run_planning_eval.py --split test --predictors oracle null --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minibooking_rlvr.planner import (
    breakdown_episodes,
    env_step,
    generate_planning_tasks,
    null_predict_batch,
    oracle_predict_batch,
    propose_candidates,
    render_planner_table,
    run_tasks_batched,
    summarize_episodes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MiniBooking 1-step MPC planning evaluation.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--planner-config", type=Path, default=REPO_ROOT / "configs" / "planner_config.json")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--predictors", nargs="+", choices=("oracle", "null"), default=["oracle", "null"])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "planner")
    parser.add_argument("--no-save", action="store_true", help="Print table without writing files.")
    args = parser.parse_args(argv)

    cfg = json.loads(args.planner_config.read_text(encoding="utf-8"))
    weights = cfg["weights"]
    tau = float(cfg["tau"])
    tasks_by_split, manifest = generate_planning_tasks(args.data_dir)
    tasks = tasks_by_split[args.split]

    predictors = {"oracle": oracle_predict_batch, "null": null_predict_batch}
    summaries = {}
    breakdowns = {}
    for name in args.predictors:
        out = run_tasks_batched(
            tasks,
            propose_candidates,
            predictors[name],
            env_step,
            weights=weights,
            tau=tau,
            max_steps=int(cfg["max_steps"]),
            batch_size=int(cfg["batch_size"]),
            max_candidates=int(cfg["max_candidates"]),
            predictor_kind="cpu",
        )
        summaries[name] = summarize_episodes(out["episodes"], tasks, out["batches"])
        breakdowns[name] = breakdown_episodes(out["episodes"], tasks)

    table_md = render_planner_table(summaries, args.predictors)
    print(f"Planning split={args.split} | weights={weights} | tau={tau}")
    print(table_md)

    if not args.no_save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"metrics_{args.split}.json").write_text(
            json.dumps({"manifest": manifest, "overall": summaries, "breakdown": breakdowns}, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / f"table_{args.split}.md").write_text(table_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
