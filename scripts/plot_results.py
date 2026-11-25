#!/usr/bin/env python3
"""Generate clean publication-style English figures from ``results/summary_metrics.json``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "results" / "summary_metrics.json"
FIG_DIR = REPO_ROOT / "results" / "figures"

PALETTE = {
    "SFT": "#64748B",
    "G1": "#2563EB",
    "G2": "#0D9488",
    "G3": "#D97706",
    "null": "#94A3B8",
    "oracle": "#0F172A",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def plot_frozen_eval(data: dict) -> None:
    models = ["SFT", "G1", "G2", "G3"]
    metrics = [
        ("test_id\nDelta Exact", [data["frozen_eval"]["test_id"][m]["exact"] for m in models]),
        ("test_comp\nDelta Exact", [data["frozen_eval"]["test_comp"][m]["exact"] for m in models]),
        ("test_comp\nReversible Macro-F1", [data["frozen_eval"]["test_comp"][m]["rev_macro_f1"] for m in models]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    x = np.arange(len(metrics))
    width = 0.19

    for idx, model in enumerate(models):
        vals = [m[1][idx] for m in metrics]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=model, color=PALETTE[model], zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1E293B",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Primary All-Sample Score (n = 300 / split)", fontsize=10.5)
    ax.set_title("Frozen Greedy Evaluation: ID vs. Compositional Metric Divergence (Single Seed)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    _style_ax(ax)

    fig.text(
        0.5,
        0.015,
        "Single-seed run (Qwen3-1.7B + LoRA, pure-PyTorch GRPO).\n"
        "G3 triggered entropy feedback once (steps 171-180, max grad ratio 7.7e-5); G3 ≈ G2 within bf16 run noise.",
        ha="center",
        fontsize=8.3,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / "fig1_frozen_eval.png", dpi=200)
    plt.close(fig)


def plot_diagnostics_and_controller(data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=200)

    # Panel A: Zero-variance group decomposition across 800 total groups (200 steps x 4 prompts)
    groups = ["G1\n(DeltaF1)", "G2\n(Composite)", "G3\n(Composite + Ctrl)"]
    zvs = data["grpo_training"]["zero_var_split"]
    keys = ["G1", "G2", "G3"]
    eff = [zvs[k]["n_effective"] / 800 for k in keys]
    ac = [zvs[k]["n_all_correct"] / 800 for k in keys]
    aw = [zvs[k]["n_all_wrong"] / 800 for k in keys]
    oth = [zvs[k]["n_other_same"] / 800 for k in keys]

    x = np.arange(len(groups))
    w = 0.50
    ax1.bar(x, eff, width=w, label="Effective (Non-Zero Var)", color="#0D9488", zorder=3)
    ax1.bar(x, ac, width=w, bottom=eff, label="Zero-Var: All-Correct (1.0)", color="#94A3B8", zorder=3)
    b2 = np.array(eff) + np.array(ac)
    ax1.bar(x, oth, width=w, bottom=b2, label="Zero-Var: Other Tie", color="#CBD5E1", zorder=3)
    b3 = b2 + np.array(oth)
    ax1.bar(x, aw, width=w, bottom=b3, label="Zero-Var: All-Wrong (0.0)", color="#EF4444", zorder=3)

    for i, k in enumerate(keys):
        ax1.text(i, eff[i] / 2, f"{zvs[k]['n_effective']}/800\n({eff[i]*100:.1f}%)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if zvs[k]["n_all_wrong"] > 0:
            ax1.text(i, b3[i] + aw[i] / 2, f"{zvs[k]['n_all_wrong']}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=9.5)
    ax1.set_ylim(0, 1.24)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel("Fraction of 800 Prompt Groups (200 steps)", fontsize=10)
    ax1.set_title("A. GRPO Zero-Variance Group Breakdown", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    _style_ax(ax1)

    # Panel B: Dev[:100] full_exact curve + G3 lambda_entropy spike
    curve = data["grpo_training"]["dev100_full_exact_curve"]
    steps = curve["steps"]
    for k in ["G1", "G2", "G3"]:
        ax2.plot(steps, curve[k], marker="o", linewidth=2.0, label=f"{k} dev[:100] full_exact", color=PALETTE[k], zorder=4)
    ax2.set_xlabel("GRPO Optimizer Step", fontsize=10)
    ax2.set_ylabel("dev[:100] Greedy Full Exact", fontsize=10)
    ax2.set_ylim(0.70, 0.99)
    ax2.set_xlim(40, 210)
    ax2.set_xticks(steps)
    _style_ax(ax2)

    ax2_r = ax2.twinx()
    lam_steps = np.arange(1, 201)
    lam_vals = np.where((lam_steps >= 171) & (lam_steps <= 180), 1e-4, 0.0)
    ax2_r.fill_between(lam_steps, lam_vals, step="mid", color="#F59E0B", alpha=0.28, label="G3 lambda_entropy (1 trigger)")
    ax2_r.plot(lam_steps, lam_vals, drawstyle="steps-mid", color="#B45309", linewidth=1.2, linestyle="--")
    ax2_r.set_ylabel("G3 lambda_entropy (log floor = 1e-4)", fontsize=9.5, color="#B45309")
    ax2_r.set_ylim(0.0, 0.0005)
    ax2_r.tick_params(axis="y", colors="#B45309")
    ax2_r.spines["top"].set_visible(False)

    ax2.set_title("B. Dev Trajectory & G3 Controller Trigger Boundary", fontsize=11, fontweight="bold", pad=10)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.2, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_zero_var_and_controller.png", dpi=200)
    plt.close(fig)


def plot_planning_closed_loop(data: dict) -> None:
    predictors = ["null", "SFT", "G1", "G2", "G3", "oracle"]
    overall = data["planning_closed_loop"]["overall"]
    sol_rates = [overall[p]["solvable_success_rate"] for p in predictors]
    irr_rates = [overall[p]["irreversible_failure_rate"] for p in predictors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.7), dpi=200)
    x = np.arange(len(predictors))
    colors = [PALETTE[p] for p in predictors]

    bars1 = ax1.bar(x, sol_rates, color=colors, width=0.58, zorder=3)
    for b, v in zip(bars1, sol_rates):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(predictors, fontsize=9.5)
    ax1.set_ylim(0.0, 0.68)
    ax1.set_ylabel("Solvable Success Rate (n = 62 solvable)", fontsize=10)
    ax1.set_title("A. Solvable Task Completion", fontsize=11, fontweight="bold")
    _style_ax(ax1)

    bars2 = ax2.bar(x, irr_rates, color=colors, width=0.58, zorder=3)
    for b, v in zip(bars2, irr_rates):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(predictors, fontsize=9.5)
    ax2.set_ylim(0.0, 0.13)
    ax2.set_ylabel("Irreversible Failure Rate (n = 100 all tasks)", fontsize=10)
    ax2.set_title("B. Irreversible Failure Rate", fontsize=11, fontweight="bold")
    _style_ax(ax2)

    fig.suptitle("Risk-Aware 1-Step Lookahead MPC Planning Closed Loop (100 Test Tasks)", fontsize=12, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.015,
        "Frozen dev-selected config: w_risk=0.0, w_irr=0.0, w_conf=0.1, tau=0.0.\n"
        "Oracle is a 1-step ground-truth control (NOT a multi-step upper bound); G3 lower irr-failure reflects idle tie-break loops.",
        ha="center",
        fontsize=8.2,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(FIG_DIR / "fig3_planning_closed_loop.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    plot_frozen_eval(data)
    plot_diagnostics_and_controller(data)
    plot_planning_closed_loop(data)
    print(f"Generated 3 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
