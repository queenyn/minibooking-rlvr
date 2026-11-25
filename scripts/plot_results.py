#!/usr/bin/env python3
"""Generate publication-style English figures from ``results/summary_metrics.json`` and ``results/training_dynamics.json``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "results" / "summary_metrics.json"
DYNAMICS_PATH = REPO_ROOT / "results" / "training_dynamics.json"
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


def _rolling_mean(values: list[float | None], window: int = 15) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    buf: list[float] = []
    for idx, val in enumerate(values, start=1):
        if val is not None:
            buf.append(float(val))
        if len(buf) > window:
            buf.pop(0)
        if buf:
            xs.append(idx)
            ys.append(sum(buf) / len(buf))
    return np.asarray(xs), np.asarray(ys)


def plot_frozen_eval(data: dict) -> None:
    models = ["SFT", "G1", "G2", "G3"]
    metrics = [
        ("test_id\nDelta Exact", [data["frozen_eval"]["test_id"][m]["exact"] for m in models]),
        ("test_comp\nDelta Exact", [data["frozen_eval"]["test_comp"][m]["exact"] for m in models]),
        ("test_comp\nReversible Macro-F1", [data["frozen_eval"]["test_comp"][m]["rev_macro_f1"] for m in models]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=200)
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


def plot_training_dynamics(dyn: dict, summary: dict) -> None:
    """2x2 panel of real 200-step GRPO training dynamics and 20-step LR stability sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), dpi=200)
    ax_kl, ax_ent, ax_eff, ax_lr = axes.flat
    groups = ["G1", "G2", "G3"]

    # Panel A: k3 KL divergence (15-step rolling mean + faint raw)
    for g in groups:
        rows = dyn["groups"][g]
        steps = [r["step"] for r in rows]
        raw_kl = [r["kl"] for r in rows]
        rx, ry = _rolling_mean(raw_kl, window=15)
        ax_kl.plot(steps, raw_kl, color=PALETTE[g], alpha=0.16, linewidth=0.9)
        ax_kl.plot(rx, ry, color=PALETTE[g], linewidth=2.0, label=f"{g} (15-step avg)")
    ax_kl.set_title("A. k3 KL Divergence to SFT Reference (beta_kl = 0.04)", fontsize=10.8, fontweight="bold")
    ax_kl.set_xlabel("Optimizer Step", fontsize=9.5)
    ax_kl.set_ylabel("Mean Sequence k3 KL", fontsize=9.5)
    ax_kl.set_xlim(1, 200)
    ax_kl.set_ylim(0.0, 0.095)
    ax_kl.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_ax(ax_kl)

    # Panel B: Effective-group token entropy H_eff_mean (T=1.0) vs H_ref / H_low
    ctrl_info = summary["grpo_training"]["g3_controller"]
    for g in groups:
        rows = dyn["groups"][g]
        raw_h = [r["H_eff_mean"] for r in rows]
        rx, ry = _rolling_mean(raw_h, window=15)
        ax_ent.plot(rx, ry, color=PALETTE[g], linewidth=2.0, label=f"{g} H_eff (15-step avg)")
    ax_ent.axhline(ctrl_info["H_ref"], color="#334155", linestyle=":", linewidth=1.3, label=f"H_ref = {ctrl_info['H_ref']:.4f}")
    ax_ent.axhline(ctrl_info["H_low"], color="#DC2626", linestyle="--", linewidth=1.3, label=f"H_low = {ctrl_info['H_low']:.4f}")
    ax_ent.axvline(170, color="#B45309", linestyle="-.", linewidth=1.1, alpha=0.8)
    ax_ent.text(167, 0.0465, "G3 trigger\n(step 170)", ha="right", va="top", fontsize=7.8, color="#B45309")
    ax_ent.set_title("B. Effective-Group Token Entropy (T = 1.0)", fontsize=10.8, fontweight="bold")
    ax_ent.set_xlabel("Optimizer Step", fontsize=9.5)
    ax_ent.set_ylabel("Token Entropy (nats)", fontsize=9.5)
    ax_ent.set_xlim(1, 200)
    ax_ent.set_ylim(0.015, 0.050)
    ax_ent.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper left")
    _style_ax(ax_ent)

    # Panel C: 20-step rolling effective groups per step (out of 4)
    for g in groups:
        rows = dyn["groups"][g]
        raw_eff = [r["n_effective"] for r in rows]
        rx, ry = _rolling_mean(raw_eff, window=20)
        ax_eff.plot(rx, ry, color=PALETTE[g], linewidth=2.0, label=f"{g} (mean={np.mean(raw_eff):.2f}/4)")
    ax_eff.set_title("C. Rolling Effective Gradient Groups per Step (out of 4)", fontsize=10.8, fontweight="bold")
    ax_eff.set_xlabel("Optimizer Step", fontsize=9.5)
    ax_eff.set_ylabel("Effective Groups / Step (20-step avg)", fontsize=9.5)
    ax_eff.set_xlim(1, 200)
    ax_eff.set_ylim(0.0, 2.6)
    ax_eff.legend(frameon=False, fontsize=8.2, loc="upper right")
    _style_ax(ax_eff)

    # Panel D: Pre-formal 20-step LR stability sweep (G1, 5e-6 vs 1e-5 vs 2e-5)
    lr_keys = ["lr_5e-6", "lr_1e-5", "lr_2e-5"]
    lr_labels = ["5e-6\n(stable)", "1e-5\n(selected)", "2e-5\n(KL > 0.05)"]
    kl_max_vals = [dyn["lr_sweep"][k]["kl_max"] for k in lr_keys]
    bar_colors = ["#64748B", "#0D9488", "#EF4444"]
    x_lr = np.arange(len(lr_keys))
    bars = ax_lr.bar(x_lr, kl_max_vals, width=0.50, color=bar_colors, zorder=3)
    ax_lr.axhline(0.05, color="#DC2626", linestyle="--", linewidth=1.3, label="Stability cap (kl_max < 0.05)")
    for b, v in zip(bars, kl_max_vals):
        ax_lr.text(b.get_x() + b.get_width() / 2, v + 0.0022, f"{v:.4f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax_lr.set_xticks(x_lr)
    ax_lr.set_xticklabels(lr_labels, fontsize=9.5)
    ax_lr.set_ylim(0.0, 0.075)
    ax_lr.set_ylabel("Peak k3 KL over 20 Steps", fontsize=9.5)
    ax_lr.set_title("D. Pre-Formal Learning Rate Stability Sweep (G1)", fontsize=10.8, fontweight="bold")
    ax_lr.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_ax(ax_lr)

    fig.suptitle("Pure-PyTorch GRPO Training Dynamics & Hyperparameter Verification (200 Steps)", fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIG_DIR / "fig4_training_dynamics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    plot_frozen_eval(data)
    plot_diagnostics_and_controller(data)
    plot_planning_closed_loop(data)
    if DYNAMICS_PATH.exists():
        dyn = json.loads(DYNAMICS_PATH.read_text(encoding="utf-8"))
        plot_training_dynamics(dyn, data)
        print(f"Generated 4 figures in {FIG_DIR}")
    else:
        print(f"Generated 3 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
