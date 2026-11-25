# Roadmap: Evolving MiniBooking-RLVR Beyond v0.1

The `v0.1` release establishes a clean, modular, byte-reproducible prototype with honest single-seed empirical diagnostics. To elevate this benchmark and training stack into a standout research-engineering showcase, the following targeted milestones directly address the concrete failure modes uncovered in `v0.1`:

## Milestone v0.2 — Confidence Calibration & Statistical Rigor

1. **Fix `confidence = 1.0` Calibration Collapse**:
   - *Diagnosis*: Constant `confidence=1.0` SFT targets caused 100% of valid outputs to emit `1.0`, making the Brier reward term degenerate into `1 - exact`.
   - *Plan*: Introduce (a) an empirical out-of-fold / perturbed confidence SFT warm-start, and (b) a token-logprob derived confidence estimator (`exp(mean_logprob)` over `changes` tokens) compared against verbalized JSON confidence using ECE and Brier decomposition (reliability vs. resolution).
2. **Multi-Seed Variance Quantification & Fixed-Lambda Ablations**:
   - *Diagnosis*: Single-seed `bf16` non-determinism created `~0.06` `dev[:100]` noise, and G3 triggered only once (`lambda=1e-4`), leaving entropy regularization untested.
   - *Plan*: Run 3-seed GRPO sweeps across `G1`, `G2`, and fixed non-zero `lambda_entropy in {1e-3, 1e-2, 5e-2}` (plus a tighter `H_low_ratio` controller sweep) and report mean +/- std and paired bootstrap confidence intervals.
3. **Hard-Negative `cancel` Arithmetic Recovery**:
   - *Diagnosis*: RL degraded non-empty `cancel` exact accuracy on `test_comp` by omitting `balance_cents`.
   - *Plan*: Add field-specific partial-credit diagnostics and optional scratchpad/CoT-verifiable reasoning comparison (`enable_thinking=True` vs `False`) under controlled token budgets.

## Milestone v0.3 — Multi-Step Risk-Aware Planning & Trajectory Replay

1. **Depth-`k` Beam / Rollout MPC Planner**:
   - *Diagnosis*: The 1-step `oracle` control achieved `0.565` (`35/62`) solely because 27 solvable tasks required multi-step intermediate states where 1-step heuristic progress did not strictly improve.
   - *Plan*: Extend `minibooking_rlvr.planner` with a depth-2/3 risk-aware rollout planner and explicit `clear` candidate coverage, evaluating Pareto frontiers of `(solvable_success_rate, irreversible_failure_rate)` across non-zero `(w_risk, w_irr)` regimes.
2. **Interactive Trajectory & Failure Replay Tool**:
   - Export self-contained HTML reports inspecting world-model predicted deltas vs. true state transitions step-by-step on planning trajectories (highlighting idle tie-break loops, myopic stops, and irreversible traps).

## Milestone v0.4 — Pluggable Verifiable State-Machine Environments

- Extract a minimal `VerifiableWorldEnv` protocol (`step_raw`, `extract_changes`, `is_reversible`, `compute_cost`) so MiniBooking sits alongside additional risk-sensitive agent environments (e.g., inventory allocation, flight/hotel multi-leg itineraries, and access-control workflows) while reusing the exact same GRPO trainer, H/D/S/C diagnostics, and MPC planner.
