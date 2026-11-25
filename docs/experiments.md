# Verified Experiment Report (v0.1 Single-Seed Prototype)

All metrics in this report are sourced from `results/summary_metrics.json` and correspond to a single-seed (`seed=20261025`) prototype run on `Qwen/Qwen3-1.7B` (`revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`) using pure-PyTorch LoRA SFT and hand-written GRPO (without TRL or vLLM).

## 1. Benchmark Splits & Leakage Protocol

- **Splits**: `train = 2,400`, `dev = 300`, `test_id = 300`, `test_comp = 300` (240 scenario families total: `128 / 16 / 16 / 80`).
- **Compositional holdout**: `(refund=1/2, change_fee=0)` and `(refund=1/1, change_fee=20)` appear exclusively in `test_comp`, while each individual factor appears in `train`.
- **Behavioral caveat on `test_comp`**: When `allow_change=False`, `change_fee_cents` has no operational effect. True behavioral compositional shift occurs on the `allow_change=True` subset (`n = 178 / 300`), which is reported separately below.

## 2. Stage 1 — LoRA SFT Initialization

- **Setup**: 600 stratified examples from `train` (disjoint from RL pool and eval splits), LoRA (`r=16, alpha=32, dropout=0.0` on all 7 linear projections across 28 layers), `lr=1e-4`, effective batch size 8, 150 steps, completion-only loss masking (`20,971` supervised tokens vs. `359,802` masked prompt tokens).
- **`dev[:100]` greedy progression**:
  - Base model: `json_valid = 0.96`, `changes_exact = 0.52`, `full_exact = 0.44`
  - Step 75: `json_valid = 1.00`, `changes_exact = 0.77`, `full_exact = 0.73`
  - Step 150: `json_valid = 1.00`, `changes_exact = 0.92`, `full_exact = 0.91`
- The merged `base + SFT adapter` serves identically as both the GRPO policy starting point and the reference model (`disable_adapter()`).

## 3. Stage 2 — Offline Difficulty Filtering & Pure-PyTorch GRPO (G1 / G2 / G3)

### Offline K=4 Filtering

Sampling `K=4` (`T=0.7`) from the frozen SFT model on the remaining 1,800 training prompts identified 1,098 all-correct zero-variance prompts (including all 254 `inspect` prompts). Filtering reduced the pool to 702 hard prompts and lowered initial zero-variance group rate from `63.89%` to `7.41%`. Ninety-eight high-signal `cancel` prompts were repeated once (minimum separation 97 steps) to form a fixed 800-slot training schedule (`200 steps x 4 prompts/step`) shared identically across G1, G2, and G3.

### Three-Arm Experimental Matrix

- **G1**: `R = DeltaF1` (invalid JSON/schema = `-1.0`), `lambda_entropy = 0.0`.
- **G2**: `R = 0.60 * DeltaF1 + 0.25 * RiskScore + 0.15 * (1 - Brier)`, `lambda_entropy = 0.0`.
- **G3**: Identical composite reward to G2, plus loss term `-lambda_entropy * H_reg(T=1.0)` driven by `EntropyController` (`H_ref=0.03256`, `H_low=0.02605`, `exact_threshold=0.9`, log mode `floor=1e-4`, `lambda_max=0.2`, updated every 10 steps). Note: `lambda_entropy` is distinct from the fixed KL penalty `beta_kl = 0.04`.

### Pre-Formal Controller Gradient-Scale Calibration

Before finalizing G3's `EntropyController`, per-component gradient L2 norms were measured across 18 effective-gradient steps (`Qwen3-1.7B` LoRA, `lr=1e-5`, G2 reward):

| Measured Quantity (18 effective steps) | Median | Implication for Controller Design |
| :--- | :---: | :--- |
| Policy gradient norm `‖∇_θ L_pg‖` | `0.8963` | Baseline update scale |
| KL penalty gradient norm `‖∇_θ (beta_kl · KL)‖` | `0.0209` | Small anchor (~2.3% of policy norm) |
| Entropy gradient norm `‖∇_θ H(T=1.0)‖` (`T=0.7`: `0.5387`) | `0.4809` | Roughly half the policy norm; `cos(∇L_pg, ∇H) ≈ 0.087` (nearly orthogonal) |
| Parity coefficient `lambda_eq = ‖∇L_pg‖ / ‖∇H(T=1.0)‖` | `1.8511` | Additive `1e-4` step needs ~18,511 updates to reach parity; old `lambda_max = 0.01` is only `0.54%` of policy norm |
| Finalized log schedule (`floor=1e-4`, `×2`, `lambda_max=0.2`) | `10.8%` cap | Reaches `lambda_max = 0.2` (~10.8% relative gradient norm) in 12 consecutive triggers within a 200-step budget |

### Training Diagnostics & Run-to-Run Noise

| Group | `dev[:100]` full_exact (50 / 100 / 150 / 200) | Effective Groups / 800 | Zero-Var All-Correct | Zero-Var All-Wrong (`0.0`) | Zero-Var Other Tie |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **G1** | 0.77 / 0.86 / 0.84 / 0.89 | 162 (20.3%) | 495 | 47 | 96 |
| **G2** | 0.77 / 0.87 / 0.90 / 0.89 | 211 (26.4%) | 479 | 0 | 110 |
| **G3** | 0.78 / 0.81 / 0.84 / 0.86 | 196 (24.5%) | 485 | 0 | 119 |

- **G3 Controller Boundary**: Out of 20 controller updates, G3 triggered **once** at step 170 (`H_ema=0.02533 < H_low=0.02605`, window exact `0.70 < 0.9`), setting `lambda_entropy = 1e-4` for steps 171–180 before decaying back to `0.0` at step 180. Maximum measured entropy-to-policy gradient norm ratio was `7.67e-5`.
- **Noise Floor**: G2 and G3 received identical rewards for steps 1–4 and diverged at step 5 while `lambda_entropy == 0.0` due to GPU `bf16` non-determinism, yielding a `0.06` gap on `dev[:100]` at step 100 (`0.87` vs `0.81`). Differences of `<= 0.03` between G1/G2/G3 cannot be attributed to the controller on a single seed.

## 4. Stage 3 — Frozen Greedy Evaluation

### Primary All-Sample Results (`n = 300` each)

| Split | Model | Parse Cov. | Exact | DeltaF1 | Full Exact | Rev. Macro-F1 | Rev. Acc | Cost MAE | Brier | `frac(conf == 1.0)` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_id` | SFT | 1.0000 | 0.8833 | 0.9282 | 0.8700 | 0.9167 | 0.9467 | 0.0388 | 0.1167 | 1.0000 |
| `test_id` | G1 | 0.9933 | 0.9133 | 0.9531 | 0.9067 | 0.9409 | 0.9567 | 0.0367 | 0.0867 | 1.0000 |
| `test_id` | G2 | 0.9933 | 0.9033 | 0.9498 | 0.8933 | 0.9324 | 0.9500 | 0.0363 | 0.0967 | 1.0000 |
| `test_id` | G3 | 0.9933 | 0.8967 | 0.9398 | 0.8833 | 0.9409 | 0.9567 | 0.0360 | 0.1033 | 1.0000 |
| `test_comp` | SFT | 0.9967 | 0.8767 | 0.9380 | 0.7267 | 0.5163 | 0.7533 | 0.0430 | 0.1233 | 1.0000 |
| `test_comp` | G1 | 0.9967 | 0.8433 | 0.9329 | 0.7433 | 0.7256 | 0.8333 | 0.0497 | 0.1567 | 1.0000 |
| `test_comp` | G2 | 0.9967 | 0.8400 | 0.9365 | 0.7567 | 0.7401 | 0.8400 | 0.0538 | 0.1600 | 1.0000 |
| `test_comp` | G3 | 0.9933 | 0.8233 | 0.9146 | 0.7333 | 0.7435 | 0.8400 | 0.0622 | 0.1767 | 1.0000 |

### Compositional Behavioral Subset (`test_comp`, `allow_change=True`, `n = 178`)

| Model | Exact | Full Exact | Rev. Macro-F1 |
| :--- | :---: | :---: | :---: |
| SFT | 0.8820 | 0.7416 | 0.5338 |
| G1 | 0.8427 | 0.7247 | 0.6289 |
| G2 | 0.8315 | 0.7191 | 0.6142 |
| G3 | 0.8371 | 0.7191 | 0.6239 |

- **`inspect` Regression Check**: Exact accuracy is `1.0000` across `dev (37)`, `test_id (37)`, and `test_comp (37)` for SFT, G1, G2, and G3 despite excluding all 254 `inspect` prompts from GRPO training.
- **Root Cause of `test_comp` Divergence**: RL loses `10–16` exact matches vs. SFT predominantly on non-empty `cancel` transitions (omitting `balance_cents`), while gaining `+23–26` correct irreversible predictions on the `(refund=1/2, fee=0)` rule pair (`8/44 -> 31–34/44`).

## 5. Stage 4 — 1-Step Lookahead MPC Planning Closed Loop

- **Task Suite**: 100 test tasks (50 from `test_id`, 50 from `test_comp`; 62 BFS-solvable within 5 steps, 38 unsolvable).
- **Frozen Dev Config**: Selected via 18-point grid search (`w_risk in {0, 0.5, 1.0} × w_irr in {0, 0.25, 0.5} × tau in {0, 0.05}`, fixed `w_conf=0.1`; full 18-config breakdown in [results/planner_dev_grid.json](../results/planner_dev_grid.json)) on 20 dev tasks (`7 solvable / 13 unsolvable`), yielding `w_risk=0.0, w_irr=0.0, w_conf=0.1, tau=0.0` (dev objective `0.5571`).
- **Audit-Grade Offline Verification**: All 2,400 frozen evaluation raw completions (`4 models × 300 test_id + 4 models × 300 test_comp`) are archived in `results/predictions/` and verified from scratch on CPU via `python -X utf8 scripts/reproduce_eval_tables.py --verify`.
- **Key Takeaway**: Under this single frozen configuration, G2 achieved the highest solvable success rate (`0.500`, `31/62`) below the 1-step ground-truth `oracle` control (`0.565`, `35/62`), while `null` scored `0.000` (`1.000` refusal). Because dev selection used only 20 tasks and weights zeroed out `w_risk` and `w_irr`, stronger multi-step risk-aware planning requires the `v0.2–v0.3` upgrades tracked in [roadmap.md](roadmap.md).
