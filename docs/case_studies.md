# Diagnostic Case Studies: Single-Step Failures & Closed-Loop Trajectories

This document curates concrete, verbatim examples from the frozen evaluation (`test_id` / `test_comp`) and 1-step MPC planning closed loop (`100` test tasks). All planning trajectories below can also be inspected step-by-step in the standalone interactive viewer: [demos/trajectory_viewer.html](../demos/trajectory_viewer.html) (backed by [results/sample_trajectories.json](../results/sample_trajectories.json)).

---

## Part A: Single-Step World-Model Failure Modes (Frozen Evaluation)

### Case 1 — Schema Violation Triggered by Domain Constraint Blindness (`test_id-00002`, Parse / Constraint Error)
- **Setup**: `rules = {price: 100, refund: 0/1, allow_change: True, fee: 0}`; `state_before = {status: "draft", day: 2, balance_cents: 0, paid_cents: 0, loss_cents: 100}`; `action = confirm()`.
- **Ground Truth**: Illegal action (`insufficient_balance`), so `changes = []`, `reversible = True`, `cost = 0.0`.
- **Observed Output (`G1`, `G2`, `G3`)**:
  ```json
  {"changes":[{"field":"balance_cents","value":-100},{"field":"paid_cents","value":100},{"field":"status","value":"confirmed"}],"reversible":false,"cost":0.0,"confidence":1.0}
  ```
- **Diagnostic Insight**: Valid JSON syntax, but rejected by the strict schema parser (`balance_cents = -100 < 0`), incurring `reward = -1.0` and full invalid penalties in the primary all-sample metric. The underlying failure is domain constraint blindness (failing to recognize `balance < price`), which only strict domain-aware validation catches.

### Case 2 — Compositional `cancel` Regression Under Unseen Rule Pairs (`test_comp-00034`, State-Delta Error)
- **Setup**: `rules = {price: 100, refund: 1/1, allow_change: True, fee: 20}` (held-out `test_comp` pair); `state_before = {status: "confirmed", day: 2, balance_cents: 300, paid_cents: 100, loss_cents: 0}`; `action = cancel()`.
- **Ground Truth**: Full refund (`balance_cents -> 400`, `paid_cents -> 0`, `status -> "draft"`), `reversible = True`, `cost = 0.0`.
- **Observed Outputs**:
  - **SFT**: Predicts `balance_cents = 400` correctly, but hallucinates `loss_cents = 50`, `reversible = False`, `cost = 0.5`.
  - **G1 / G2 / G3**: Omit `balance_cents` entirely and predict `loss_cents = 100`, `reversible = False`, `cost = 1.0` (collapsing onto the zero-refund cancel template).
- **Diagnostic Insight**: Explains the primary `test_comp` exact drop of RL vs. SFT (`0.8433 / 0.8400 / 0.8233` vs. `0.8767`): all three RL arms share the exact same non-empty `cancel` template bias on unseen `(refund, fee)` pairs.

### Case 3 — Reversibility Misclassification Despite Perfect State Delta (`test_id-00218`, Risk Error)
- **Setup**: `rules = {price: 100, refund: 1/1, allow_change: False, fee: 0}`; `state_before = {status: "draft", day: 2, balance_cents: 150, paid_cents: 0, loss_cents: 0}`; `action = confirm()`.
- **Ground Truth**: `changes = [balance_cents: 50, paid_cents: 100, status: "confirmed"]`, `reversible = True` (full refund allows lossless `cancel()`), `cost = 0.0`.
- **Observed Outputs**: SFT predicts `reversible = True` correctly; G1/G2/G3 get all three delta fields and `cost = 0.0` exact, yet predict `reversible = False` with `confidence = 1.0`.
- **Diagnostic Insight**: Demonstrates why tracking `exact` and `reversible macro-F1` separately is essential—a model can be 100% accurate on state deltas while misjudging multi-step recoverability.

---

## Part B: Closed-Loop 1-Step MPC Planning Trajectories (`w_risk=0, w_irr=0, w_conf=0.1, tau=0`)

### Case 4 — Illegal `create` Tie-Break Idling Loop (`test_comp-000` & `test_id-010`)
- **Task (`test_comp-000`)**: `start_state = {status: "draft", day: 2, balance_cents: 50, paid_cents: 0, loss_cents: 100}`, `goal = {status == "draft", day == 3}` (BFS shortest solution: 1 step `change_day(3)`).
- **G2 / SFT / G1 / oracle**: Correctly predict `create(3)` has `changes = []` (`score = 0.5`) while `change_day(3)` updates `day -> 3` (`score = 1.0`), completing the task in **1 step**.
- **G3**: Hallucinates that illegal `create(3)` on `draft` changes `day -> 3` (`score = 1.0`), tying with `change_day(3)` (`score = 1.0`). Deterministic tie-breaking selects candidate 0 (`create(3)`), which fails in the real environment (`wrong_status`, state unchanged), repeating identically for all 5 steps until `max_steps`.
- **Why it matters**: 30 of G3's 40 solvable failures follow this exact pattern, proving that G3's lower `irreversible_failure_rate` (`0.040` vs. G2 `0.080`) is a side effect of idling rather than cautious planning. On `test_id-010`, **only G2** (among learned models) avoids this trap and solves the task alongside `oracle`.

### Case 5 — 1-Step Horizon Stop Even Under Ground-Truth `oracle` (`test_id-038`)
- **Task (`test_id-038`)**: `start_state = {status: "confirmed", day: 2, balance_cents: 100, paid_cents: 200, loss_cents: 0}`, `goal = {status == "confirmed", day == 1}`, `allow_change = False` (BFS shortest solution: 4 steps `cancel -> clear -> create(1) -> confirm`).
- **Behavior**: Initial `progress = 0.5` (`status == "confirmed"` already holds). The only valid escape (`cancel()`) temporarily drops `status` to `"draft"` (`progress = 0.0 < 0.5`), while all no-op candidates score `0.5`. Because `max_score - current_progress = 0.0 <= tau = 0.0`, `oracle` (along with SFT, G1, G3) immediately stops at step 0 (`score_threshold`).
- **Takeaway**: Directly explains why `oracle` achieves `0.565` (`35/62`) and is a **diagnostic 1-step control**, not an upper bound—motivating the depth-2/3 Beam MPC upgrade in `v0.3`.

### Case 6 — Unsolvable Task & Zero Risk Weight Trap (`test_id-032`) vs. Clean Refusal (`test_id-000`)
- **Task (`test_id-032`)**: Unsolvable goal requiring both `status == "confirmed"` and a balance exceeding money conservation (`balance + paid + loss = const`).
- **Behavior**: Under the dev-selected frozen config (`w_irr = 0.0`), moving `draft -> confirmed` increases heuristic progress from `0.0` to `0.5` with zero irreversibility penalty. Consequently, SFT, G1, G2, G3, and even `oracle` all execute an irreversible `confirm()` at step 0 before stopping at step 1 (`dangerous_rate = 0.040` overall). By contrast, on `test_id-000` (where no single step raises progress), all six predictors cleanly refuse at step 0.
- **Takeaway**: Exposes the concrete limitation of selecting planner weights on a tiny 20-task dev set (`w_irr = 0.0`) and motivates Pareto frontier tuning with non-zero `(w_risk, w_irr)` in `v0.3`.
