# Architecture & Module Boundaries

`minibooking-rlvr` is structured around six decoupled capability packages under `src/minibooking_rlvr/` so that domain rules, verifiable reward math, pure-PyTorch RL optimization, evaluation, and closed-loop planning can be understood, tested, and extended independently.

## Package Layout

| Module | Responsibilities | Dependencies | Key Invariants |
| :--- | :--- | :--- | :--- |
| `minibooking_rlvr.env` | Domain `State` / `Rules` / `Action`, deterministic `step_raw`, delta diff/cost, depth-3 BFS `is_reversible`, `label_transition` | Pure stdlib | Exact integer-cent arithmetic; `step_raw` never mutates input `State`; `reversibility` calls only `step_raw` (never `label_transition`). |
| `minibooking_rlvr.rewards` | Strict JSON object parser, schema validator (`PredictionError`), order-invariant `canonical_changes`, `DeltaF1`, `RiskScore`, `Brier`, `compute_reward` | `env.state` | No silent coercion; rejects fences, duplicate keys, NaN/Inf; pure deterministic functions. |
| `minibooking_rlvr.data` | Scenario-family partitioning, random-walk state discovery, SHA256 input dedupe, cross-split leakage checks, frozen `minibooking_prompt_v1` | `env`, `rewards` | Families partitioned before sampling; `build_messages` accesses strictly `(rules, state_before, action)`. |
| `minibooking_rlvr.rl` | Pure-PyTorch GRPO math (`grpo.py`), H/D/S/C diagnostics & zero-var split (`diagnostics.py`), `EntropyController` (`controller.py`), offline filtering (`filtering.py`) | `rewards`, `numpy`, lazy `torch` (only in `grpo.py`) | `masked_token_entropy` is detached; `seq_entropy_with_grad` preserves analytical gradient; controller enforces strict consecutive steps and config matching on load. |
| `minibooking_rlvr.eval` | Primary all-sample & valid-only metrics, reversible macro-F1, confidence histograms, compositional subsets, batched HF greedy inference | `rewards`, `data.prompt`, lazy `torch` | Invalid outputs are never silently dropped from primary metrics. |
| `minibooking_rlvr.planner` | Isolated candidate proposer, env/oracle/null adapters, 1-step MPC replanning loop, BFS task generator, closed-loop metrics | `env` (in `adapter`/`tasks` only) | `candidates.py` is AST-tested to never import `env` internals or perform oracle legality filtering. |

## End-to-End Data & Control Flow

1. **Data Construction (`scripts/generate_dataset.py`)**:
   `assign_families()` -> `build_pool()` via `step_raw` -> `label_transition()` (adds depth-3 BFS reversibility) -> `cross_split_checks()` -> LF-normalized JSONL splits matching `FROZEN_DATA_SHA256`.
2. **SFT & Offline Filtering (`scripts/run_sft_train.py`, `rl/filtering.py`)**:
   `build_prompt()` + `gold_completion()` -> completion-only cross-entropy loss (`prompt` tokens masked to `-100`) -> sample `K=4` completions on remaining RL pool -> drop saturated `all_correct` prompts before GRPO.
3. **Hand-Written GRPO Loop (`scripts/run_grpo_train.py`, `rl/grpo.py`)**:
   Merge SFT LoRA into base weights -> attach fresh RL LoRA -> sample `K=4` per prompt -> score via `make_sample_row()` -> `compute_group_advantages()` -> reference logprobs via `model.disable_adapter()` -> `compute_grpo_objective()` (`pg + beta_kl * k3_kl - lambda_entropy * H_reg`) -> update online `group_stats` / `batch_stats` and `EntropyController.observe()`.
4. **Frozen Evaluation (`scripts/run_frozen_eval.py`, `eval/metrics.py`)**:
   Greedy decode (`max_new_tokens=256`, `batch_size=16`) -> `score_raw()` -> `evaluate_model_outputs()` -> dual-view tables & subset decompositions (`allow_change`, `cancel_gold_nonempty`, `inspect` regression).
5. **Closed-Loop MPC Planning (`scripts/run_planning_eval.py`, `planner/mpc.py`)**:
   `generate_planning_tasks()` (full-action BFS `<= 5` steps for solvability ground truth) -> `run_tasks_batched()` (`propose_candidates` -> batched world-model predictions -> `decide_step` -> `env_step` real execution -> replan from observed true state) -> `summarize_episodes()` & `breakdown_episodes()`.

## Verification Entry Points

- `python -X utf8 -m pytest -v`: Runs the full deterministic CPU test suite across all six modules.
- `python -X utf8 scripts/generate_dataset.py --check`: Re-generates all four splits in memory and verifies byte-for-byte identity against disk and `FROZEN_DATA_SHA256`.
- `python -X utf8 scripts/run_planning_eval.py --split test --predictors oracle null --no-save`: Sanity-checks the full planning closed loop on CPU.
