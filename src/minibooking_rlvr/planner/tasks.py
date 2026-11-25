"""Planning task generator with full-action BFS solvability labeling (``<= 5`` steps).

Constructs ``dev`` (20 tasks) and ``test`` (100 tasks: 50 ``test_id`` + 50 ``test_comp``)
from the frozen dataset splits without ever reading ``gold`` or ``state_after``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from minibooking_rlvr.env import candidate_actions, step_raw
from minibooking_rlvr.planner.adapter import to_action, to_rules, to_state
from minibooking_rlvr.planner.mpc import goal_met

SEED = 20261025
BFS_MAX_STEPS = 5
CATEGORIES = ("create_confirm", "cancel_change", "loss_limit", "unsolvable")
PLAN = {"dev": ("dev", 20), "test_id": ("test", 50), "test_comp": ("test", 50)}
GENERATOR_VERSION = "planning_tasks_v1"


def _canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def category_counts(n: int) -> List[int]:
    a, b, c = round(n * 0.3), round(n * 0.3), round(n * 0.2)
    return [a, b, c, n - a - b - c]


def _cond(field: str, op: str, value: Any) -> Dict[str, Any]:
    return {"field": field, "op": op, "value": value}


def make_goal(category: str, state: Dict[str, Any], rules: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    days = list(rules["allowed_days"])
    other = [d for d in days if d != state["day"]] or days
    if category == "create_confirm":
        return {"conditions": [_cond("status", "eq", "confirmed"), _cond("day", "eq", rng.choice(days))]}
    if category == "cancel_change":
        if state["status"] == "confirmed" and rng.random() < 0.5:
            refund = state["paid_cents"] * rules["refund_num"] // rules["refund_den"]
            return {
                "conditions": [
                    _cond("status", "eq", "draft"),
                    _cond("balance_cents", "ge", state["balance_cents"] + refund),
                ]
            }
        if state["status"] == "empty":
            return {
                "conditions": [
                    _cond("status", "eq", "confirmed"),
                    _cond("day", "eq", rng.choice(days)),
                    _cond("balance_cents", "ge", state["balance_cents"] - rules["price_cents"]),
                ]
            }
        return {"conditions": [_cond("status", "eq", state["status"]), _cond("day", "eq", rng.choice(other))]}
    if category == "loss_limit":
        return {
            "conditions": [_cond("status", "eq", "confirmed"), _cond("day", "eq", rng.choice(days))],
            "loss_limit_cents": state["loss_cents"],
        }
    if category == "unsolvable":
        # Money conservation (balance + paid + loss = const, loss non-decreasing) guarantees unsolvability.
        return {
            "conditions": [
                _cond("status", "eq", "confirmed"),
                _cond("balance_cents", "ge", state["balance_cents"] + state["paid_cents"] + 1),
            ]
        }
    raise ValueError(f"Unknown task category: {category!r}")


def bfs_label(
    start: Dict[str, Any],
    goal: Dict[str, Any],
    rules: Dict[str, Any],
    max_steps: int = BFS_MAX_STEPS,
) -> Dict[str, Any]:
    """Exhaustive full-action BFS over all 10 domain actions up to ``max_steps``."""
    rules_obj = to_rules(rules)
    acts = [a.to_dict() for a in candidate_actions(rules_obj)]
    key_fn = lambda st: tuple(st[k] for k in sorted(st))
    queue = deque([(start, [])])
    seen = {key_fn(start)}
    nodes = 0
    while queue:
        st, path = queue.popleft()
        nodes += 1
        if goal_met(st, goal):
            return {"solvable": True, "shortest_steps": len(path), "reference_solution": path, "bfs_nodes": nodes}
        if len(path) >= max_steps:
            continue
        for action in acts:
            nxt = step_raw(to_state(st), to_action(action), rules_obj).state.to_dict()
            k = key_fn(nxt)
            if k not in seen:
                seen.add(k)
                queue.append((nxt, path + [action]))
    return {"solvable": False, "shortest_steps": None, "reference_solution": None, "bfs_nodes": nodes}


def build_task_split(
    split: str,
    records: Sequence[Dict[str, Any]],
    n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Build ``n`` deduplicated planning tasks from ``records`` without inspecting gold labels."""
    pools: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        if "split" in rec and rec["split"] != split:
            raise ValueError(f"Record split mismatch: expected {split!r}, got {rec['split']!r}")
        sc = {
            "scenario_id": rec["scenario_id"],
            "rules": rec["rules"],
            "start_state": rec["state_before"],
            "source_sample": rec["sample_id"],
        }
        bucket = pools.setdefault(rec["scenario_id"], {})
        bucket.setdefault(_canon([sc["rules"], sc["start_state"]]), sc)
    if not pools:
        raise ValueError(f"{split}: no source scenarios available")

    families = sorted(pools)
    rng.shuffle(families)
    cats = [c for c, k in zip(CATEGORIES, category_counts(n)) for _ in range(k)]
    rng.shuffle(cats)

    used = set()
    tasks: List[Dict[str, Any]] = []
    for i, cat in enumerate(cats):
        goal = None
        chosen_sc = None
        for attempt in range(max(1000, n * 100)):
            family = families[(i + attempt) % len(families)]
            choices = [pools[family][k] for k in sorted(pools[family])]
            sc = rng.choice(choices)
            g = make_goal(cat, sc["start_state"], sc["rules"], rng)
            key = _canon([sc["rules"], sc["start_state"], g])
            if not goal_met(sc["start_state"], g) and key not in used:
                goal = g
                chosen_sc = sc
                used.add(key)
                break
        if goal is None or chosen_sc is None:
            raise ValueError(f"{split}#{i}: insufficient unique (state, goal) combinations")

        lab = bfs_label(chosen_sc["start_state"], goal, chosen_sc["rules"])
        r = chosen_sc["rules"]
        tasks.append(
            {
                "task_id": f"{split}-{i:03d}",
                "split": split,
                "category": cat,
                "scenario_id": chosen_sc["scenario_id"],
                "source_sample": chosen_sc["source_sample"],
                "rules": r,
                "start_state": chosen_sc["start_state"],
                "goal": goal,
                **lab,
                "tags": {
                    "source": split,
                    "allow_change": r["allow_change"],
                    "full_refund_fee20": r["refund_num"] == r["refund_den"] and r["change_fee_cents"] == 20,
                    "ref_contains_cancel": bool(lab["reference_solution"])
                    and any(a["name"] == "cancel" for a in lab["reference_solution"]),
                },
            }
        )
    return tasks


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_planning_tasks(
    data_dir: str | Path,
    *,
    seed: int = SEED,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Generate ``dev`` (20) and ``test`` (100) planning benchmarks from frozen data splits."""
    data_path = Path(data_dir)
    rng = random.Random(seed)
    out: Dict[str, List[Dict[str, Any]]] = {"dev": [], "test": []}
    src: Dict[str, Any] = {}
    seen_families = set()
    for split, (bucket, n) in PLAN.items():
        p = data_path / f"{split}.jsonl"
        src[split] = {"file": f"{split}.jsonl", "sha256": _sha256_file(p)}
        records = _load_jsonl(p)
        families = {r["scenario_id"] for r in records}
        if seen_families & families:
            raise ValueError("Source scenario families overlap across splits")
        seen_families.update(families)
        out[bucket] += build_task_split(split, records, n, rng)

    manifest: Dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "bfs_max_steps": BFS_MAX_STEPS,
        "category_ratio": "30/30/20/20",
        "source_data": src,
    }
    for bucket, ts in out.items():
        manifest[f"counts_{bucket}"] = {
            "n": len(ts),
            "by_category": dict(Counter(t["category"] for t in ts)),
            "solvable": sum(t["solvable"] for t in ts),
            "by_source": dict(Counter(t["split"] for t in ts)),
        }
    return out, manifest
