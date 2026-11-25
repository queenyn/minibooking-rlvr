"""Leakage-free, byte-reproducible dataset generator for MiniBooking-RLVR.

Pipeline invariant:
  1. Partition scenario families ``(rules, initial_balance_cents)`` across
     ``train / dev / test_id / test_comp`` BEFORE any state sampling.
  2. Hold out two unseen ``(refund_ratio, change_fee_cents)`` combinations exclusively for
     ``test_comp`` while ensuring every component appears in ``train``.
  3. Within each split, sample reachable states via short random walks using ``step_raw``,
     enumerate candidate actions plus out-of-range day actions, and deduplicate by canonical
     input SHA256.
  4. Round-robin sample by action type with target ``illegal_ratio = 0.25`` and verify zero
     cross-split hash or scenario-family overlap.

All outputs are guaranteed byte-identical to ``FROZEN_DATA_SHA256`` across platforms (UTF-8, LF).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from minibooking_rlvr.env import ENV_VERSION, Action, Rules, State, candidate_actions, label_transition, step_raw

GENERATOR_VERSION = "make_dataset_v1"
SEED = 20261025

PRICE_GRID = [100, 200]
REFUND_GRID = [(0, 1), (1, 2), (1, 1)]
ALLOW_CHANGE_GRID = [True, False]
FEE_GRID = [0, 20]
BALANCE_GRID = [0, 50, 100, 150, 200, 250, 300, 400, 600, 1000]

# Held-out compositional pairs exclusively assigned to test_comp.
HELDOUT_REFUND_FEE = [((1, 2), 0), ((1, 1), 20)]

TARGET_SIZES = {"train": 2400, "dev": 300, "test_id": 300, "test_comp": 300}
SPLITS = list(TARGET_SIZES)
ILLEGAL_RATIO = 0.25
WALKS_PER_FAMILY = 60
MAX_WALK_LEN = 8
OUT_OF_RANGE_ACTIONS = [
    Action("create", 0),
    Action("create", 4),
    Action("change_day", 0),
    Action("change_day", 4),
]

# Byte-level SHA256 hashes of the frozen v1 benchmark splits.
FROZEN_DATA_SHA256 = {
    "train.jsonl": "6b2fa4029078672d563447508a0bf6f5c78632784e8c7975266a6edd7919d2d7",
    "dev.jsonl": "fd64e7a0525135d313df2bfa1a9cf3c91dcecb71e060bd145bf4a5cad9fa1d21",
    "test_id.jsonl": "ad0bc9329860417e8c6077f9a0b0ec23bf6b789b463e007ea741284ab95c3bb7",
    "test_comp.jsonl": "378c7eec15a24611908431c6a116e4c5ced248d4f5e10888596bf9cc372a7cfa",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_hash(rules: Rules, state: State, action: Action) -> str:
    """Hash strictly the model input ``(rules, state_before, action)`` for deduplication."""
    payload = {"rules": rules.to_dict(), "state_before": state.to_dict(), "action": action.to_dict()}
    return sha256_text(canonical_json(payload))


def make_rng(*parts: Any, seed: int = SEED) -> random.Random:
    return random.Random(":".join([str(seed), *map(str, parts)]))


def all_rules() -> List[Rules]:
    rules_list: List[Rules] = []
    for price, refund, allow, fee in itertools.product(PRICE_GRID, REFUND_GRID, ALLOW_CHANGE_GRID, FEE_GRID):
        rules = Rules(
            price_cents=price,
            refund_num=refund[0],
            refund_den=refund[1],
            allow_change=allow,
            change_fee_cents=fee,
        )
        rules.validate()
        rules_list.append(rules)
    return rules_list


def refund_fee_pair(rules: Rules) -> Tuple[Tuple[int, int], int]:
    return (rules.refund_num, rules.refund_den), rules.change_fee_cents


def is_heldout(rules: Rules) -> bool:
    return refund_fee_pair(rules) in HELDOUT_REFUND_FEE


def scenario_id(rules: Rules, balance: int) -> str:
    return "sc-" + sha256_text(canonical_json({"rules": rules.to_dict(), "initial_balance_cents": balance}))[:12]


def assign_families(seed: int = SEED) -> Dict[str, List[Tuple[Rules, int]]]:
    """Partition all 240 scenario families across splits prior to state sampling."""
    rng = make_rng("split", seed=seed)
    id_rules = [r for r in all_rules() if not is_heldout(r)]
    comp_rules = [r for r in all_rules() if is_heldout(r)]
    rules_order = list(range(len(id_rules)))
    rng.shuffle(rules_order)
    balance_order = list(range(len(BALANCE_GRID)))
    rng.shuffle(balance_order)
    half = len(BALANCE_GRID) // 2

    families: Dict[str, List[Tuple[Rules, int]]] = {s: [] for s in SPLITS}
    for slot, rule_index in enumerate(rules_order):
        rules = id_rules[rule_index]
        dev_b = BALANCE_GRID[balance_order[slot % len(BALANCE_GRID)]]
        test_b = BALANCE_GRID[balance_order[(slot + half) % len(BALANCE_GRID)]]
        for balance in BALANCE_GRID:
            split = "dev" if balance == dev_b else "test_id" if balance == test_b else "train"
            families[split].append((rules, balance))
    for rules in comp_rules:
        for balance in BALANCE_GRID:
            families["test_comp"].append((rules, balance))
    return families


def _state_sort_key(state: State) -> str:
    return canonical_json(state.to_dict())


def reachable_states(rules: Rules, balance: int, rng: random.Random) -> List[State]:
    """Explore reachable states from an empty order using short legal random walks."""
    start = State("empty", None, balance, 0, 0)
    seen = {start.key(): start}
    actions = candidate_actions(rules)
    for _ in range(WALKS_PER_FAMILY):
        state = start
        for _ in range(rng.randint(0, MAX_WALK_LEN)):
            legal_next = []
            for action in actions:
                result = step_raw(state, action, rules)
                if result.legal and result.state.key() != state.key():
                    legal_next.append(result.state)
            if not legal_next:
                break
            state = rng.choice(legal_next)
            seen.setdefault(state.key(), state)
    return sorted(seen.values(), key=_state_sort_key)


def build_pool(families: List[Tuple[Rules, int]], seed: int = SEED) -> Dict[str, Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    for rules, balance in families:
        sid = scenario_id(rules, balance)
        rng = make_rng("walk", sid, seed=seed)
        for state in reachable_states(rules, balance, rng):
            actions = candidate_actions(rules) + [rng.choice(OUT_OF_RANGE_ACTIONS)]
            for action in actions:
                h = input_hash(rules, state, action)
                if h in pool:
                    continue
                legal = step_raw(state, action, rules).legal
                pool[h] = {
                    "hash": h,
                    "scenario_id": sid,
                    "rules": rules,
                    "balance": balance,
                    "state": state,
                    "action": action,
                    "legal": legal,
                }
    return pool


def _round_robin_take(items: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in sorted(items, key=lambda x: x["hash"]):
        groups.setdefault(item["action"].name, []).append(item)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    chosen: List[Dict[str, Any]] = []
    names = sorted(groups)
    while len(chosen) < n and any(groups[k] for k in names):
        for name in names:
            if groups[name] and len(chosen) < n:
                chosen.append(groups[name].pop())
    return chosen


def _split_quota(n: int) -> Tuple[int, int]:
    n_illegal = round(n * ILLEGAL_RATIO)
    return n - n_illegal, n_illegal


def _capacity_scale(pools: Dict[str, Dict[str, Dict[str, Any]]]) -> float:
    scale = 1.0
    for split, pool in pools.items():
        n_legal, n_illegal = _split_quota(TARGET_SIZES[split])
        legal = sum(1 for x in pool.values() if x["legal"])
        illegal = len(pool) - legal
        scale = min(scale, legal / n_legal, illegal / n_illegal if n_illegal else 1.0)
    return scale


def _make_record(split: str, index: int, item: Dict[str, Any], seed: int = SEED) -> Dict[str, Any]:
    labeled = label_transition(item["state"], item["action"], item["rules"])
    metadata = dict(labeled["metadata"])
    metadata.update(
        {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "input_hash": item["hash"],
            "initial_balance_cents": item["balance"],
        }
    )
    return {
        "sample_id": f"{split}-{index:05d}",
        "scenario_id": item["scenario_id"],
        "split": split,
        "rules": labeled["rules"],
        "state_before": labeled["state_before"],
        "action": labeled["action"],
        "state_after": labeled["state_after"],
        "gold": labeled["gold"],
        "metadata": metadata,
    }


def generate_dataset(seed: int = SEED) -> Dict[str, Any]:
    """Generate all four splits in memory with pool statistics and scaling metadata."""
    families = assign_families(seed=seed)
    pools = {s: build_pool(families[s], seed=seed) for s in SPLITS}
    pool_stats = {}
    for s, pool in pools.items():
        legal = sum(1 for x in pool.values() if x["legal"])
        pool_stats[s] = {"unique_items": len(pool), "legal": legal, "illegal": len(pool) - legal}

    scale = _capacity_scale(pools)
    actual_targets = dict(TARGET_SIZES) if scale >= 1.0 else {s: int(n * scale) for s, n in TARGET_SIZES.items()}

    records: Dict[str, List[Dict[str, Any]]] = {}
    for s in SPLITS:
        rng = make_rng("sample", s, seed=seed)
        pool = list(pools[s].values())
        n_legal, n_illegal = _split_quota(actual_targets[s])
        chosen = _round_robin_take([x for x in pool if x["legal"]], n_legal, rng)
        chosen += _round_robin_take([x for x in pool if not x["legal"]], n_illegal, rng)
        chosen.sort(key=lambda x: x["hash"])
        rng.shuffle(chosen)
        records[s] = [_make_record(s, i, item, seed=seed) for i, item in enumerate(chosen)]

    return {
        "families": families,
        "pool_stats": pool_stats,
        "scale": scale,
        "actual_targets": actual_targets,
        "records": records,
    }


def jsonl_text(records: List[Dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)


def _cost_bucket(cost: float) -> str:
    if cost == 0:
        return "0"
    if cost <= 0.25:
        return "(0,0.25]"
    if cost <= 0.5:
        return "(0.25,0.5]"
    return "(0.5,1]"


def _quantiles(values: List[int]) -> Dict[str, int]:
    data = sorted(values)
    out = {}
    for q in (0, 25, 50, 75, 90, 95, 99, 100):
        idx = max(0, min(len(data) - 1, -(-q * len(data) // 100) - 1)) if q else 0
        out[f"p{q}"] = data[idx]
    return out


def split_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = [r["metadata"] for r in records]
    costs = [r["gold"]["cost"] for r in records]
    legal_n = sum(m["legal"] for m in meta)
    illegal_n = len(meta) - legal_n
    nc_legal = sum(1 for m in meta if m["legal"] and m["no_change"])
    nc_illegal = sum(1 for m in meta if not m["legal"] and m["no_change"])
    rev = Counter(str(r["gold"]["reversible"]).lower() for r in records)
    rev_legal = Counter(str(r["gold"]["reversible"]).lower() for r in records if r["metadata"]["legal"])
    rev_legal_change = Counter(
        str(r["gold"]["reversible"]).lower()
        for r in records
        if r["metadata"]["legal"] and not r["metadata"]["no_change"]
    )
    return {
        "num_scenarios": len({r["scenario_id"] for r in records}),
        "action_type": dict(sorted(Counter(m["action_type"] for m in meta).items())),
        "action_type_legal": dict(sorted(Counter(m["action_type"] for m in meta if m["legal"]).items())),
        "reason": dict(sorted(Counter(m["reason"] for m in meta).items())),
        "legal": {"legal": legal_n, "illegal": illegal_n},
        "no_change": {
            "legal_no_change": nc_legal,
            "legal_total": legal_n,
            "legal_no_change_ratio": round(nc_legal / legal_n, 4) if legal_n else None,
            "illegal_no_change": nc_illegal,
            "illegal_total": illegal_n,
            "illegal_no_change_ratio": round(nc_illegal / illegal_n, 4) if illegal_n else None,
        },
        "reversible": dict(sorted(rev.items())),
        "reversible_legal_only": dict(sorted(rev_legal.items())),
        "reversible_legal_with_change": dict(sorted(rev_legal_change.items())),
        "cost": {
            "min": min(costs),
            "max": max(costs),
            "all_in_0_1": all(0.0 <= c <= 1.0 for c in costs),
            "buckets": dict(sorted(Counter(_cost_bucket(c) for c in costs).items())),
            "values": {str(k): v for k, v in sorted(Counter(costs).items())},
        },
        "initial_balance_cents": {
            str(k): v for k, v in sorted(Counter(r["metadata"]["initial_balance_cents"] for r in records).items())
        },
        "rules_refund_fee_pairs": dict(
            sorted(
                Counter(
                    f"refund={r['rules']['refund_num']}/{r['rules']['refund_den']},fee={r['rules']['change_fee_cents']}"
                    for r in records
                ).items()
            )
        ),
        "gold_json_chars": _quantiles([len(canonical_json(r["gold"])) for r in records]),
    }


def cross_split_checks(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Verify zero cross-split hash/scenario overlap, unique hashes, and held-out comp rules."""
    hashes = {s: [r["metadata"]["input_hash"] for r in recs] for s, recs in records.items()}
    scen = {s: {r["scenario_id"] for r in recs} for s, recs in records.items()}
    within_unique = {s: len(h) == len(set(h)) for s, h in hashes.items()}
    pair_hash: Dict[str, int] = {}
    pair_scen: Dict[str, int] = {}
    for a, b in itertools.combinations(SPLITS, 2):
        pair_hash[f"{a}&{b}"] = len(set(hashes[a]) & set(hashes[b]))
        pair_scen[f"{a}&{b}"] = len(scen[a] & scen[b])

    def _pairs(split: str) -> set:
        return {
            ((r["rules"]["refund_num"], r["rules"]["refund_den"]), r["rules"]["change_fee_cents"])
            for r in records[split]
        }

    non_comp_pairs = _pairs("train") | _pairs("dev") | _pairs("test_id")
    comp_pairs = _pairs("test_comp")
    train_refunds = {p[0] for p in _pairs("train")}
    train_fees = {p[1] for p in _pairs("train")}
    recomputed_ok = all(
        r["metadata"]["input_hash"]
        == sha256_text(canonical_json({"rules": r["rules"], "state_before": r["state_before"], "action": r["action"]}))
        for recs in records.values()
        for r in recs
    )
    ok = (
        all(within_unique.values())
        and not any(pair_hash.values())
        and not any(pair_scen.values())
        and not (comp_pairs & non_comp_pairs)
        and all(p[0] in train_refunds and p[1] in train_fees for p in comp_pairs)
        and recomputed_ok
    )
    return {
        "all_passed": ok,
        "within_split_hash_unique": within_unique,
        "cross_split_hash_intersection": pair_hash,
        "cross_split_scenario_intersection": pair_scen,
        "input_hash_recomputed_ok": recomputed_ok,
        "comp_pairs_absent_from_train_dev_test_id": not (comp_pairs & non_comp_pairs),
        "comp_components_seen_in_train": all(p[0] in train_refunds and p[1] in train_fees for p in comp_pairs),
    }


def build_protocol(result: Dict[str, Any], seed: int = SEED) -> Dict[str, Any]:
    fams = result["families"]
    return {
        "env_version": ENV_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "scenario_family": {
            "definition": "(price_cents, refund_ratio, allow_change, change_fee_cents, initial_balance_cents); all states of a family belong strictly to one split.",
            "grids": {
                "price_cents": PRICE_GRID,
                "refund": [f"{n}/{d}" for n, d in REFUND_GRID],
                "allow_change": ALLOW_CHANGE_GRID,
                "change_fee_cents": FEE_GRID,
                "initial_balance_cents": BALANCE_GRID,
                "allowed_days": [1, 2, 3],
            },
            "num_rules": len(PRICE_GRID) * len(REFUND_GRID) * len(ALLOW_CHANGE_GRID) * len(FEE_GRID),
            "num_families": sum(len(f) for f in fams.values()),
            "families_per_split": {s: len(f) for s, f in fams.items()},
        },
        "comp_heldout": {
            "refund_fee_pairs": [{"refund": f"{n}/{d}", "change_fee_cents": fee} for (n, d), fee in HELDOUT_REFUND_FEE],
            "note": "Held-out (refund, change_fee_cents) pairs appear only in test_comp while each individual factor appears in train. When allow_change=False, change_fee_cents has no behavioral effect; compositional evaluation should separate allow_change=True and allow_change=False.",
        },
        "generation": {
            "walks_per_family": WALKS_PER_FAMILY,
            "max_walk_len": MAX_WALK_LEN,
            "illegal_ratio_target": ILLEGAL_RATIO,
            "dedupe": "SHA256(canonical_json({rules, state_before, action}))",
            "sampling": "Round-robin by action_type separately for legal and illegal pools without replacement",
        },
        "target_sizes": TARGET_SIZES,
        "actual_sizes": {s: len(r) for s, r in result["records"].items()},
        "scaled_down": result["scale"] < 1.0,
        "scale_factor": 1.0 if result["scale"] >= 1.0 else round(result["scale"], 6),
        "unique_pool_sizes": result["pool_stats"],
    }


def build_manifest(result: Dict[str, Any], texts: Dict[str, str], seed: int = SEED) -> Dict[str, Any]:
    files = {}
    for s in SPLITS:
        text = texts[s]
        files[f"data/{s}.jsonl"] = {
            "lines": text.count("\n"),
            "sha256": sha256_text(text),
            **split_stats(result["records"][s]),
        }
    return {
        "env_version": ENV_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "target_sizes": TARGET_SIZES,
        "actual_sizes": {s: len(r) for s, r in result["records"].items()},
        "scaled_down": result["scale"] < 1.0,
        "unique_pool_sizes": result["pool_stats"],
        "files": files,
        "checks": cross_split_checks(result["records"]),
    }


def _write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def materialize_dataset(
    data_dir: Path,
    config_dir: Path | None = None,
    *,
    check_only: bool = False,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Generate or byte-verify ``data/{train,dev,test_id,test_comp}.jsonl`` and manifests."""
    result = generate_dataset(seed=seed)
    texts = {s: jsonl_text(result["records"][s]) for s in SPLITS}
    manifest = build_manifest(result, texts, seed=seed)
    protocol = build_protocol(result, seed=seed)
    if not manifest["checks"]["all_passed"]:
        raise RuntimeError(f"Cross-split integrity checks failed: {manifest['checks']}")

    outputs: Dict[Path, str] = {data_dir / f"{s}.jsonl": texts[s] for s in SPLITS}
    outputs[data_dir / "data_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if config_dir is not None:
        outputs[config_dir / "data_protocol.json"] = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"

    if check_only:
        mismatched = [
            str(path) for path, text in outputs.items() if not path.exists() or path.read_bytes() != text.encode("utf-8")
        ]
        return {"ok": len(mismatched) == 0, "mismatched": mismatched, "manifest": manifest}

    for path, text in outputs.items():
        _write_lf(path, text)
    return {"ok": True, "mismatched": [], "manifest": manifest}
