"""Frozen prompt protocol (``minibooking_prompt_v1``) and boundary sample selection.

Strict leakage rule:
``build_messages`` reads ONLY ``INPUT_KEYS = ("rules", "state_before", "action")`` and
never accesses ``gold``, ``state_after``, or ``metadata``.

The system prompt text is intentionally frozen in Chinese to match the exact ``v1`` prompt
used to train and evaluate the reported Qwen3-1.7B SFT / GRPO checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

PROMPT_VERSION = "minibooking_prompt_v1"
INPUT_KEYS = ("rules", "state_before", "action")

SYSTEM_PROMPT = """你是 MiniBooking 订单环境的世界模型。给定场景规则 rules、动作前状态 state_before 和一个动作 action，预测执行该动作后状态发生的变化。

状态有五个字段：status（empty/draft/confirmed）、day（null 或 1/2/3）、balance_cents（余额，单位分）、paid_cents（当前订单已付金额，分）、loss_cents（累计不可返还的损失，分）。

动作语义：
- create(day)：要求 status=empty 且 day 在 allowed_days 内；结果 status→draft，day→目标日期。
- confirm()：要求 status=draft 且 balance_cents ≥ price_cents；结果 status→confirmed，balance_cents 减 price_cents，paid_cents→price_cents。
- cancel()：要求 status=confirmed；退款 = paid_cents × refund_num / refund_den；结果 status→draft（day 不变），balance_cents 加退款，paid_cents→0，loss_cents 加 (paid_cents − 退款)。
- change_day(day)：要求 status 为 draft 或 confirmed、allow_change=true、day 在 allowed_days 内。目标日期与当前日期相同：无变化、不收费。draft：只改 day。confirmed：改 day，balance_cents 减 change_fee_cents，loss_cents 加 change_fee_cents；余额不足以支付改期费则不能执行。
- clear()：要求 status=draft；结果 status→empty，day→null。
- inspect()：总能执行，状态不变。
条件不满足的动作不改变任何字段。

只输出一个 JSON 对象，不要代码围栏，不要任何解释文字。对象恰好包含四个键：
- "changes"：列表，只列出值发生改变的字段，每项为 {"field": 字段名, "value": 新值}，按字段名字母序排列；没有任何变化时为 []。
- "reversible"：布尔值 true/false，表示最多再执行 3 个动作能否让五个字段全部回到动作前的值（没有变化时为 true）。
- "cost"：本次动作新增的 loss_cents 除以 price_cents，限制在 [0, 1] 内。
- "confidence"：你认为 changes 完全正确的把握，0 到 1 之间的数。"""


def _canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_jsonl(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages strictly from ``INPUT_KEYS`` without touching labels."""
    x = {k: record[k] for k in INPUT_KEYS}
    user = (
        "rules: " + _canon(x["rules"]) + "\n"
        "state_before: " + _canon(x["state_before"]) + "\n"
        "action: " + _canon(x["action"]) + "\n"
        "请输出 JSON。"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_prompt(tokenizer: Any, record: Dict[str, Any], enable_thinking: bool = False) -> str:
    """Apply chat template with non-thinking mode enabled by default."""
    return tokenizer.apply_chat_template(
        build_messages(record),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def gold_completion(record: Dict[str, Any], confidence: float = 1.0) -> str:
    """Format compact target JSON string for supervised fine-tuning."""
    g = record["gold"]
    obj = {"changes": g["changes"], "reversible": g["reversible"], "cost": g["cost"], "confidence": confidence}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _m(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["metadata"]


def _refund(record: Dict[str, Any], kind: str) -> bool:
    num, den = record["rules"]["refund_num"], record["rules"]["refund_den"]
    return {"zero": num == 0, "full": num == den, "half": 0 < num < den}[kind]


BOUNDARY_RULES: Dict[str, Tuple[Callable[[Dict[str, Any]], bool], List[Callable[[Dict[str, Any]], bool]]]] = {
    "normal_change": (
        lambda r: _m(r)["legal"] and not _m(r)["no_change"] and _m(r)["action_type"] in ("create", "confirm", "clear"),
        [
            lambda r: _m(r)["action_type"] == "confirm",
            lambda r: _m(r)["action_type"] == "create",
            lambda r: _m(r)["action_type"] == "clear",
        ],
    ),
    "no_change": (
        lambda r: _m(r)["legal"] and _m(r)["no_change"],
        [lambda r: _m(r)["reason"] == "inspect", lambda r: _m(r)["reason"] == "same_day"],
    ),
    "illegal_action": (
        lambda r: not _m(r)["legal"] and _m(r)["action_type"] != "change_day",
        [
            lambda r: _m(r)["reason"] == "insufficient_balance",
            lambda r: _m(r)["reason"] == "invalid_day",
            lambda r: _m(r)["reason"] == "wrong_status",
        ],
    ),
    "refund_boundary": (
        lambda r: _m(r)["action_type"] == "cancel" and _m(r)["legal"],
        [lambda r: _refund(r, "zero"), lambda r: _refund(r, "full"), lambda r: _refund(r, "half")],
    ),
    "change_day_boundary": (
        lambda r: _m(r)["action_type"] == "change_day" and _m(r)["reason"] != "same_day",
        [
            lambda r: _m(r)["legal"]
            and r["state_before"]["status"] == "confirmed"
            and r["rules"]["change_fee_cents"] > 0,
            lambda r: _m(r)["reason"] == "insufficient_balance",
            lambda r: _m(r)["reason"] == "change_not_allowed",
            lambda r: _m(r)["reason"] == "invalid_day",
            lambda r: _m(r)["legal"],
        ],
    ),
}


def select_boundary_samples(
    records: Sequence[Dict[str, Any]],
    n_per_category: int = 2,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Deterministically pick representative boundary cases across five domain categories."""
    used = set()
    picked: List[Tuple[str, Dict[str, Any]]] = []
    for cat, (base, subs) in BOUNDARY_RULES.items():
        got = 0
        for pred in list(subs) + [lambda _r: True]:
            if got >= n_per_category:
                break
            for rec in records:
                if rec["sample_id"] not in used and base(rec) and pred(rec):
                    used.add(rec["sample_id"])
                    picked.append((cat, rec))
                    got += 1
                    break
    return picked
