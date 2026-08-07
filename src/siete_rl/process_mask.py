"""Process mask 规则层：只产出 per-token 二元权重 α∈{0,1}。

职责边界：本模块不接触 TRL/torch，也不管 base_mask 的合成与 loss 归一，
只按规则把命中的 turn 区间置 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from siete_rl.models import Action, Step

TurnKind = Literal["step", "invalid_call", "plain_message"]


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """一段连续 token 区间的来源标注。"""

    token_start: int
    token_end: int
    kind: TurnKind
    step_index: int | None


def action_signature(action: Action) -> tuple | None:
    """动作去重签名；无签名的动作（如 finish）返回 None，不参与计数。"""
    args = action.arguments
    if action.tool_name == "execute_bash":
        return ("execute_bash", args.get("command"))
    if action.tool_name == "str_replace_editor":
        old_str = args.get("old_str") or ""
        return ("str_replace_editor", args.get("command"), args.get("path"), old_str[:200])
    return None


class ProcessMaskRule(Protocol):
    name: str

    def masked(self, turn: TurnRecord, occurrence: int | None) -> bool: ...


class InvalidCallMask:
    """无效调用（解析失败/非法工具）的 turn 整体置 0。"""

    name = "invalid_call"

    def masked(self, turn: TurnRecord, occurrence: int | None) -> bool:
        return turn.kind == "invalid_call"


class DuplicateActionMask:
    """同签名动作第 threshold 次出现起，对应 turn 置 0（成败均计数）。"""

    name = "duplicate_action"
    threshold = 3

    def masked(self, turn: TurnRecord, occurrence: int | None) -> bool:
        return occurrence is not None and occurrence >= self.threshold


RULE_REGISTRY: dict[str, type] = {
    InvalidCallMask.name: InvalidCallMask,
    DuplicateActionMask.name: DuplicateActionMask,
}


def resolve_rules(names: list[str]) -> list[ProcessMaskRule]:
    rules = []
    for name in names:
        if name not in RULE_REGISTRY:
            raise ValueError(f"unknown process mask rule: {name}")
        rules.append(RULE_REGISTRY[name]())
    return rules


def build_alpha(
    turns: list[TurnRecord],
    steps: list[Step],
    n_tokens: int,
    rules: list[ProcessMaskRule],
) -> tuple[list[float], int]:
    """按 turn 顺序遍历，任一规则命中则把 [token_start, token_end) 置 0。

    返回 (alpha, masked_turns)；masked_turns 是被任一规则命中的 turn 数。
    end 允许超出 n_tokens（回滚边缘情形，clamp 到 n_tokens），start 必须合法：
    token_start < 0 或 token_end < token_start 直接抛 ValueError。
    """
    alpha = [1.0] * n_tokens
    counts: dict[tuple, int] = {}
    masked_turns = 0
    for turn in turns:
        if turn.token_start < 0 or turn.token_end < turn.token_start:
            raise ValueError(f"invalid turn token range: {turn}")
        occurrence = None
        if turn.kind == "step" and turn.step_index is not None:
            signature = action_signature(steps[turn.step_index].action)
            if signature is not None:
                counts[signature] = counts.get(signature, 0) + 1
                occurrence = counts[signature]
        if any(rule.masked(turn, occurrence) for rule in rules):
            masked_turns += 1
            for i in range(turn.token_start, min(turn.token_end, n_tokens)):
                alpha[i] = 0.0
    return alpha, masked_turns
