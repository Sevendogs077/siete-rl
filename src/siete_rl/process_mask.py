"""Process mask 纯决策层：只产出 per-token 二元权重 α∈{0,1}。

职责边界：本模块不接触 TRL/torch，也不管 base_mask 的合成；候选识别、
advantage 门控与治理终止都在这里完成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from siete_rl.models import Action, Step, Termination

TurnKind = Literal["step", "invalid_call", "plain_message"]


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """一段连续 token 区间的来源标注。"""

    token_start: int
    token_end: int
    kind: TurnKind
    step_index: int | None


_REPEATABLE_TOOLS = frozenset({"execute_bash", "str_replace_editor"})


def _action_key(action: Action) -> tuple[str, str] | None:
    """返回完整、稳定的字面动作 key；终止和未知工具不参与重复检测。"""
    if action.tool_name not in _REPEATABLE_TOOLS:
        return None
    arguments = json.dumps(
        action.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return action.tool_name, arguments


_GOVERNANCE_TERMINATIONS = frozenset({"infra_error", "context_overlong"})


@dataclass(frozen=True, slots=True)
class ProcessMaskStats:
    candidate_turns: int
    applied_turns: int
    retained_negative_turns: int
    masked_token_frac: float
    governance_masked: bool


def _validate_turn(turn: TurnRecord, *, n_tokens: int, n_steps: int) -> None:
    if turn.token_start < 0 or turn.token_end < turn.token_start or turn.token_end > n_tokens:
        raise ValueError(f"invalid turn token range: {turn}")
    if turn.kind == "step" and (
        turn.step_index is None or turn.step_index < 0 or turn.step_index >= n_steps
    ):
        raise ValueError(f"invalid step turn index: {turn}")
    if turn.kind != "step" and turn.step_index is not None:
        raise ValueError(f"non-step turn has step index: {turn}")


def _candidate_turn_positions(
    turns: list[TurnRecord], steps: list[Step], n_tokens: int
) -> list[int]:
    candidates: list[int] = []
    previous_key: tuple[str, str] | None = None
    streak = 0
    for position, turn in enumerate(turns):
        _validate_turn(turn, n_tokens=n_tokens, n_steps=len(steps))
        if turn.kind == "invalid_call":
            candidates.append(position)
            previous_key = None
            streak = 0
            continue
        if turn.kind != "step":
            previous_key = None
            streak = 0
            continue
        key = _action_key(steps[turn.step_index].action)
        if key is None:
            previous_key = None
            streak = 0
            continue
        if key == previous_key:
            streak += 1
        else:
            previous_key = key
            streak = 1
        if streak >= 3:
            candidates.append(position)
    return candidates


def _masked_fraction(base_mask: list[float], weights: list[float]) -> float:
    base_tokens = sum(value == 1.0 for value in base_mask)
    if base_tokens == 0:
        return 0.0
    retained_tokens = sum(value == 1.0 for value in weights)
    return (base_tokens - retained_tokens) / base_tokens


def build_process_token_weights(
    *,
    turns: list[TurnRecord],
    steps: list[Step],
    termination: Termination,
    advantage: float,
    base_mask: list[float],
) -> tuple[list[float], ProcessMaskStats]:
    if any(value not in (0.0, 1.0) for value in base_mask):
        raise ValueError("base_mask must be binary")
    candidates = _candidate_turn_positions(turns, steps, len(base_mask))
    if termination in _GOVERNANCE_TERMINATIONS:
        weights = [0.0] * len(base_mask)
        return weights, ProcessMaskStats(
            candidate_turns=len(candidates),
            applied_turns=0,
            retained_negative_turns=0,
            masked_token_frac=_masked_fraction(base_mask, weights),
            governance_masked=True,
        )

    weights = list(base_mask)
    applied_turns = 0
    retained_negative_turns = 0
    if advantage > 0.0:
        for position in candidates:
            turn = turns[position]
            weights[turn.token_start : turn.token_end] = [0.0] * (
                turn.token_end - turn.token_start
            )
        applied_turns = len(candidates)
    elif advantage < 0.0:
        retained_negative_turns = len(candidates)

    return weights, ProcessMaskStats(
        candidate_turns=len(candidates),
        applied_turns=applied_turns,
        retained_negative_turns=retained_negative_turns,
        masked_token_frac=_masked_fraction(base_mask, weights),
        governance_masked=False,
    )
