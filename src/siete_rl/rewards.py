"""唯一的 custom reward adapter（binary / layered 共用，分层逻辑在环境 _finalize 内）。"""

from __future__ import annotations

from typing import Any

from siete_rl.environment import SWEEnvironment


def binary_reward(
    completions: list[object],
    environments: list[SWEEnvironment],
    **kwargs: Any,
) -> list[float]:
    """同位置 finalize；基础设施异常已由 environment 降级为零分 infra_error 样本。

    此处不再做错误政策：契约错误（计数不匹配）照样抛出，单样本 infra 降级
    由 SWEEnvironment._finalize 统一收口，系统性故障由 _recording_reward 熔断。
    """

    del kwargs
    if len(completions) != len(environments):
        raise ValueError("completion and environment counts do not match")
    return [
        environment._finalize(completion)
        for completion, environment in zip(completions, environments, strict=True)
    ]
