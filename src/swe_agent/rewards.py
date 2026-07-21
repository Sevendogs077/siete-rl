"""唯一 custom binary reward adapter。"""

from __future__ import annotations

from typing import Any

from swe_agent.environment import SWEEnvironment


def binary_reward(
    completions: list[object],
    environments: list[SWEEnvironment],
    **kwargs: Any,
) -> list[float]:
    """同位置 finalize；策略失败映射零，基础设施异常原样传播。"""

    del kwargs
    if len(completions) != len(environments):
        raise ValueError("completion and environment counts do not match")
    return [
        environment._finalize(completion)
        for completion, environment in zip(completions, environments, strict=True)
    ]
