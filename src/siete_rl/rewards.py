"""唯一的 custom reward adapter（binary / layered 共用，分层逻辑在环境 _finalize 内）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from siete_rl.environment import SWEEnvironment


def binary_reward(
    completions: list[object],
    environments: list[SWEEnvironment],
    max_workers: int = 1,
    **kwargs: Any,
) -> list[float]:
    """同位置 finalize；基础设施异常已由 environment 降级为零分 infra_error 样本。

    此处不再做错误政策：契约错误（计数不匹配）照样抛出，单样本 infra 降级
    由 SWEEnvironment._finalize 统一收口，系统性故障由 _recording_reward 熔断。

    max_workers > 1 且样本数 > 1 时跨样本并行 finalize：每条 trajectory 的
    environment/verifier/容器状态互相独立（实例级隔离），pool.map 按提交序保序返回，
    reward 与 completion 的索引对齐不变；异常在消费对应索引时传播，已提交的其他
    environment 仍会独立收束。max_workers == 1 走原串行列表推导，不建池。
    """

    del kwargs
    if len(completions) != len(environments):
        raise ValueError("completion and environment counts do not match")
    if max_workers > 1 and len(environments) > 1:
        with ThreadPoolExecutor(
            max_workers=min(len(environments), max_workers)
        ) as pool:
            return list(
                pool.map(
                    lambda pair: pair[0]._finalize(pair[1]),
                    zip(environments, completions, strict=True),
                )
            )
    return [
        environment._finalize(completion)
        for completion, environment in zip(completions, environments, strict=True)
    ]
