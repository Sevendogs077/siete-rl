"""binary_reward 并行 finalize 的保序、异常隔离、串行基线与长尾覆盖。"""

from __future__ import annotations

import time

import pytest

from siete_rl.docker import DockerRuntimeError
from siete_rl.rewards import binary_reward


class FakeEnvironment:
    """按 index 可识别的 _finalize；记录调用次数用于隔离断言。"""

    def __init__(
        self,
        index: int,
        *,
        delay: float = 0.0,
        error: BaseException | None = None,
    ) -> None:
        self.index = index
        self.delay = delay
        self.error = error
        self.finalize_calls = 0

    def _finalize(self, completion: object) -> float:
        self.finalize_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return float(self.index) + 0.5


def test_parallel_finalize_preserves_index_alignment() -> None:
    envs = [FakeEnvironment(i) for i in range(8)]
    rewards = binary_reward([None] * 8, envs, max_workers=8)
    assert rewards == [float(i) + 0.5 for i in range(8)]
    assert all(env.finalize_calls == 1 for env in envs)


def test_default_serial_path_builds_no_pool(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("max_workers=1 不得创建 ThreadPoolExecutor")

    monkeypatch.setattr("siete_rl.rewards.ThreadPoolExecutor", boom)
    envs = [FakeEnvironment(i) for i in range(3)]
    assert binary_reward([None] * 3, envs) == [0.5, 1.5, 2.5]
    assert binary_reward([None], [FakeEnvironment(0)], max_workers=8) == [0.5]


def test_parallel_finalize_exception_does_not_corrupt_other_environments() -> None:
    envs = [
        FakeEnvironment(0),
        FakeEnvironment(1, error=DockerRuntimeError("verifier boom")),
        FakeEnvironment(2),
    ]
    with pytest.raises(DockerRuntimeError):
        binary_reward([None] * 3, envs, max_workers=3)
    assert envs[0].finalize_calls == 1 and envs[2].finalize_calls == 1

    envs_serial = [
        FakeEnvironment(0),
        FakeEnvironment(1, error=DockerRuntimeError("verifier boom")),
        FakeEnvironment(2),
    ]
    with pytest.raises(DockerRuntimeError):
        binary_reward([None] * 3, envs_serial)
    assert envs_serial[0].finalize_calls == 1 and envs_serial[2].finalize_calls == 0


def test_parallel_finalize_wall_time_covers_long_tail() -> None:
    def make_envs():
        return [FakeEnvironment(0, delay=1.0)] + [
            FakeEnvironment(i, delay=0.1) for i in range(1, 8)
        ]

    started = time.monotonic()
    serial = binary_reward([None] * 8, make_envs())
    serial_time = time.monotonic() - started
    started = time.monotonic()
    parallel = binary_reward([None] * 8, make_envs(), max_workers=8)
    parallel_time = time.monotonic() - started
    assert parallel == serial
    assert parallel_time < 0.75 * serial_time
