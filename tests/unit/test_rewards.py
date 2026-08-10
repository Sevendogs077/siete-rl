"""binary_reward 并行 finalize 的保序、异常隔离与调度策略。"""

from __future__ import annotations

import pytest

from siete_rl.docker import DockerRuntimeError
from siete_rl.models import Settlement
from siete_rl.rewards import binary_reward


class FakeEnvironment:
    """按 index 可识别的 _finalize；记录调用次数用于隔离断言。"""

    def __init__(
        self,
        index: int,
        *,
        error: BaseException | None = None,
        settlement: str = "unresolved",
    ) -> None:
        self.index = index
        self.error = error
        self.finalize_calls = 0
        self.settlement = Settlement(status=settlement)

    def _finalize(self, completion: object) -> float:
        self.finalize_calls += 1
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


def test_parallel_finalize_uses_pool_bounded_by_environment_count(monkeypatch) -> None:
    created: list[object] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            self.jobs: list[tuple[FakeEnvironment, object]] = []
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def map(self, function, jobs):
            self.jobs = list(jobs)
            return [function(job) for job in self.jobs]

    monkeypatch.setattr("siete_rl.rewards.ThreadPoolExecutor", RecordingExecutor)
    envs = [FakeEnvironment(i) for i in range(3)]

    rewards = binary_reward([None] * 3, envs, max_workers=99)

    executor = created[0]
    assert executor.max_workers == 3
    assert [environment for environment, _ in executor.jobs] == envs
    assert rewards == [0.5, 1.5, 2.5]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_infra_error_is_unscorable_instead_of_zero_reward(max_workers: int) -> None:
    envs = [
        FakeEnvironment(0, settlement="infra_error"),
        FakeEnvironment(1),
    ]

    assert binary_reward([None, None], envs, max_workers=max_workers) == [None, 1.5]
