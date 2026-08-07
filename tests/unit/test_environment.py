from __future__ import annotations

import inspect
from collections import deque

import pytest

from siete_rl.docker import (
    CommandResult,
    ContainerCleanupError,
    ContainerCreateError,
    ContainerExecError,
    DockerRuntimeError,
)
from siete_rl.environment import SWEEnvironment
from siete_rl.models import Environment, Evaluation, Sample, Task, Verification
from siete_rl.verifier import VerificationInfrastructureError


class Sandbox:
    def __init__(self, sample, episode_id, scope) -> None:
        self.environment = sample.environment; self.episode_id = episode_id; self.scope = scope
        self.container_id = "a" * 64; self.container_name = "fake"; self.responses = deque(); self.diff = ""; self.closed = 0
    def open(self): return self
    def exec(self, command, *, input_text=None, timeout_sec=None):
        if command[:2] == ["python", "-c"] and len(command) == 4:
            return CommandResult(argv=[], exit_code=0, stdout="", stderr="", duration_sec=0, timed_out=False)
        return self.responses.popleft()
    def get_diff(self): return self.diff
    def close(self): self.closed += 1; self.container_id = None
    def drain_cleanup_operations(self): return []


class Verifier:
    _active_sandbox = None
    def __init__(self, verification=None): self.calls = 0; self._verification = verification
    def verify(self, patch):
        self.calls += 1
        return self._verification or Verification(result="resolved", patch_apply_status="applied", pytest_started=True, exit_code=0, stdout="", stderr="")
    def close(self): pass
    def drain_cleanup_events(self): return []


def harness(verification=None, evaluation=None, verifier_cls=None, **env_kwargs):
    task = Task(task_id="owner/repo", repo_name="owner/repo", base_commit="0" * 40, problem_statement="fix")
    environment = Environment(environment_id="id", task_id=task.task_id, image_name="image", expected_image_id="sha256:" + "0" * 64, expected_registry_digest="sha256:" + "0" * 64, workdir="/testbed", cpus=1, memory="1g", pids_limit=1, exec_timeout_sec=1, verifier_timeout_sec=1)
    sandboxes = []; verifiers = []
    def make_sandbox(*args):
        value = Sandbox(*args); sandboxes.append(value); return value
    def make_verifier(*args):
        value = (verifier_cls or Verifier)(verification); verifiers.append(value); return value
    return SWEEnvironment(task_context={task.task_id: (Sample(task=task, environment=environment), evaluation or Evaluation(offline_eval_script="echo"))}, sandbox_factory=make_sandbox, verifier_factory=make_verifier, output_limit_chars=30000, max_timeout_sec=10, **env_kwargs), task.task_id, sandboxes, verifiers


def _make_submitted_env(reward_type="binary"):
    """构造已提交 episode：unresolved + applied + pytest 摘要含 PASSED/FAILED。"""

    verification = Verification(
        result="unresolved",
        patch_apply_status="applied",
        pytest_started=True,
        exit_code=1,
        stdout="PASSED test_a\nPASSED test_c\nFAILED test_b - assert 1 == 2\n",
        stderr="",
    )
    evaluation = Evaluation(
        offline_eval_script="echo",
        fail_to_pass=["test_a", "test_b"],
        pass_to_pass=["test_c"],
    )
    env, task_id, sandboxes, _ = harness(verification=verification, evaluation=evaluation, reward_type=reward_type)
    env.reset(task_id)
    sandboxes[0].diff = "diff --git a/x b/x\n"
    env.finish()
    return env


def test_only_three_public_tools_and_reset_is_silent() -> None:
    env, task_id, sandboxes, _ = harness()
    methods = [name for name, member in inspect.getmembers(env, predicate=inspect.ismethod) if not name.startswith("_") and name != "reset"]
    assert methods == ["execute_bash", "finish", "str_replace_editor"]
    assert env.reset(task_id, prompt="ignored") is None
    assert sandboxes


def test_finish_freezes_empty_patch_without_verifier() -> None:
    env, task_id, _, verifiers = harness(); env.reset(task_id)
    assert env.finish() == ""
    assert env.terminated
    assert env.frozen_patch == ""
    assert env._finalize([]) == 0.0
    assert not verifiers


def test_finish_nonempty_patch_verifies_once() -> None:
    env, task_id, sandboxes, verifiers = harness(); env.reset(task_id); sandboxes[0].diff = "diff --git a/x b/x\n"
    env.finish()
    assert env._finalize([]) == 1.0
    assert env._finalize([]) == 1.0
    assert verifiers[0].calls == 1


def test_finalize_layered_gives_partial_score() -> None:
    env = _make_submitted_env(reward_type="layered")
    reward = env._finalize(completion=None)
    assert 0.0 < reward < 1.0


def test_finalize_binary_unchanged_by_default() -> None:
    env = _make_submitted_env()  # 默认 reward_type="binary"
    reward = env._finalize(completion=None)
    assert reward in (0.0, 1.0)


def test_close_rollout_cleanup_failure_is_not_fatal() -> None:
    """docker rm 瞬时失败不应杀死训练 run：容器 ID 保留（residual），交给外层 sweeper 重试。"""
    env, task_id, sandboxes, _ = harness()
    env.reset(task_id)

    def failing_close() -> None:
        raise ContainerCleanupError("failed to remove container (exit=124, timeout=True)")

    sandboxes[0].close = failing_close
    env.finish()

    assert env._finalize([]) == 0.0
    event = env._events[-1]
    assert event["scope"] == "rollout" and event["residual"] is True


def test_finalize_degrades_tool_infra_error_to_zero_reward() -> None:
    """finish 时 get_diff 失败（如容器内 gitlink 状态）：单样本记 infra_error，不再传播杀 run。"""
    env, task_id, sandboxes, _ = harness()
    env.reset(task_id)

    def failing_diff() -> str:
        raise ContainerExecError("failed to register untracked files (exit=128)")

    sandboxes[0].get_diff = failing_diff
    with pytest.raises(DockerRuntimeError):
        env.finish()  # 生成循环（trainer）会接住此异常并继续轮询 terminated

    assert env.terminated
    assert env._finalize([]) == 0.0
    assert env.trajectory.termination == "infra_error"


def test_finalize_degrades_verifier_infra_error() -> None:
    """verifier 基础设施失败（如离线 pytest 超时）：记 infra_error + reward 0，不传播。"""

    class FailingVerifier(Verifier):
        def verify(self, patch):
            raise VerificationInfrastructureError("offline pytest evaluation timed out")

    env, task_id, sandboxes, _ = harness(verifier_cls=FailingVerifier)
    env.reset(task_id)
    sandboxes[0].diff = "diff --git a/x b/x\n"
    env.finish()

    assert env._finalize([]) == 0.0
    assert env.trajectory.termination == "infra_error"


def test_turn_records_reset_clears_list() -> None:
    """turn_records 由 trainer 逐段写入：reset 必须清空，避免跨 episode 串台。"""
    from siete_rl.process_mask import TurnRecord

    env, task_id, _, _ = harness()
    env.reset(task_id)
    assert env.turn_records == []
    env.turn_records.append(TurnRecord(0, 1, "plain_message", None))
    env.reset(task_id)
    assert env.turn_records == []


def test_reset_infra_failure_terminates_episode_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """rollout 容器创建失败：episode 在第 0 步终止并降级，不向 trainer 传播。"""

    def failing_open(self):
        raise ContainerCreateError("failed to start container (exit=125)")

    monkeypatch.setattr(Sandbox, "open", failing_open)
    env, task_id, _, _ = harness()

    assert env.reset(task_id) is None
    assert env.terminated
    assert env._finalize([]) == 0.0
    assert env.trajectory.termination == "infra_error"
