from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path
from typing import Sequence

import pytest

from swe_agent.config import load_config
from swe_agent.docker import CommandResult, ContainerCleanupError, DockerRuntimeError
from swe_agent.environment import SWEEnvironment
from swe_agent.models import Evaluation, Verification
from swe_agent.rewards import binary_reward
from swe_agent.swegym import load_task_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


def command_result(
    *, exit_code: int = 0, stdout: str = "", stderr: str = "", timed_out: bool = False
) -> CommandResult:
    return CommandResult(
        argv=[],
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_sec=0.1,
        timed_out=timed_out,
    )


class FakeSandbox:
    def __init__(self, sample, episode_id: str, scope: str) -> None:
        self.task = sample.task
        self.environment = sample.environment
        self.episode_id = episode_id
        self.scope = scope
        self.container_name = f"{scope}-{episode_id}"
        self.container_id: str | None = "a" * 64
        self.responses: deque[CommandResult] = deque()
        self.diff = ""
        self.opened = False
        self.close_calls = 0
        self.close_failures = 0
        self.cleanup_sequence = 0
        self.cleanup_buffer: list[dict[str, object]] = []
        self.raise_infra = False

    def open(self):
        self.opened = True
        return self

    def exec(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        del command, input_text, timeout_sec
        if self.raise_infra:
            raise DockerRuntimeError("daemon unavailable")
        if not self.responses:
            raise AssertionError("missing fake response")
        return self.responses.popleft()

    def get_diff(self) -> str:
        if self.raise_infra:
            raise DockerRuntimeError("daemon unavailable")
        return self.diff

    def close(self) -> None:
        self.close_calls += 1
        if self.container_id is None:
            return
        self.cleanup_sequence += 1
        if self.close_failures:
            self.close_failures -= 1
            self.cleanup_buffer.append({"sequence": self.cleanup_sequence, "result": "failed"})
            raise ContainerCleanupError("remove failed")
        self.cleanup_buffer.append({"sequence": self.cleanup_sequence, "result": "success"})
        self.container_id = None

    def drain_cleanup_operations(self):
        operations = list(self.cleanup_buffer)
        self.cleanup_buffer.clear()
        return operations


class FakeVerifier:
    def __init__(self, result: str = "resolved") -> None:
        self.result = result
        self.verify_calls = 0
        self.close_calls = 0
        self.cleanup_events: list[dict[str, object]] = []
        self._active_sandbox = None

    def verify(self, patch: str) -> Verification:
        assert patch.strip()
        self.verify_calls += 1
        return Verification(
            result=self.result,
            patch_apply_status="applied",
            pytest_started=True,
            exit_code=0 if self.result == "resolved" else 1,
            stdout="+ pytest",
            stderr="",
        )

    def close(self) -> None:
        self.close_calls += 1

    def drain_cleanup_events(self):
        events = list(self.cleanup_events)
        self.cleanup_events.clear()
        return events


@pytest.fixture
def harness():
    config, project_root, _ = load_config(CONFIG_PATH)
    context = load_task_context(config, project_root)
    sandboxes: list[FakeSandbox] = []
    verifiers: list[FakeVerifier] = []

    def sandbox_factory(sample, episode_id, scope):
        sandbox = FakeSandbox(sample, episode_id, scope)
        sandboxes.append(sandbox)
        return sandbox

    def verifier_factory(evaluation: Evaluation, episode_id: str):
        del evaluation, episode_id
        verifier = FakeVerifier()
        verifiers.append(verifier)
        return verifier

    environment = SWEEnvironment(
        task_context=context,
        sandbox_factory=sandbox_factory,  # type: ignore[arg-type]
        verifier_factory=verifier_factory,  # type: ignore[arg-type]
        output_limit_chars=config.chat.max_observation_chars,
        max_timeout_sec=config.docker.exec_timeout_sec,
    )
    return config, environment, sandboxes, verifiers


def test_constructor_has_no_side_effect_and_exact_six_methods_are_tools(harness) -> None:
    _, environment, sandboxes, _ = harness
    assert not sandboxes
    methods = [
        name
        for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name != "reset" and not name.startswith("_")
    ]
    assert methods == [
        "edit_file",
        "list_files",
        "read_file",
        "run_command",
        "search_code",
        "submit",
    ]


def test_reset_creates_fresh_sandbox_only_after_old_cleanup(harness) -> None:
    config, environment, sandboxes, _ = harness
    environment.reset(config.dataset.task_id, prompt="ignored")
    first = sandboxes[0]
    environment.reset(config.dataset.task_id)
    assert first.container_id is None
    assert len(sandboxes) == 2
    assert sandboxes[1] is not first


def test_cleanup_failure_blocks_next_reset_and_preserves_handle(harness) -> None:
    config, environment, sandboxes, _ = harness
    environment.reset(config.dataset.task_id)
    sandboxes[0].close_failures = 1
    with pytest.raises(ContainerCleanupError, match="remove failed"):
        environment.reset(config.dataset.task_id)
    assert len(sandboxes) == 1
    assert sandboxes[0].container_id is not None
    environment._close()
    assert sandboxes[0].container_id is None


def test_multiple_calls_record_contiguous_steps_and_policy_failure(harness) -> None:
    config, environment, sandboxes, verifiers = harness
    environment.reset(config.dataset.task_id)
    sandboxes[0].responses.extend(
        [command_result(stdout="one.py"), command_result(stdout="1: content")]
    )
    environment.list_files(".")
    environment.read_file("one.py")
    reward = environment._finalize([{"role": "assistant", "content": "done"}])
    assert reward == 0.0
    assert environment.trajectory is not None
    assert [step.index for step in environment.trajectory.steps] == [0, 1]
    assert environment.trajectory.termination == "model_stopped"
    assert not verifiers


def test_submit_final_turn_verifies_once_and_is_idempotent(harness) -> None:
    config, environment, sandboxes, verifiers = harness
    environment.reset(config.dataset.task_id)
    sandboxes[0].diff = "diff --git a/x b/x\n"
    environment.submit()
    completion = [{"role": "assistant", "content": "final", "tool_calls": []}]
    assert environment._finalize(completion) == 1.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "submitted"
    assert environment.verification is not None
    assert environment.frozen_patch == sandboxes[0].diff
    assert environment._finalize(completion) == 1.0
    assert verifiers[0].verify_calls == 1


def test_invalid_arguments_do_not_lock_termination(harness) -> None:
    config, environment, sandboxes, verifiers = harness
    environment.reset(config.dataset.task_id)
    invalid = environment.read_file("x", start_line=3, end_line=2)
    assert "end_line" in invalid
    assert environment._finalize([]) == 0.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "model_stopped"
    assert not verifiers


def test_finalize_uses_terminal_event_or_loop_exit(harness) -> None:
    config, environment, sandboxes, verifiers = harness
    environment.reset(config.dataset.task_id)
    assert not environment.terminated
    unknown = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "unknown_tool", "arguments": {}},
                }
            ],
        }
    ]
    assert environment._finalize(unknown) == 0.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "model_stopped"
    assert environment.trajectory.steps == []
    assert not verifiers

    environment.reset(config.dataset.task_id)
    environment._record_loop_exit("iteration_cap")
    assert environment._finalize([]) == 0.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "iteration_cap"

    environment.reset(config.dataset.task_id)
    sandboxes[-1].diff = "diff --git a/x b/x\n"
    environment.submit()
    assert environment.terminated
    environment.list_files(".")
    assert environment._finalize([]) == 1.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "submitted"
    assert verifiers[-1].verify_calls == 1


def test_tool_error_does_not_lock_termination_and_submit_still_wins(harness) -> None:
    config, environment, sandboxes, verifiers = harness
    environment.reset(config.dataset.task_id)
    sandboxes[0].responses.append(command_result(exit_code=1, stderr="boom"))
    result = environment.run_command("cat missing.py")
    assert "boom" in result
    assert not environment.terminated
    sandboxes[0].diff = "diff --git a/x b/x\n"
    environment.submit()
    assert environment._finalize([]) == 1.0
    assert environment.trajectory is not None
    assert environment.trajectory.termination == "submitted"
    assert verifiers[-1].verify_calls == 1


def test_docker_infrastructure_error_propagates_and_runner_close_releases(harness) -> None:
    config, environment, sandboxes, _ = harness
    environment.reset(config.dataset.task_id)
    sandboxes[0].raise_infra = True
    with pytest.raises(DockerRuntimeError, match="daemon unavailable"):
        environment.read_file("x")
    environment._close()
    assert sandboxes[0].container_id is None


def test_binary_reward_is_position_aligned_and_strict(harness) -> None:
    config, first, first_sandboxes, _ = harness
    second = SWEEnvironment(
        task_context=first._task_context,
        sandbox_factory=first._sandbox_factory,
        verifier_factory=first._verifier_factory,
        output_limit_chars=first._output_limit_chars,
        max_timeout_sec=first._max_timeout_sec,
    )
    first.reset(config.dataset.task_id)
    second.reset(config.dataset.task_id)
    first_sandboxes[0].diff = "diff --git a/x b/x\n"
    first.submit()
    rewards = binary_reward(
        completions=[
            [{"role": "assistant", "content": "final"}],
            [{"role": "assistant", "content": "no patch"}],
        ],
        environments=[first, second],
    )
    assert rewards == [1.0, 0.0]
    assert first_sandboxes[1].container_id is None
    with pytest.raises(ValueError, match="counts"):
        binary_reward(completions=[], environments=[first])
