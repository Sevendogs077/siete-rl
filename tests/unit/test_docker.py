from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Sequence

import pytest

from swe_agent.config import load_config
from swe_agent.docker import (
    CommandResult,
    ContainerCleanupError,
    ContainerCreateError,
    ContainerExecError,
    DockerSandbox,
    build_create_command,
    inspect_image,
    sweep_run_containers,
)
from swe_agent.swegym import load_qualified_instance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"
CONTAINER_ID = "a" * 64


def result(
    argv: Sequence[str],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> CommandResult:
    return CommandResult(
        argv=list(argv),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_sec=0.01,
        timed_out=timed_out,
    )


class FakeClient:
    def __init__(self, responses: list[CommandResult]) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[list[str], str | None, int]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int,
    ) -> CommandResult:
        command = list(argv)
        self.calls.append((command, input_text, timeout_sec))
        if not self.responses:
            raise AssertionError(f"unexpected Docker call: {command}")
        response = self.responses.popleft()
        return CommandResult(
            argv=command,
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            duration_sec=response.duration_sec,
            timed_out=response.timed_out,
        )


@pytest.fixture(scope="module")
def domain():
    config, project_root, _ = load_config(CONFIG_PATH)
    sample, _ = load_qualified_instance(config, project_root)
    return sample.task, sample.environment


def image_inspect(environment, **updates: object) -> CommandResult:
    payload = {
        "Id": environment.expected_image_id,
        "Os": "linux",
        "Architecture": "amd64",
        "RepoDigests": [],
        "RepoTags": [environment.image_name],
        "Size": 2_849_787_451,
        **updates,
    }
    return result([], stdout=json.dumps([payload]))


def ready_responses(task, environment) -> list[CommandResult]:
    return [
        image_inspect(environment),
        result([], stdout=CONTAINER_ID + "\n"),
        result([], stdout=CONTAINER_ID + "\n"),
        result([], stdout=task.base_commit + "\n"),
        result([], stdout=""),
    ]


def make_sandbox(client: FakeClient, task, environment) -> DockerSandbox:
    return DockerSandbox(
        client=client,
        task=task,
        environment=environment,
        run_id="run",
        episode_id="episode",
        scope="rollout",
    )


def test_create_command_has_fixed_isolation_and_no_mount(domain) -> None:
    task, environment = domain
    sandbox = make_sandbox(FakeClient([]), task, environment)
    command = build_create_command(
        name=sandbox.container_name,
        environment=environment,
        labels=sandbox.labels,
    )
    assert command[:2] == ["docker", "create"]
    assert "--pull=never" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert not ({"-v", "--volume", "--mount"} & set(command))
    assert environment.image_name in command


def test_open_saves_id_before_start_and_closes_by_id(domain) -> None:
    task, environment = domain
    client = FakeClient([*ready_responses(task, environment), result([], stdout=CONTAINER_ID)])
    sandbox = make_sandbox(client, task, environment)
    sandbox.open()
    assert sandbox.container_id == CONTAINER_ID
    assert client.calls[2][0] == ["docker", "start", CONTAINER_ID]
    sandbox.close()
    assert client.calls[-1][0] == ["docker", "rm", "-f", CONTAINER_ID]
    assert sandbox.container_id is None


def test_start_failure_preserves_primary_and_runs_cleanup(domain) -> None:
    task, environment = domain
    client = FakeClient(
        [
            image_inspect(environment),
            result([], stdout=CONTAINER_ID),
            result([], exit_code=1, stderr="start failed"),
            result([], stdout=CONTAINER_ID),
        ]
    )
    sandbox = make_sandbox(client, task, environment)
    with pytest.raises(ContainerCreateError, match="failed to start"):
        sandbox.open()
    assert client.calls[-1][0] == ["docker", "rm", "-f", CONTAINER_ID]
    assert sandbox.container_id is None


def test_base_mismatch_is_primary_and_container_is_removed(domain) -> None:
    task, environment = domain
    responses = ready_responses(task, environment)
    responses[3] = result([], stdout="0" * 40)
    responses.append(result([], stdout=CONTAINER_ID))
    sandbox = make_sandbox(FakeClient(responses), task, environment)
    with pytest.raises(ContainerCreateError, match="base commit mismatch"):
        sandbox.open()
    assert sandbox.container_id is None


def test_cleanup_failure_retains_handle_and_next_close_retries(domain) -> None:
    task, environment = domain
    client = FakeClient(
        [
            *ready_responses(task, environment),
            result([], exit_code=1, stderr="daemon unavailable"),
            result([], stdout=CONTAINER_ID),
        ]
    )
    sandbox = make_sandbox(client, task, environment).open()
    with pytest.raises(ContainerCleanupError, match="daemon unavailable"):
        sandbox.close()
    assert sandbox.container_id == CONTAINER_ID
    first_operations = sandbox.drain_cleanup_operations()
    assert first_operations[0]["result"] == "failed"
    sandbox.close()
    assert sandbox.container_id is None
    assert sandbox.drain_cleanup_operations()[0]["sequence"] == 2
    calls_after_success = len(client.calls)
    sandbox.close()
    assert len(client.calls) == calls_after_success


def test_ambiguous_create_recovers_exact_id_then_cleans(domain) -> None:
    task, environment = domain
    client = FakeClient(
        [
            image_inspect(environment),
            result([], exit_code=1, stderr="timeout", timed_out=True),
            result([], stdout=json.dumps([{"Id": CONTAINER_ID}])),
            result([], stdout=CONTAINER_ID),
        ]
    )
    sandbox = make_sandbox(client, task, environment)
    with pytest.raises(ContainerCreateError, match="failed to create"):
        sandbox.open()
    assert sandbox.container_id is None
    assert client.calls[-1][0] == ["docker", "rm", "-f", CONTAINER_ID]


def test_image_identity_mismatch_fails_before_create(domain) -> None:
    task, environment = domain
    client = FakeClient([image_inspect(environment, Id="sha256:" + "0" * 64)])
    with pytest.raises(ContainerCreateError, match="image ID"):
        inspect_image(client, environment)
    assert len(client.calls) == 1


def test_get_diff_registers_untracked_files_before_diff(domain) -> None:
    task, environment = domain
    client = FakeClient(
        [
            result([], stdout=""),
            result([], stdout="diff --git a/new.py b/new.py\n"),
        ]
    )
    sandbox = make_sandbox(client, task, environment)
    sandbox.container_id = CONTAINER_ID
    sandbox.started = True

    diff = sandbox.get_diff()

    assert diff == "diff --git a/new.py b/new.py\n"
    assert client.calls[0][0][:4] == ["docker", "exec", "-i", CONTAINER_ID]
    assert "add" in client.calls[0][0] and "-N" in client.calls[0][0]
    assert "diff" in client.calls[1][0]


def test_get_diff_fails_when_untracked_registration_fails(domain) -> None:
    task, environment = domain
    client = FakeClient([result([], exit_code=1, stderr="index lock")])
    sandbox = make_sandbox(client, task, environment)
    sandbox.container_id = CONTAINER_ID
    sandbox.started = True

    with pytest.raises(ContainerExecError, match="failed to register untracked"):
        sandbox.get_diff()


def test_sweep_removes_only_labeled_containers() -> None:
    client = FakeClient(
        [
            result([], stdout=f"{'a' * 64}\n{'b' * 64}\n"),
            result([]),
            result([]),
        ]
    )

    removed = sweep_run_containers(client, "run-1")

    assert removed == ["a" * 64, "b" * 64]
    assert client.calls[0][0] == [
        "docker",
        "ps",
        "-aq",
        "--filter",
        "label=swe_agent.run_id=run-1",
    ]
    assert client.calls[1][0] == ["docker", "rm", "-f", "a" * 64]
    assert client.calls[2][0] == ["docker", "rm", "-f", "b" * 64]


def test_sweep_tolerates_empty_and_missing_but_fails_on_errors() -> None:
    empty = FakeClient([result([], stdout="")])
    assert sweep_run_containers(empty, "run-1") == []

    missing = FakeClient(
        [result([], stdout="a" * 64), result([], exit_code=1, stderr="No such container")]
    )
    assert sweep_run_containers(missing, "run-1") == []

    listing_failed = FakeClient([result([], exit_code=1, stderr="daemon down")])
    with pytest.raises(ContainerCleanupError, match="failed to list"):
        sweep_run_containers(listing_failed, "run-1")

    remove_failed = FakeClient(
        [result([], stdout="a" * 64), result([], exit_code=1, stderr="permission denied")]
    )
    with pytest.raises(ContainerCleanupError, match="failed to remove"):
        sweep_run_containers(remove_failed, "run-1")
