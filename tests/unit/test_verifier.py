from __future__ import annotations

from collections import deque
from typing import Sequence

import pytest

from siete_rl.docker import CommandResult, ContainerCleanupError, DockerRuntimeError
from siete_rl.models import Environment, Evaluation
from siete_rl.verifier import SWEGymVerifier, VerificationInfrastructureError

PATCH = "diff --git a/x b/x\n"
TASK_ID = "getmoto__moto-7023"


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
    def __init__(self, environment, responses: list[CommandResult], *, close_error: bool = False) -> None:
        self.environment = environment
        self.responses = deque(responses)
        self.close_error = close_error
        self.container_name = f"verifier-{id(self)}"
        self.container_id: str | None = "a" * 64
        self.acquired_container_id = self.container_id
        self.opened = False
        self.closed = False
        self.calls: list[tuple[list[str], str | None, int | None]] = []

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
        self.calls.append((list(command), input_text, timeout_sec))
        if not self.responses:
            raise DockerRuntimeError("exec plumbing failed")
        return self.responses.popleft()

    def close(self) -> None:
        if self.close_error:
            raise ContainerCleanupError("remove failed")
        self.closed = True
        self.container_id = None

    def drain_cleanup_operations(self):
        return [
            {
                "sequence": 1,
                "operation": "remove",
                "result": "failed" if self.close_error else "success",
            }
        ]


@pytest.fixture(scope="module")
def domain():
    environment = Environment(
        environment_id=f"swegym:{TASK_ID}",
        task_id=TASK_ID,
        image_name="image",
        expected_image_id="sha256:" + "1" * 64,
        expected_registry_digest="sha256:" + "2" * 64,
        workdir="/testbed",
        cpus=1,
        memory="1g",
        pids_limit=64,
        exec_timeout_sec=30,
        verifier_timeout_sec=60,
    )
    evaluation = Evaluation(
        offline_eval_script="set -eux\npytest -q\n",
        fail_to_pass=["tests/test_target.py::test_fix"],
        pass_to_pass=["tests/test_target.py::test_regression"],
    )
    return environment, evaluation


def verifier_for(domain, responses: list[CommandResult], *, close_error: bool = False):
    environment, evaluation = domain
    sandboxes: list[FakeSandbox] = []

    def factory():
        sandbox = FakeSandbox(environment, list(responses), close_error=close_error)
        sandboxes.append(sandbox)
        return sandbox

    verifier = SWEGymVerifier(
        sandbox_factory=factory,  # type: ignore[arg-type]
        evaluation=evaluation,
    )
    return verifier, sandboxes


def test_empty_patch_never_creates_verification_or_container(domain) -> None:
    verifier, sandboxes = verifier_for(domain, [])
    with pytest.raises(VerificationInfrastructureError, match="non-empty"):
        verifier.verify("")
    assert not sandboxes


def test_patch_check_and_apply_failures_are_attributable_unresolved(domain) -> None:
    checked, sandboxes = verifier_for(
        domain, [command_result(exit_code=1, stderr="does not apply")]
    )
    check_result = checked.verify(PATCH)
    assert check_result.result == "unresolved"
    assert check_result.patch_apply_status == "check_failed"
    assert sandboxes[0].closed

    applied, _ = verifier_for(
        domain,
        [command_result(), command_result(exit_code=1, stderr="apply failed")],
    )
    apply_result = applied.verify(PATCH)
    assert apply_result.result == "unresolved"
    assert apply_result.patch_apply_status == "apply_failed"


def test_pytest_marker_maps_exit_code_to_binary_result(domain) -> None:
    passed, _ = verifier_for(
        domain,
        [command_result(), command_result(), command_result(stdout="+ pytest -n0\n9 passed")],
    )
    assert passed.verify(PATCH).result == "resolved"

    failed, _ = verifier_for(
        domain,
        [
            command_result(),
            command_result(),
            command_result(exit_code=1, stdout="+ pytest -n0\n1 failed"),
        ],
    )
    assert failed.verify(PATCH).result == "unresolved"


@pytest.mark.parametrize(
    "responses",
    [
        [command_result(timed_out=True)],
        [command_result(), command_result(timed_out=True)],
        [
            command_result(),
            command_result(),
            command_result(timed_out=True, stdout="setup completed"),
        ],
        [command_result(), command_result(), command_result(stdout="setup completed")],
    ],
)
def test_timeout_and_missing_pytest_marker_are_infrastructure(domain, responses) -> None:
    verifier, sandboxes = verifier_for(domain, responses)
    with pytest.raises(VerificationInfrastructureError):
        verifier.verify(PATCH)
    assert sandboxes[0].closed


def test_timeout_after_pytest_started_is_unresolved(domain) -> None:
    verifier, sandboxes = verifier_for(
        domain,
        [
            command_result(),
            command_result(),
            command_result(
                exit_code=124,
                stdout="+ pytest -n0\n",
                timed_out=True,
            ),
        ],
    )

    verification = verifier.verify(PATCH)
    assert verification.result == "unresolved"
    assert verification.patch_apply_status == "applied"
    assert verification.pytest_started is True
    assert verification.exit_code == 124
    assert sandboxes[0].closed


def test_cleanup_failure_does_not_replace_successful_verification(domain) -> None:
    verifier, sandboxes = verifier_for(
        domain,
        [command_result(), command_result(), command_result(stdout="+ pytest\npassed")],
        close_error=True,
    )
    verification = verifier.verify(PATCH)
    assert verification.result == "resolved"
    events = verifier.drain_cleanup_events()
    assert events[0]["residual"] is True
    assert sandboxes[0].container_id is not None


def test_exec_and_cleanup_errors_keep_exec_as_primary(domain) -> None:
    verifier, _ = verifier_for(domain, [], close_error=True)
    with pytest.raises(DockerRuntimeError, match="exec plumbing") as captured:
        verifier.verify(PATCH)
    assert any("cleanup failed" in note for note in captured.value.__notes__)


def test_each_verification_uses_a_fresh_sandbox(domain) -> None:
    verifier, sandboxes = verifier_for(
        domain, [command_result(exit_code=1, stderr="does not apply")]
    )
    verifier.verify(PATCH)
    verifier.verify(PATCH)
    assert len(sandboxes) == 2
    assert sandboxes[0] is not sandboxes[1]
