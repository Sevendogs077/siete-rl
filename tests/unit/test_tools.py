from __future__ import annotations

from collections import deque
from typing import Sequence

import pytest

from swe_agent.docker import CommandResult, DockerRuntimeError
from swe_agent.models import Action
from swe_agent.tools import (
    BY_NAME,
    TOOL_SPECS,
    ToolContractError,
    ToolExecutor,
    native_tool_schemas,
    validate_tool_arguments,
)


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
    def __init__(self, responses: list[CommandResult] | None = None, *, diff: str = "") -> None:
        self.responses = deque(responses or [])
        self.diff = diff
        self.calls: list[tuple[list[str], str | None, int | None]] = []
        self.raise_infra = False

    def exec(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        self.calls.append((list(command), input_text, timeout_sec))
        if self.raise_infra:
            raise DockerRuntimeError("daemon unavailable")
        if not self.responses:
            raise AssertionError("missing fake response")
        return self.responses.popleft()

    def get_diff(self) -> str:
        if self.raise_infra:
            raise DockerRuntimeError("daemon unavailable")
        return self.diff


def test_exact_six_native_tool_schemas() -> None:
    assert tuple(spec.name for spec in TOOL_SPECS) == (
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_command",
        "submit",
    )
    schemas = native_tool_schemas()
    assert [schema["function"]["name"] for schema in schemas] == list(BY_NAME)
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False
        for schema in schemas
    )


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_file", {}, "missing"),
        ("read_file", {"path": "x", "unknown": 1}, "unknown"),
        ("read_file", {"path": "x", "start_line": True}, "integer"),
        ("read_file", {"path": "x", "start_line": 3, "end_line": 2}, "end_line"),
        ("edit_file", {"path": "x", "operation": "replace", "old_text": "x"}, "missing"),
        (
            "edit_file",
            {"path": "x", "operation": "create", "content": "", "new_text": "x"},
            "incompatible",
        ),
        ("submit", {"value": 1}, "unknown"),
        ("unknown", {}, "unsupported"),
    ],
)
def test_argument_validation_fails_closed(name: str, arguments: dict, message: str) -> None:
    with pytest.raises(ToolContractError, match=message):
        validate_tool_arguments(name, arguments)


def test_executor_file_read_and_output_truncation() -> None:
    sandbox = FakeSandbox([command_result(stdout="123456789")])
    executor = ToolExecutor(sandbox, output_limit_chars=5, max_timeout_sec=300)  # type: ignore[arg-type]
    action = Action(tool_name="read_file", arguments={"path": "moto/api.py"})
    observation = executor.execute(action)
    assert observation.exit_code == 0
    assert observation.truncated is True
    assert observation.text.startswith("12345")
    assert sandbox.calls[0][0][:2] == ["python", "-c"]
    assert sandbox.calls[0][0][-4:] == ["read", "moto/api.py", "1", "200"]


def test_edit_requires_nonempty_diff() -> None:
    action = Action(
        tool_name="edit_file",
        arguments={
            "path": "file.py",
            "operation": "replace",
            "old_text": "bad",
            "new_text": "good",
        },
    )
    failed = ToolExecutor(
        FakeSandbox([command_result()], diff=""),  # type: ignore[arg-type]
        output_limit_chars=1000,
        max_timeout_sec=300,
    ).execute(action)
    assert failed.error_type == "tool_error"
    assert "did not produce" in failed.text

    passed = ToolExecutor(
        FakeSandbox([command_result()], diff="diff --git a/file.py b/file.py"),  # type: ignore[arg-type]
        output_limit_chars=1000,
        max_timeout_sec=300,
    ).execute(action)
    assert passed.error_type is None
    assert "diff --git" in passed.text


def test_run_command_policy_timeout_and_cap() -> None:
    denied_sandbox = FakeSandbox()
    denied = ToolExecutor(
        denied_sandbox, output_limit_chars=1000, max_timeout_sec=30  # type: ignore[arg-type]
    ).execute(Action(tool_name="run_command", arguments={"command": "docker ps"}))
    assert denied.error_type == "tool_error"
    assert not denied_sandbox.calls

    sandbox = FakeSandbox([command_result(exit_code=124, stderr="timeout", timed_out=True)])
    observation = ToolExecutor(
        sandbox, output_limit_chars=1000, max_timeout_sec=30  # type: ignore[arg-type]
    ).execute(
        Action(
            tool_name="run_command",
            arguments={"command": "pytest -q", "timeout_sec": 999},
        )
    )
    assert observation.exit_code == 124
    assert observation.error_type == "tool_error"
    assert observation.timed_out is True
    assert sandbox.calls[0][2] == 30


def test_submit_freezes_exact_nonempty_patch() -> None:
    empty_executor = ToolExecutor(
        FakeSandbox(diff=""), output_limit_chars=1000, max_timeout_sec=30  # type: ignore[arg-type]
    )
    failed = empty_executor.execute(Action(tool_name="submit", arguments={}))
    assert failed.error_type == "tool_error"
    assert empty_executor.submitted_patch is None

    patch = "diff --git a/x b/x\n"
    executor = ToolExecutor(
        FakeSandbox(diff=patch), output_limit_chars=1000, max_timeout_sec=30  # type: ignore[arg-type]
    )
    passed = executor.execute(Action(tool_name="submit", arguments={}))
    assert passed.error_type is None
    assert executor.submitted_patch == patch


def test_docker_infrastructure_error_is_not_converted_to_tool_error() -> None:
    sandbox = FakeSandbox()
    sandbox.raise_infra = True
    executor = ToolExecutor(
        sandbox, output_limit_chars=1000, max_timeout_sec=30  # type: ignore[arg-type]
    )
    with pytest.raises(DockerRuntimeError, match="daemon unavailable"):
        executor.execute(Action(tool_name="read_file", arguments={"path": "x"}))
