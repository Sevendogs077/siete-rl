from __future__ import annotations

from collections import deque

import pytest

from siete_rl.docker import CommandResult
from siete_rl.models import Action
from siete_rl.tools import TOOL_SPECS, ToolExecutor, validate_tool_arguments


class FakeSandbox:
    def __init__(self) -> None:
        self.responses: deque[CommandResult] = deque()
        self.diff = ""
        self.commands = []
        self.get_diff_calls = 0

    def exec(self, command, *, input_text=None, timeout_sec=None):
        self.commands.append((command, input_text, timeout_sec))
        return self.responses.popleft()

    def get_diff(self) -> str:
        self.get_diff_calls += 1
        return self.diff


def result(stdout: str = "", stderr: str = "", exit_code: int = 0, timed_out: bool = False) -> CommandResult:
    return CommandResult(argv=[], exit_code=exit_code, stdout=stdout, stderr=stderr, duration_sec=0, timed_out=timed_out)


def test_exact_openhands_tool_schemas() -> None:
    assert tuple(spec.name for spec in TOOL_SPECS) == ("execute_bash", "finish", "str_replace_editor")


def test_execute_bash_uses_workspace_and_openhands_exit_text() -> None:
    sandbox = FakeSandbox(); sandbox.responses.append(result(stdout="ok", exit_code=7))
    observation = ToolExecutor(sandbox, output_limit_chars=30_000, max_timeout_sec=12, workspace="/workspace/task").execute(Action(tool_name="execute_bash", arguments={"command": "pwd"}))
    assert sandbox.commands == [(["/bin/bash", "-lc", "cd /workspace/task && pwd"], None, 12)]
    assert observation.text == "ok\n[Command finished with exit code 7]"
    assert observation.error_type == "tool_error"


def test_finish_does_not_capture_workspace_diff() -> None:
    sandbox = FakeSandbox(); sandbox.diff = ""
    executor = ToolExecutor(sandbox, output_limit_chars=30_000, max_timeout_sec=12, workspace="/workspace/task")
    assert executor.execute(Action(tool_name="finish", arguments={})).error_type is None
    assert sandbox.get_diff_calls == 0


def test_editor_errors_are_tool_errors_without_host_filesystem() -> None:
    sandbox = FakeSandbox(); sandbox.responses.append(result(stdout="missing\n"))
    executor = ToolExecutor(sandbox, output_limit_chars=30_000, max_timeout_sec=12, workspace="/workspace/task")
    observation = executor.execute(Action(tool_name="str_replace_editor", arguments={"command": "view", "path": "/repo/missing"}))
    assert observation.error_type == "tool_error"
    assert "does not exist" in observation.text


@pytest.mark.parametrize("name,args", [("execute_bash", {}), ("finish", {"extra": 1}), ("str_replace_editor", {"command": "view"})])
def test_argument_validation_rejects_invalid_contract(name, args) -> None:
    with pytest.raises(ValueError):
        validate_tool_arguments(name, args)
