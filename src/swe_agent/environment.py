"""TRL environment adapter：固定六工具、episode 状态与资源收束。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal
from uuid import uuid4

from swe_agent.docker import DockerRuntimeError, DockerSandbox
from swe_agent.models import (
    Action,
    Evaluation,
    Observation,
    Sample,
    Step,
    Trajectory,
    Verification,
)
from swe_agent.swegym import TaskContext
from swe_agent.tools import ToolContractError, ToolExecutor, validate_tool_arguments
from swe_agent.verifier import SWEGymVerifier


SandboxFactory = Callable[[Sample, str, Literal["rollout", "verifier"]], DockerSandbox]
VerifierFactory = Callable[[Evaluation, str], SWEGymVerifier]


class SWEEnvironment:
    """构造无副作用；只有 reset 才创建 rollout container。"""

    def __init__(
        self,
        *,
        task_context: TaskContext,
        sandbox_factory: SandboxFactory,
        verifier_factory: VerifierFactory,
        output_limit_chars: int,
        max_timeout_sec: int,
    ) -> None:
        self._task_context = task_context
        self._sandbox_factory = sandbox_factory
        self._verifier_factory = verifier_factory
        self._output_limit_chars = output_limit_chars
        self._max_timeout_sec = max_timeout_sec
        self._sample: Sample | None = None
        self._evaluation: Evaluation | None = None
        self._sandbox: DockerSandbox | None = None
        self._executor: ToolExecutor | None = None
        self._verifier: SWEGymVerifier | None = None
        self._steps: list[Step] = []
        self._events: list[dict[str, object]] = []
        self._termination: str | None = None
        self._infrastructure_error: BaseException | None = None
        self._submitted = False
        self._frozen_patch: str | None = None
        self._finalized = False
        self._reward: float | None = None
        self._trajectory: Trajectory | None = None
        self._verification: Verification | None = None
        self.episode_id: str | None = None

    @property
    def trajectory(self) -> Trajectory | None:
        return self._trajectory

    @property
    def verification(self) -> Verification | None:
        return self._verification

    @property
    def frozen_patch(self) -> str | None:
        return self._frozen_patch

    def reset(self, task_id: str, **kwargs: object) -> str:
        """Start a fresh repository episode for one public task ID."""

        del kwargs
        self._close()
        try:
            sample, evaluation = self._task_context[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown qualified task_id: {task_id}") from exc

        self._sample = sample
        self._evaluation = evaluation
        self.episode_id = uuid4().hex
        self._steps = []
        self._termination = None
        self._infrastructure_error = None
        self._submitted = False
        self._frozen_patch = None
        self._finalized = False
        self._reward = None
        self._trajectory = None
        self._verification = None
        sandbox = self._sandbox_factory(sample, self.episode_id, "rollout")
        self._sandbox = sandbox
        try:
            sandbox.open()
            self._executor = ToolExecutor(
                sandbox,
                output_limit_chars=self._output_limit_chars,
                max_timeout_sec=self._max_timeout_sec,
            )
        except BaseException as exc:
            self._infrastructure_error = exc
            self._termination = "infrastructure_interrupted"
            try:
                self._close_rollout()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "rollout cleanup failed during reset: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        return (
            f"Fresh repository for task {task_id} is ready at "
            f"{sample.environment.workdir}. Use the native tools to inspect and repair it."
        )

    def list_files(self, path: str, max_entries: int = 200) -> str:
        """List repository files with bounded output.

        Args:
            path: Repository-relative directory.
            max_entries: Maximum number of file paths to list.
        """

        return self._call_tool("list_files", {"path": path, "max_entries": max_entries})

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        """Read a numbered window from one UTF-8 repository file.

        Args:
            path: Repository-relative file path.
            start_line: Optional first line number, starting at one.
            end_line: Optional final line number, inclusive.
        """

        return self._call_tool(
            "read_file",
            _without_none({"path": path, "start_line": start_line, "end_line": end_line}),
        )

    def search_code(self, query: str, path: str = ".", max_matches: int = 50) -> str:
        """Search for exact text in repository files.

        Args:
            query: Exact text to search for.
            path: Repository-relative file or directory.
            max_matches: Maximum number of matching lines.
        """

        return self._call_tool(
            "search_code", {"query": query, "path": path, "max_matches": max_matches}
        )

    def edit_file(
        self,
        path: str,
        operation: str,
        old_text: str | None = None,
        new_text: str | None = None,
        content: str | None = None,
        line: int | None = None,
    ) -> str:
        """Edit one repository file with replace, insert, or create.

        Args:
            path: Repository-relative file path.
            operation: One of replace, insert, or create.
            old_text: Unique old text required by replace.
            new_text: Replacement or inserted text.
            content: Complete file content required by create.
            line: One-based insertion line required by insert.
        """

        return self._call_tool(
            "edit_file",
            _without_none(
                {
                    "path": path,
                    "operation": operation,
                    "old_text": old_text,
                    "new_text": new_text,
                    "content": content,
                    "line": line,
                }
            ),
        )

    def run_command(self, command: str, timeout_sec: int | None = None) -> str:
        """Run a diagnostic or public test command inside /testbed.

        Args:
            command: Shell command to execute.
            timeout_sec: Optional timeout capped by the configured maximum.
        """

        return self._call_tool(
            "run_command", _without_none({"command": command, "timeout_sec": timeout_sec})
        )

    def submit(self) -> str:
        """Submit the current non-empty git diff and enter terminal-pending state."""

        return self._call_tool("submit", {})

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._executor is None or self._sandbox is None or self.episode_id is None:
            raise DockerRuntimeError("environment has no active rollout container")
        if self._submitted:
            action = Action(tool_name=name, arguments=arguments)
            observation = Observation(
                text="A tool was called after submit; the episode is invalid.",
                exit_code=1,
                error_type="tool_error",
            )
            self._append_step(action, observation)
            self._termination = "invalid_after_submit"
            return observation.model_dump_json()
        try:
            validate_tool_arguments(name, arguments)
        except ToolContractError as exc:
            self._termination = "invalid_tool_call"
            return Observation(
                text=str(exc), exit_code=1, error_type="tool_error"
            ).model_dump_json()

        action = Action(tool_name=name, arguments=arguments)
        try:
            observation = self._executor.execute(action)
        except DockerRuntimeError as exc:
            self._infrastructure_error = exc
            self._termination = "infrastructure_interrupted"
            raise
        self._append_step(action, observation)
        if observation.timed_out:
            self._termination = "tool_timeout"
        elif observation.error_type is not None:
            self._termination = "invalid_tool_call"
        elif name == "submit":
            if self._executor.submitted_patch is None:
                raise RuntimeError("successful submit did not freeze a patch")
            self._submitted = True
            self._frozen_patch = self._executor.submitted_patch
        return observation.model_dump_json()

    def _append_step(self, action: Action, observation: Observation) -> None:
        self._steps.append(Step(index=len(self._steps), action=action, observation=observation))

    def _finalize(self, completion: object) -> float:
        if self._finalized:
            if self._reward is None:
                raise RuntimeError("finalized environment is missing its reward")
            return self._reward
        if self._infrastructure_error is not None:
            raise self._infrastructure_error
        if self._sample is None or self._evaluation is None or self.episode_id is None:
            raise RuntimeError("environment was finalized before reset")

        termination = self._derive_termination(completion)
        self._termination = termination
        self._trajectory = Trajectory(
            task_id=self._sample.task.task_id,
            environment_id=self._sample.environment.environment_id,
            steps=list(self._steps),
            termination=termination,
        )
        if termination != "submitted":
            self._close_rollout()
            self._reward = 0.0
            self._finalized = True
            return self._reward

        if self._frozen_patch is None or not self._frozen_patch.strip():
            raise RuntimeError("submitted episode is missing its frozen patch")
        self._close_rollout()
        if self._verifier is None:
            self._verifier = self._verifier_factory(self._evaluation, self.episode_id)
        try:
            self._verification = self._verifier.verify(self._frozen_patch)
        finally:
            self._events.extend(self._verifier.drain_cleanup_events())
        self._reward = 1.0 if self._verification.result == "resolved" else 0.0
        self._finalized = True
        return self._reward

    def _derive_termination(self, completion: object) -> str:
        if self._termination in {
            "invalid_after_submit",
            "invalid_tool_call",
            "tool_timeout",
            "infrastructure_interrupted",
        }:
            return self._termination
        wire_failure = _wire_failure(completion)
        if wire_failure is not None:
            return wire_failure
        if not self._submitted:
            if _ends_with_tool_call(completion):
                return "max_turns"
            return "no_tool_call"
        if _ends_with_tool_call(completion):
            return "invalid_after_submit"
        return "submitted"

    def _close_rollout(self) -> None:
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        try:
            sandbox.close()
        finally:
            self._events.append(
                {
                    "scope": "rollout",
                    "episode_id": self.episode_id,
                    "container_name": sandbox.container_name,
                    "container_id": getattr(sandbox, "acquired_container_id", sandbox.container_id),
                    "operations": sandbox.drain_cleanup_operations(),
                    "residual": sandbox.container_id is not None,
                }
            )
            if sandbox.container_id is None:
                self._sandbox = None
                self._executor = None

    def _close(self) -> None:
        errors: list[BaseException] = []
        if self._verifier is not None:
            try:
                self._verifier.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._events.extend(self._verifier.drain_cleanup_events())
            if self._verifier._active_sandbox is None:
                self._verifier = None
        try:
            self._close_rollout()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            primary = errors[0]
            for extra in errors[1:]:
                primary.add_note(f"additional cleanup failure: {type(extra).__name__}: {extra}")
            raise primary

    def _drain_events(self) -> list[dict[str, object]]:
        events = list(self._events)
        self._events.clear()
        return events


def _without_none(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in arguments.items() if value is not None}


def _ends_with_tool_call(completion: object) -> bool:
    if not isinstance(completion, list) or not completion:
        return False
    last = completion[-1]
    return isinstance(last, dict) and bool(last.get("tool_calls"))


def _wire_failure(completion: object) -> str | None:
    """只读核对被 TRL 拒绝、因而未进入 bound method 的 wire tool call。"""

    if not isinstance(completion, list):
        return None
    submitted = False
    for message in completion:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                return "invalid_tool_call"
            function = call.get("function")
            if not isinstance(function, dict):
                return "invalid_tool_call"
            name = function.get("name")
            arguments = function.get("arguments")
            if submitted:
                return "invalid_after_submit"
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return "invalid_tool_call"
            try:
                validate_tool_arguments(name, arguments)
            except ToolContractError:
                return "invalid_tool_call"
            if name == "submit":
                submitted = True
    return None
