"""TRL environment adapter：OpenHands 三工具、episode 状态与资源收束。"""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any, Literal
from uuid import uuid4

from siete_rl.docker import ContainerCleanupError, DockerRuntimeError, DockerSandbox
from siete_rl.models import (
    Action,
    Evaluation,
    LoopExit,
    Observation,
    Sample,
    Step,
    TerminalEvent,
    Trajectory,
    Verification,
)
from siete_rl.process_mask import TurnRecord
from siete_rl.scoring import DEFAULT_LAMBDA, DEFAULT_MU, layered_score
from siete_rl.swegym import TaskContext
from siete_rl.tools import ToolContractError, ToolExecutor, validate_tool_arguments
from siete_rl.verifier import SWEGymVerifier


SandboxFactory = Callable[[Sample, str, Literal["rollout", "verifier"]], DockerSandbox]
VerifierFactory = Callable[[Sample, Evaluation, str], SWEGymVerifier]


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
        reward_type: Literal["binary", "layered"] = "binary",
        layered_lambda: float = DEFAULT_LAMBDA,
        layered_mu: float = DEFAULT_MU,
    ) -> None:
        self._task_context = task_context
        self._sandbox_factory = sandbox_factory
        self._verifier_factory = verifier_factory
        self._output_limit_chars = output_limit_chars
        self._max_timeout_sec = max_timeout_sec
        self._reward_type = reward_type
        self._layered_lambda = layered_lambda
        self._layered_mu = layered_mu
        self._sample: Sample | None = None
        self._evaluation: Evaluation | None = None
        self._sandbox: DockerSandbox | None = None
        self._executor: ToolExecutor | None = None
        self._verifier: SWEGymVerifier | None = None
        self._steps: list[Step] = []
        self._events: list[dict[str, object]] = []
        self._terminal_event: TerminalEvent | None = None
        self._loop_exit: LoopExit | None = None
        self._infrastructure_error: BaseException | None = None
        self._submitted = False
        self._frozen_patch: str | None = None
        self._finalized = False
        self._reward: float | None = None
        self._trajectory: Trajectory | None = None
        self._verification: Verification | None = None
        self.episode_id: str | None = None
        # rollout turn 记录：trainer 逐段写入，process mask 组装读取
        self.turn_records: list[TurnRecord] = []

    @property
    def trajectory(self) -> Trajectory | None:
        return self._trajectory

    @property
    def terminated(self) -> bool:
        """供 TRL tool-call loop 轮询的终止信号；property 不会被暴露为工具。"""

        return self._terminal_event is not None

    @property
    def verification(self) -> Verification | None:
        return self._verification

    @property
    def frozen_patch(self) -> str | None:
        return self._frozen_patch

    def reset(self, task_id: str, **kwargs: object) -> None:
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
        self.turn_records = []
        self._terminal_event = None
        self._loop_exit = None
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
            workspace = _workspace_for_task(task_id)
            _install_workspace_aliases(sandbox, workspace)
            self._executor = ToolExecutor(
                sandbox,
                output_limit_chars=self._output_limit_chars,
                max_timeout_sec=self._max_timeout_sec,
                workspace=workspace,
            )
        except DockerRuntimeError as exc:
            # 单样本基础设施失败不传播：episode 在第 0 步终止，由 _finalize 统一降级。
            self._infrastructure_error = exc
            self._terminal_event = TerminalEvent(kind="infra_error", step_index=0)
            try:
                self._close_rollout()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "rollout cleanup failed during reset: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        return None

    def execute_bash(self, command: str) -> str:
        """Execute one bash command in the episode workspace."""
        return self._call_tool("execute_bash", {"command": command})

    def str_replace_editor(
        self, command: str, path: str, file_text: str | None = None,
        old_str: str | None = None, new_str: str | None = None,
        insert_line: int | None = None, view_range: list[int] | None = None,
    ) -> str:
        """View or edit a container file with the OpenHands editor."""
        return self._call_tool("str_replace_editor", _without_none({"command": command, "path": path, "file_text": file_text, "old_str": old_str, "new_str": new_str, "insert_line": insert_line, "view_range": view_range}))

    def finish(self) -> str:
        """Freeze the current diff, including an empty diff, and terminate."""
        return self._call_tool("finish", {})

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._executor is None or self._sandbox is None or self.episode_id is None:
            raise DockerRuntimeError("environment has no active rollout container")
        if self._submitted:
            action = Action(tool_name=name, arguments=arguments)
            observation = Observation(
                text="The patch is already submitted; no further tool call is needed.",
                exit_code=1,
                error_type="tool_error",
            )
            self._append_step(action, observation)
            return observation.text
        try:
            validate_tool_arguments(name, arguments)
        except ToolContractError as exc:
            return str(exc)

        action = Action(tool_name=name, arguments=arguments)
        try:
            observation = self._executor.execute(action)
        except DockerRuntimeError as exc:
            self._infrastructure_error = exc
            self._terminal_event = TerminalEvent(
                kind="infra_error", step_index=len(self._steps)
            )
            raise
        self._append_step(action, observation)
        if name == "finish" and observation.error_type is None and not observation.timed_out:
            if self._executor.submitted_patch is None:
                raise RuntimeError("successful submit did not freeze a patch")
            self._submitted = True
            self._frozen_patch = self._executor.submitted_patch
            self._terminal_event = TerminalEvent(
                kind="submitted", step_index=len(self._steps) - 1
            )
        return observation.text

    def _record_loop_exit(self, reason: LoopExit) -> None:
        """由 trainer 在 tool-call loop 出口写回该样本的循环结束原因。"""

        self._loop_exit = reason

    def _append_step(self, action: Action, observation: Observation) -> None:
        self._steps.append(Step(index=len(self._steps), action=action, observation=observation))

    def _finalize(self, completion: object) -> float:
        del completion  # 终止原因不再从 completion 推导
        if self._finalized:
            if self._reward is None:
                raise RuntimeError("finalized environment is missing its reward")
            return self._reward
        if self._sample is None or self._evaluation is None or self.episode_id is None:
            raise RuntimeError("environment was finalized before reset")

        if self._terminal_event is not None:
            termination = self._terminal_event.kind
        elif self._loop_exit is not None:
            termination = self._loop_exit
        elif self._infrastructure_error is not None:
            termination = "infra_error"
        else:
            raise RuntimeError("environment finalized without a terminal event or loop exit")
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

        if self._frozen_patch is None:
            raise RuntimeError("submitted episode is missing its frozen patch")
        self._close_rollout()
        if not self._frozen_patch.strip():
            self._reward = 0.0
            self._finalized = True
            return self._reward
        if self._verifier is None:
            self._verifier = self._verifier_factory(
                self._sample, self._evaluation, self.episode_id
            )
        try:
            self._verification = self._verifier.verify(self._frozen_patch)
        except DockerRuntimeError as exc:
            # verifier 基础设施失败（容器、apply/pytest 超时等）降级为单样本 infra_error。
            self._infrastructure_error = exc
            self._trajectory = Trajectory(
                task_id=self._sample.task.task_id,
                environment_id=self._sample.environment.environment_id,
                steps=list(self._steps),
                termination="infra_error",
            )
            self._reward = 0.0
            self._finalized = True
            return self._reward
        finally:
            self._events.extend(self._verifier.drain_cleanup_events())
        if self._reward_type == "layered":
            self._reward = layered_score(
                verification=self._verification,
                fail_to_pass=self._evaluation.fail_to_pass,
                pass_to_pass=self._evaluation.pass_to_pass,
                lambda_=self._layered_lambda,
                mu=self._layered_mu,
            )
        else:
            self._reward = 1.0 if self._verification.result == "resolved" else 0.0
        self._finalized = True
        return self._reward

    def _close_rollout(self) -> None:
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        try:
            sandbox.close()
        except ContainerCleanupError:
            # close 失败保留容器 ID（docker.py 既有契约），下方事件 residual=True；
            # 外层 sweeper 会重试清理。瞬时 docker 抖动不应杀死整个训练 run
            # （见 run 20260727T131837Z-ea40 step 3 的 rm 30s 超时）。
            pass
        finally:
            self._events.append(
                {
                    "scope": "rollout",
                    "episode_id": self.episode_id,
                    "task_id": sandbox.environment.task_id,
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


def _without_none(arguments: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in arguments.items() if value is not None}


def _workspace_for_task(task_id: str) -> str:
    return "/workspace/" + re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)


def _install_workspace_aliases(sandbox: DockerSandbox, workspace: str) -> None:
    """以 argv 调用容器 Python helper；冲突的 alias 必须 fail-closed。"""
    program = """import os, pathlib, sys
workspace = pathlib.Path(sys.argv[1]); target = pathlib.Path('/testbed')
workspace.parent.mkdir(parents=True, exist_ok=True)
for alias in (workspace, pathlib.Path('/repo')):
    if alias.is_symlink() and alias.resolve() == target: continue
    if alias.exists() or alias.is_symlink(): raise SystemExit(f'alias conflict: {alias}')
    alias.symlink_to(target)
"""
    result = sandbox.exec(["python", "-c", program, workspace])
    if result.timed_out or result.exit_code != 0:
        raise DockerRuntimeError((result.stderr or result.stdout or "failed to create workspace aliases").strip())
