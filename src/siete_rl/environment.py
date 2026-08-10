"""TRL environment adapter：OpenHands 三工具、episode 状态与资源收束。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future
import re
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from siete_rl.docker import (
    ContainerCleanupError,
    DockerRuntimeError,
    DockerSandbox,
    WorkspaceStateError,
)
from siete_rl.models import (
    Action,
    Evaluation,
    LoopExit,
    Observation,
    Sample,
    Settlement,
    Step,
    TerminalEvent,
    Termination,
    Trajectory,
    Verification,
)
from siete_rl.process_mask import TurnRecord
from siete_rl.scoring import DEFAULT_LAMBDA, layered_score
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
        reset_executor: Executor | None = None,
    ) -> None:
        self._task_context = task_context
        self._sandbox_factory = sandbox_factory
        self._verifier_factory = verifier_factory
        self._output_limit_chars = output_limit_chars
        self._max_timeout_sec = max_timeout_sec
        self._reward_type = reward_type
        self._layered_lambda = layered_lambda
        self._reset_executor = reset_executor
        self._pending_reset: Future[None] | None = None
        self._last_reset_started_at: float | None = None
        self._last_reset_finished_at: float | None = None
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
        self._settlement: Settlement | None = None
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

    @property
    def settlement(self) -> Settlement | None:
        return self._settlement

    @property
    def scorable(self) -> bool:
        """该 episode 的 reward 是否能可靠归因给 policy。"""

        return (
            self._finalized
            and self._settlement is not None
            and self._settlement.status != "infra_error"
        )

    def reset(self, task_id: str, **kwargs: object) -> None:
        """Start a fresh repository episode for one public task ID."""

        self._await_reset()
        self._last_reset_started_at = perf_counter()
        self._last_reset_finished_at = None
        if self._reset_executor is not None:
            try:
                self._pending_reset = self._reset_executor.submit(
                    self._reset_now, task_id, **kwargs
                )
            except BaseException:
                self._last_reset_finished_at = perf_counter()
                raise
            return None
        return self._reset_now(task_id, **kwargs)

    def _reset_now(self, task_id: str, **kwargs: object) -> None:
        del kwargs
        try:
            self._close_resources()
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
            self._settlement = None
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
        finally:
            self._last_reset_finished_at = perf_counter()

    def _await_reset(self) -> None:
        pending = self._pending_reset
        if pending is None:
            return None
        try:
            return pending.result()
        finally:
            if self._pending_reset is pending:
                self._pending_reset = None

    def _reset_timing(self) -> tuple[float, float] | None:
        if (
            self._last_reset_started_at is None
            or self._last_reset_finished_at is None
        ):
            return None
        return self._last_reset_started_at, self._last_reset_finished_at

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
        """Record a submit event; final workspace capture happens at settlement."""
        return self._call_tool("finish", {})

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self._await_reset()
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
            self._submitted = True
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
        self._await_reset()
        del completion  # 终止原因不再从 completion 推导
        if self._finalized:
            if self._reward is None:
                raise RuntimeError("finalized environment is missing its reward")
            return self._reward
        if self._sample is None or self._evaluation is None or self.episode_id is None:
            raise RuntimeError("environment was finalized before reset")

        termination = self._termination()
        if termination == "infra_error":
            self._close_rollout()
            detail = str(self._infrastructure_error) if self._infrastructure_error else None
            return self._finish_settlement(
                termination, Settlement(status="infra_error", detail=detail), 0.0
            )

        capture_failure = self._capture_final_patch()
        self._close_rollout()
        if capture_failure is not None:
            return self._finish_settlement(termination, capture_failure, 0.0)

        if self._frozen_patch is None:
            raise RuntimeError("settled episode is missing its frozen patch")
        if not self._frozen_patch.strip():
            return self._finish_settlement(
                termination, Settlement(status="empty_patch"), 0.0
            )
        settlement, reward = self._verify_final_patch()
        return self._finish_settlement(termination, settlement, reward)

    def _termination(self) -> Termination:
        if self._terminal_event is not None and self._terminal_event.kind == "infra_error":
            return self._terminal_event.kind
        if self._loop_exit == "context_overlong":
            return self._loop_exit
        if self._terminal_event is not None:
            return self._terminal_event.kind
        if self._loop_exit is not None:
            return self._loop_exit
        if self._infrastructure_error is not None:
            return "infra_error"
        raise RuntimeError("environment finalized without a terminal event or loop exit")

    def _capture_final_patch(self) -> Settlement | None:
        if self._sandbox is None:
            raise RuntimeError("healthy episode is missing its rollout sandbox")
        try:
            self._frozen_patch = self._sandbox.get_diff()
        except WorkspaceStateError as exc:
            return Settlement(status="agent_error", detail=str(exc))
        except DockerRuntimeError as exc:
            self._infrastructure_error = exc
            return Settlement(status="infra_error", detail=str(exc))
        return None

    def _verify_final_patch(self) -> tuple[Settlement, float]:
        if self._frozen_patch is None or not self._frozen_patch.strip():
            raise RuntimeError("verifier requires a non-empty frozen patch")
        if self._verifier is None:
            self._verifier = self._verifier_factory(
                self._sample, self._evaluation, self.episode_id
            )
        try:
            self._verification = self._verifier.verify(self._frozen_patch)
        except DockerRuntimeError as exc:
            self._infrastructure_error = exc
            return Settlement(status="infra_error", detail=str(exc)), 0.0
        finally:
            self._events.extend(self._verifier.drain_cleanup_events())
        if self._verification.result == "resolved":
            return Settlement(status="resolved"), 1.0
        if self._reward_type == "layered":
            reward = layered_score(
                verification=self._verification,
                fail_to_pass=self._evaluation.fail_to_pass,
                pass_to_pass=self._evaluation.pass_to_pass,
                lambda_=self._layered_lambda,
            )
        else:
            reward = 0.0
        return Settlement(status="unresolved"), reward

    def _finish_settlement(
        self, termination: Termination, settlement: Settlement, reward: float
    ) -> float:
        self._settlement = settlement
        self._trajectory = Trajectory(
            task_id=self._sample.task.task_id,
            environment_id=self._sample.environment.environment_id,
            steps=list(self._steps),
            termination=termination,
            settlement=settlement,
        )
        self._reward = reward
        self._finalized = True
        return reward

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
        try:
            self._await_reset()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._close_resources()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            primary = errors[0]
            for extra in errors[1:]:
                primary.add_note(f"additional cleanup failure: {type(extra).__name__}: {extra}")
            raise primary

    def _close_resources(self) -> None:
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
