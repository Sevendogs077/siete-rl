"""固定 SWE-Gym 镜像的最小、可重试 Docker 容器边界。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, Sequence
from uuid import uuid4

from siete_rl.models import Environment, Task


DEFAULT_DOCKER_HOST = "unix:///run/docker-swegym/docker.sock"
"""项目唯一 Docker daemon：专用 docker-swegym；与共享 daemon 完全撇清。"""


class DockerRuntimeError(RuntimeError):
    """Docker daemon、镜像、容器或 exec plumbing 基础设施错误。"""


class ContainerCreateError(DockerRuntimeError):
    pass


class ContainerExecError(DockerRuntimeError):
    pass


class ContainerCleanupError(DockerRuntimeError):
    pass


class WorkspaceStateError(RuntimeError):
    """容器内 workspace Git 已启动，但最终状态无法可靠读取。"""


_WORKSPACE_GIT_STARTED = "__SIETE_WORKSPACE_GIT_STARTED__"


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False


class DockerClient(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int,
    ) -> CommandResult: ...


class SubprocessDockerClient:
    """唯一真实 CLI 边界；本类没有 pull/build/load/rmi/prune 操作。

    每次调用都显式钉死 DOCKER_HOST 到专用 docker-swegym daemon，
    不继承也不回落到共享 daemon；可用 SWE_AGENT_DOCKER_HOST 覆盖（调试用）。
    """

    def __init__(self, docker_host: str | None = None) -> None:
        self.docker_host = (
            docker_host or os.environ.get("SWE_AGENT_DOCKER_HOST") or DEFAULT_DOCKER_HOST
        )
        if self.docker_host.startswith("unix://"):
            socket_path = self.docker_host.removeprefix("unix://")
            if not os.path.exists(socket_path):
                raise DockerRuntimeError(
                    f"docker socket does not exist: {socket_path} "
                    "(专用 docker-swegym daemon 未运行；拒绝回落到共享 daemon)"
                )

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int,
    ) -> CommandResult:
        command = list(argv)
        env = dict(os.environ)
        env["DOCKER_HOST"] = self.docker_host
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_sec,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                argv=command,
                exit_code=124,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr),
                duration_sec=time.monotonic() - started,
                timed_out=True,
            )
        except OSError as exc:
            raise DockerRuntimeError(f"Docker command could not be executed: {exc}") from exc
        return CommandResult(
            argv=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_sec=time.monotonic() - started,
        )


@dataclass(slots=True)
class DockerSandbox:
    client: DockerClient
    task: Task
    environment: Environment
    run_id: str
    episode_id: str
    scope: Literal["rollout", "verifier"]
    container_name: str = field(init=False)
    container_id: str | None = field(default=None, init=False)
    acquired_container_id: str | None = field(default=None, init=False)
    started: bool = field(default=False, init=False)
    cleanup_operations: list[dict[str, object]] = field(default_factory=list, init=False)
    cleanup_sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.task.task_id != self.environment.task_id:
            raise ValueError("DockerSandbox task and environment do not match")
        suffix = uuid4().hex[:12]
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]", "-", self.episode_id)[-28:]
        self.container_name = f"swe_agent-{self.scope}-{safe_episode}-{suffix}"

    @property
    def labels(self) -> dict[str, str]:
        return {
            "swe_agent.run_id": self.run_id,
            "swe_agent.episode_id": self.episode_id,
            "swe_agent.task_id": self.task.task_id,
            "swe_agent.scope": self.scope,
        }

    def open(self) -> "DockerSandbox":
        if self.container_id is not None:
            raise ContainerCreateError("sandbox already owns a container handle")
        inspect_image(self.client, self.environment)
        create_result = self.client.run(
            build_create_command(
                name=self.container_name,
                environment=self.environment,
                labels=self.labels,
            ),
            timeout_sec=60,
        )
        primary: BaseException | None = None
        try:
            if create_result.exit_code == 0 and not create_result.timed_out:
                self.container_id = _parse_container_id(create_result.stdout)
                if self.container_id is None:
                    self.container_id = self._inspect_ambiguous_create()
                if self.container_id is None:
                    raise ContainerCreateError("docker create succeeded without a recoverable container ID")
                self.acquired_container_id = self.container_id
            else:
                self.container_id = self._inspect_ambiguous_create()
                if self.container_id is not None:
                    self.acquired_container_id = self.container_id
                raise ContainerCreateError(_failure("failed to create container", create_result))

            started = self.client.run(["docker", "start", self.container_id], timeout_sec=60)
            if started.exit_code != 0 or started.timed_out:
                raise ContainerCreateError(_failure("failed to start container", started))
            self.started = True
            self._verify_base_contract()
            return self
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if primary is not None and self.container_id is not None:
                try:
                    self.close()
                except BaseException as cleanup_exc:
                    primary.add_note(
                        "container cleanup failed during open: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )

    def __enter__(self) -> "DockerSandbox":
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, traceback
        try:
            self.close()
        except BaseException as cleanup_exc:
            if isinstance(exc, BaseException):
                exc.add_note(
                    "container cleanup failed during context exit: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                return
            raise

    def exec(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        if self.container_id is None or not self.started:
            raise ContainerExecError("container has not been started")
        return self.client.run(
            ["docker", "exec", "-i", self.container_id, *command],
            input_text=input_text,
            timeout_sec=timeout_sec or self.environment.exec_timeout_sec,
        )

    def get_diff(self) -> str:
        self._workspace_git(
            "failed to register untracked files",
            "add",
            "-N",
            "--",
            ".",
        )
        result = self._workspace_git(
            "failed to extract git diff",
            "diff",
            self.task.base_commit,
            "--binary",
            "--no-ext-diff",
        )
        return result.stdout

    def _workspace_git(self, failure_prefix: str, *arguments: str) -> CommandResult:
        script = (
            f"printf '%s\\n' '{_WORKSPACE_GIT_STARTED}' >&2; "
            'workspace=$1; shift; exec git -C "$workspace" "$@"'
        )
        result = self.exec(
            [
                "/bin/bash",
                "-lc",
                script,
                "siete-workspace-git",
                self.environment.workdir,
                *arguments,
            ]
        )
        started = _WORKSPACE_GIT_STARTED in result.stderr
        stderr = result.stderr.replace(_WORKSPACE_GIT_STARTED + "\n", "").replace(
            _WORKSPACE_GIT_STARTED, ""
        )
        cleaned = CommandResult(
            argv=result.argv,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=stderr,
            duration_sec=result.duration_sec,
            timed_out=result.timed_out,
        )
        if cleaned.exit_code != 0 or cleaned.timed_out:
            error_type = WorkspaceStateError if started else ContainerExecError
            raise error_type(_failure(failure_prefix, cleaned))
        return cleaned

    def close(self) -> None:
        """按明确 ID 删除；失败保留 ID，后续调用会真正重试。"""

        if self.container_id is None:
            return
        target = self.container_id
        result = self.client.run(["docker", "rm", "-f", target], timeout_sec=120)
        missing = _is_missing_container(result)
        success = result.exit_code == 0 and not result.timed_out
        self.cleanup_sequence += 1
        self.cleanup_operations.append(
            {
                "sequence": self.cleanup_sequence,
                "at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "operation": "remove",
                "result": "success" if success else "not_found" if missing else "failed",
                "error": None if success or missing else _failure("failed to remove container", result),
            }
        )
        if success or missing:
            self.started = False
            self.container_id = None
            return
        raise ContainerCleanupError(_failure("failed to remove container", result))

    def drain_cleanup_operations(self) -> list[dict[str, object]]:
        operations = list(self.cleanup_operations)
        self.cleanup_operations.clear()
        return operations

    def _verify_base_contract(self) -> None:
        head = self.exec(["git", "-C", self.environment.workdir, "rev-parse", "HEAD"])
        if head.exit_code != 0 or head.timed_out or head.stdout.strip() != self.task.base_commit:
            raise ContainerCreateError(
                "container base commit mismatch: "
                f"expected={self.task.base_commit}, actual={head.stdout.strip()}"
            )
        status = self.exec(["git", "-C", self.environment.workdir, "status", "--porcelain"])
        if status.exit_code != 0 or status.timed_out:
            raise ContainerCreateError(_failure("failed to inspect container worktree", status))
        if status.stdout.strip():
            raise ContainerCreateError("container worktree is not clean at startup")

    def _inspect_ambiguous_create(self) -> str | None:
        inspected = self.client.run(
            ["docker", "container", "inspect", self.container_name], timeout_sec=30
        )
        if inspected.exit_code != 0 or inspected.timed_out:
            return None
        try:
            values = json.loads(inspected.stdout)
            container_id = values[0]["Id"]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            return None
        return container_id if isinstance(container_id, str) and re.fullmatch(r"[0-9a-f]{12,64}", container_id) else None


def build_create_command(
    *, name: str, environment: Environment, labels: dict[str, str]
) -> list[str]:
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--pull=never",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--entrypoint",
        "/bin/bash",
        "--cpus",
        str(environment.cpus),
        "--memory",
        environment.memory,
        "--pids-limit",
        str(environment.pids_limit),
    ]
    for key, value in sorted(labels.items()):
        command.extend(["--label", f"{key}={value}"])
    command.extend([environment.image_name, "-lc", "exec sleep infinity"])
    return command


def inspect_image(client: DockerClient, environment: Environment) -> dict[str, object]:
    result = client.run(["docker", "image", "inspect", environment.image_name], timeout_sec=30)
    if result.exit_code != 0 or result.timed_out:
        raise ContainerCreateError(_failure("local fixed image does not exist", result))
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContainerCreateError("docker image inspect returned invalid JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ContainerCreateError("docker image inspect returned an invalid structure")
    payload = values[0]
    if payload.get("Os") != "linux" or payload.get("Architecture") != "amd64":
        raise ContainerCreateError("local image platform must be linux/amd64")
    return {
        "image": environment.image_name,
        "image_id": payload.get("Id"),
        "repo_tags": payload.get("RepoTags", []),
        "size_bytes": payload.get("Size"),
        "os": payload.get("Os"),
        "architecture": payload.get("Architecture"),
    }


def sweep_run_containers(client: DockerClient, run_id: str) -> list[str]:
    """按 run label 兜底清扫孤儿容器；返回实际删除的容器 ID 列表。"""

    listed = client.run(
        ["docker", "ps", "-aq", "--filter", f"label=swe_agent.run_id={run_id}"],
        timeout_sec=30,
    )
    if listed.exit_code != 0 or listed.timed_out:
        raise ContainerCleanupError(_failure("failed to list run containers", listed))
    removed: list[str] = []
    for line in listed.stdout.splitlines():
        container_id = line.strip()
        if not container_id:
            continue
        result = client.run(["docker", "rm", "-f", container_id], timeout_sec=30)
        if result.exit_code != 0 or result.timed_out:
            if _is_missing_container(result):
                continue
            raise ContainerCleanupError(_failure("failed to remove container", result))
        removed.append(container_id)
    return removed


def _parse_container_id(stdout: str) -> str | None:
    candidate = stdout.strip().splitlines()[0] if stdout.strip() else ""
    return candidate if re.fullmatch(r"[0-9a-f]{12,64}", candidate) else None


def _is_missing_container(result: CommandResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return result.exit_code != 0 and (
        "no such container" in combined or "no such object" in combined
    )


def _failure(prefix: str, result: CommandResult) -> str:
    output = "\n".join(value.rstrip() for value in (result.stdout, result.stderr) if value).strip()
    suffix = f"\n{output}" if output else ""
    return f"{prefix} (exit={result.exit_code}, timeout={result.timed_out}){suffix}"


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
