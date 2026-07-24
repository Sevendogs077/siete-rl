"""vLLM server 生命周期与 GPU 拆分（替代 scripts/grpo.sh 的编排职责）。

设计约束：
- ``split_visible_gpus`` 必须在任何 torch CUDA 初始化之前调用；
- server 子进程 ``start_new_session=True`` 自成进程组，退出时按
  TERM → 宽限 → KILL 升级（对齐原 bash cleanup 语义）；
- server stdout/stderr tee 到 run 目录下的 ``vllm.log``。
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from swe_agent.config import ProjectConfig


HEALTH_TIMEOUT_SEC = 1200
TERM_GRACE_SEC = 10.0


class LauncherError(RuntimeError):
    """vLLM server 启动或生命周期管理失败。"""


def split_visible_gpus(config: ProjectConfig) -> str | None:
    """server 模式把多卡 CUDA_VISIBLE_DEVICES 拆成 server/trainer 两张卡。

    当前进程的 CUDA_VISIBLE_DEVICES 改写为 trainer 卡（供后续单卡校验），
    返回 server 卡索引；非 server 模式返回 None，维持调用方环境。
    """

    if config.vllm.mode != "server":
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if len(devices) < 2:
        raise LauncherError(
            "vllm server mode requires CUDA_VISIBLE_DEVICES to list at least "
            f"two GPUs (server, trainer); got {visible!r}"
        )
    server_gpu, trainer_gpu = devices[0], devices[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu
    return server_gpu


def build_server_command(config: ProjectConfig) -> list[str]:
    """由配置推导 `trl vllm-serve` 参数（对齐原 grpo.sh 的参数表）。"""

    if config.vllm.mode != "server" or config.vllm.server_base_url is None:
        raise LauncherError("build_server_command requires vllm server mode configuration")
    url = urlparse(config.vllm.server_base_url)
    if url.scheme != "http" or url.hostname is None or url.port is None:
        raise LauncherError("vllm.server_base_url must be an http URL with an explicit port")
    command = [
        _trl_executable(),
        "vllm-serve",
        "--model",
        config.model.model_path,
        "--tensor-parallel-size",
        str(config.vllm.tensor_parallel_size or 1),
        "--gpu-memory-utilization",
        str(config.vllm.gpu_memory_utilization),
        "--dtype",
        config.model.dtype,
        "--max-model-len",
        str(config.vllm.max_model_length),
        "--host",
        url.hostname,
        "--port",
        str(url.port),
    ]
    if config.model.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def health_probe(url: str) -> bool:
    """绕过代理的 /health 探测；任何网络/HTTP 错误都视为未就绪。"""

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


class VLLMServer:
    """一个 vLLM server 子进程的生命周期；close 幂等且返回 runtime handle 记录。"""

    def __init__(
        self,
        command: list[str],
        *,
        server_gpu: str,
        base_url: str,
        log_path: Path,
        health_timeout_sec: int = HEALTH_TIMEOUT_SEC,
        probe: Callable[[str], bool] = health_probe,
    ) -> None:
        self._command = command
        self._server_gpu = server_gpu
        self._base_url = base_url.rstrip("/")
        self._log_path = log_path
        self._health_timeout_sec = health_timeout_sec
        self._probe = probe
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None
        self._operations: list[dict[str, object]] = []

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None:
            raise LauncherError("vLLM server already started")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("ab")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self._server_gpu
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for key in ("NO_PROXY", "no_proxy"):
            existing = env.get(key)
            env[key] = f"{existing},127.0.0.1,localhost" if existing else "127.0.0.1,localhost"
        try:
            self._process = subprocess.Popen(
                self._command,
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._log_handle.close()
            self._log_handle = None
            raise LauncherError(f"vLLM server could not be spawned: {exc}") from exc
        self._record("spawn", "success")
        deadline = time.monotonic() + self._health_timeout_sec
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise LauncherError(
                    f"vLLM server exited before becoming ready; see {self._log_path}"
                )
            if self._probe(f"{self._base_url}/health"):
                self._record("health", "success")
                return
            time.sleep(1)
        raise LauncherError(
            f"vLLM server did not become ready within {self._health_timeout_sec} seconds"
        )

    def close(self) -> dict[str, Any]:
        if self._process is None:
            return self._handle("not_initialized")
        process = self._process
        pid = process.pid
        self._process = None
        if process.poll() is None:
            self._terminate(process, signal.SIGTERM, "terminate")
            deadline = time.monotonic() + TERM_GRACE_SEC
            while time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.1)
            if process.poll() is None:
                self._terminate(process, signal.SIGKILL, "kill")
        try:
            process.wait(timeout=TERM_GRACE_SEC)
        except subprocess.TimeoutExpired:
            pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        residual = process.poll() is None
        return self._handle(
            "residual" if residual else "terminated", residual=residual, pid=pid
        )

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _terminate(self, process: subprocess.Popen[bytes], signum: int, operation: str) -> None:
        try:
            os.killpg(process.pid, signum)
        except OSError as exc:
            self._record(operation, "failed", str(exc))
        else:
            self._record(operation, "success")

    def _record(self, operation: str, result: str, error: str | None = None) -> None:
        self._operations.append(
            {
                "sequence": len(self._operations) + 1,
                "at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "operation": operation,
                "result": result,
                "error": error,
            }
        )

    def _handle(
        self, final_state: str, *, residual: bool = False, pid: int | None = None
    ) -> dict[str, Any]:
        owner = pid if pid is not None else self.pid
        return {
            "scope": "vllm_server",
            "identifier": str(owner) if owner is not None else None,
            "operations": list(self._operations),
            "final_state": final_state,
            "residual": residual,
        }


def _trl_executable() -> str:
    candidate = Path(sys.executable).with_name("trl")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("trl")
    if found is None:
        raise LauncherError("trl executable not found next to the interpreter or on PATH")
    return found
