"""不初始化 CUDA 的 GRPO run supervisor。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_agent.config import load_config
from swe_agent.launcher import (
    TERM_GRACE_SEC,
    LauncherError,
    VLLMServer,
    allow_loopback_without_proxy,
    allocate_vllm_endpoints,
    build_server_command,
    resolve_gpu_topology,
)
from swe_agent.recording import generate_run_id
from swe_agent.train import TrainingNotReadyError, preflight


WORKER_TERM_GRACE_SEC = 30.0


class _SignalLatch:
    """把 supervisor 信号转换为轮询状态，避免在子进程 wait 中抛异常。"""

    def __init__(self) -> None:
        self.signum: int | None = None
        self._previous: dict[signal.Signals, object] = {}

    def __enter__(self) -> "_SignalLatch":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def _handle(self, signum: int, frame: object) -> None:
        del frame
        if self.signum is None:
            self.signum = signum


def run(config_path: str | Path) -> dict[str, Any]:
    """启动 vLLM 与 worker，并在 worker 卡死时仍能完成有界收束。"""

    config, project_root, resolved_config_path = load_config(config_path)
    report = preflight(config, project_root)
    report["config_path"] = resolved_config_path.as_posix()
    if report["missing_domain_modules"]:
        missing = ", ".join(report["missing_domain_modules"])
        raise TrainingNotReadyError(
            "preflight 已通过模型与运行配置检查，但领域实现尚未完成，"
            f"拒绝启动真实训练；缺少: {missing}"
        )
    topology = resolve_gpu_topology(config)
    if topology is None:
        raise RuntimeError("supervised GRPO requires vLLM server mode")
    server_gpu, trainer_gpu = topology
    run_id = config.output.run_id or generate_run_id()
    endpoints = allocate_vllm_endpoints(config)
    output_dir = _prepare_workspace(config.output.output_root, run_id)
    state_path = output_dir / "supervisor.json"
    state: dict[str, Any] = {
        "run_id": run_id,
        "state": "starting",
        "started_at": _utc_now(),
        "endpoints": {
            "server_url": endpoints.base_url,
            "server_port": endpoints.server_port,
            "group_port": endpoints.group_port,
        },
        "worker": None,
        "server": None,
        "cleanup": {"containers_removed": [], "errors": []},
    }
    _write_json(state_path, state)

    server = VLLMServer(
        build_server_command(config, endpoints),
        server_gpu=server_gpu,
        base_url=endpoints.base_url,
        log_path=output_dir / "vllm.log",
    )
    worker: subprocess.Popen[bytes] | None = None
    interrupted_signum: int | None = None
    try:
        with _SignalLatch() as latch:
            try:
                server.start(cancelled=lambda: latch.signum is not None)
            except LauncherError:
                if latch.signum is None:
                    raise
                interrupted_signum = latch.signum
            if interrupted_signum is None:
                state["server"] = {"pid": server.pid, "state": "ready"}
                state["state"] = "running"
                _write_json(state_path, state)
                worker = _start_worker(
                    resolved_config_path,
                    run_id=run_id,
                    endpoints=endpoints,
                    trainer_gpu=trainer_gpu,
                )
                state["worker"] = {"pid": worker.pid, "state": "running"}
                _write_json(state_path, state)
                while worker.poll() is None:
                    if latch.signum is not None:
                        interrupted_signum = latch.signum
                        _terminate_process_group(worker, latch.signum, WORKER_TERM_GRACE_SEC)
                        break
                    time.sleep(0.1)
                if worker.poll() is None:
                    worker.wait(timeout=1)
    finally:
        if worker is not None and worker.poll() is None:
            _terminate_process_group(worker, signal.SIGTERM, WORKER_TERM_GRACE_SEC)
        server_handle = server.close()
        state["server"] = server_handle
        _sweep_run_containers(run_id, state)
        state["finished_at"] = _utc_now()
        state["interrupted_signum"] = interrupted_signum
        state["worker"] = _worker_state(worker)
        state["state"] = "interrupted" if interrupted_signum is not None else "finished"
        _write_json(state_path, state)

    if interrupted_signum is not None and worker is not None and worker.poll() is not None:
        _mark_interrupted_run(output_dir, interrupted_signum)
    worker_report = _load_worker_report(output_dir)
    if worker_report is None:
        return {
            "run_id": run_id,
            "status": "interrupted" if interrupted_signum is not None else "failed",
            "failure": {
                "category": "interrupted" if interrupted_signum is not None else "supervisor",
                "message": "worker exited before writing run.json",
            },
            "cleanup": state["cleanup"],
            "interrupted_signum": interrupted_signum,
        }
    if interrupted_signum is not None:
        worker_report["status"] = "interrupted"
        worker_report["interrupted_signum"] = interrupted_signum
    return worker_report


def _prepare_workspace(output_root: str, run_id: str) -> Path:
    output_dir = (Path(output_root) / run_id).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    (output_dir / ".swe-agent-supervisor-workspace").write_text(run_id, encoding="utf-8")
    return output_dir


def _start_worker(
    config_path: Path,
    *,
    run_id: str,
    endpoints: Any,
    trainer_gpu: str,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "swe_agent.worker",
        "--config",
        config_path.as_posix(),
        "--run-id",
        run_id,
        "--server-host",
        endpoints.host,
        "--server-port",
        str(endpoints.server_port),
        "--group-port",
        str(endpoints.group_port),
        "--trainer-gpu",
        trainer_gpu,
    ]
    env = dict(os.environ)
    env["SWE_AGENT_RUN_ID"] = run_id
    # TRL 的 VLLMClient 在 trainer 构造阶段以 requests.get 探测本机 server；
    # 该实现不设 timeout，必须确保 loopback 不会被 HTTP(S)_PROXY 截获。
    allow_loopback_without_proxy(env)
    return subprocess.Popen(command, env=env, start_new_session=True)


def _terminate_process_group(
    process: subprocess.Popen[bytes], signum: int, grace_sec: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=TERM_GRACE_SEC)
    except subprocess.TimeoutExpired:
        pass


def _sweep_run_containers(run_id: str, state: dict[str, Any]) -> None:
    try:
        from swe_agent.docker import SubprocessDockerClient, sweep_run_containers

        state["cleanup"]["containers_removed"] = sweep_run_containers(
            SubprocessDockerClient(), run_id
        )
    except BaseException as exc:  # noqa: BLE001 - 不能跳过 server 收束记录
        state["cleanup"]["errors"].append(f"{type(exc).__name__}: {exc}")


def _load_worker_report(output_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mark_interrupted_run(output_dir: Path, signum: int) -> None:
    """在 worker 已退出时接管其未完成的 run.json 终态。

    正常中断由 worker 自己写入终态。只有 supervisor 已确认 worker 退出、却仍
    看到旧的 ``running`` 状态时才接管，避免 SIGKILL 后把磁盘记录永久留在运行中。
    ``supervisor.json`` 仍是 server 与容器收束的详细记录。
    """

    path = output_dir / "run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or payload.get("status") == "interrupted":
        return
    signal_name = signal.Signals(signum).name
    payload["status"] = "interrupted"
    payload["failure"] = {
        "category": "interrupted",
        "type": "SupervisorTermination",
        "message": f"supervisor received {signal_name}; worker exited before normal cleanup",
        "stage": "supervisor",
        "log": "train.log",
    }
    time_info = payload.get("time")
    if isinstance(time_info, dict):
        finished_at = _utc_now()
        time_info["finished_at"] = finished_at
        try:
            started_at = datetime.fromisoformat(str(time_info["started_at"]).replace("Z", "+00:00"))
            finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            time_info["duration_seconds"] = max(0.0, (finished - started_at).total_seconds())
        except (KeyError, TypeError, ValueError):
            pass
    _write_json(path, payload)


def _worker_state(process: subprocess.Popen[bytes] | None) -> dict[str, Any] | None:
    if process is None:
        return None
    return {"pid": process.pid, "returncode": process.poll()}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
