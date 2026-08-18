from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from siete_rl.config import load_config
from siete_rl.launcher import (
    LauncherError,
    RunEndpoints,
    VLLMServer,
    allocate_vllm_endpoints,
    build_server_command,
    resolve_gpu_topology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1.yaml"


def load_project_config():
    config, _, _ = load_config(CONFIG_PATH)
    return config


class FakePopen:
    def __init__(self, polls: list[int | None]) -> None:
        self.pid = 4321
        self._polls = list(polls)

    def poll(self) -> int | None:
        if not self._polls:
            return 0
        value = self._polls[0]
        if len(self._polls) > 1:
            self._polls.pop(0)
        return value

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("0,1,2,3", ("0,1,2,3", "0,1,2,3")),
        ("4,6,1,7", ("4,6,1,7", "4,6,1,7")),
    ],
)
def test_resolve_gpu_topology_preserves_visible_order(visible, expected) -> None:
    assert resolve_gpu_topology(load_project_config(), visible) == expected


def test_resolve_two_gpu_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_COUNT", "2")

    assert resolve_gpu_topology(load_project_config(), "4,6") == ("4,6", "4,6")


@pytest.mark.parametrize("visible", ["0,1,2", "0,1,2,3,4"])
def test_resolve_gpu_topology_requires_configured_device_count(visible) -> None:
    with pytest.raises(LauncherError, match="CUDA_VISIBLE_DEVICES"):
        resolve_gpu_topology(load_project_config(), visible)


def test_run_endpoints_are_distinct_and_override_fixed_config_port(monkeypatch) -> None:
    config = load_project_config()
    ports = iter((18421, 18422, 18423))
    monkeypatch.setattr("siete_rl.launcher._reserve_ephemeral_port", lambda host, excluded=None: next(ports))
    endpoints = allocate_vllm_endpoints(config)
    assert endpoints.host == "127.0.0.1"
    assert len({endpoints.server_port, endpoints.group_port, endpoints.ddp_port}) == 3


def test_build_server_command_matches_config() -> None:
    config = load_project_config()
    config = config.model_copy(
        update={
            "vllm": config.vllm.model_copy(
                update={
                    "mode": "server",
                    "server_base_url": "http://127.0.0.1:8000",
                }
            )
        }
    )
    command = build_server_command(config)
    assert command[1] == "vllm-serve"
    assert config.model.model_path in command
    assert "127.0.0.1" in command and "8000" in command
    assert "--trust-remote-code" in command
    assert command[command.index("--dtype") + 1] == "bfloat16"


def make_server(monkeypatch, tmp_path: Path, polls: list[int | None], probe_result: bool = True):
    spawned: dict[str, object] = {}
    signals: list[tuple[int, signal.Signals]] = []

    def fake_popen(command, **kwargs):
        spawned["command"] = command
        spawned.update(kwargs)
        return FakePopen(polls)

    monkeypatch.setattr("siete_rl.launcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "siete_rl.launcher.os.killpg",
        lambda pid, signum: signals.append((pid, signal.Signals(signum))),
    )
    server = VLLMServer(
        ["trl", "vllm-serve"],
        server_gpu="0",
        base_url="http://127.0.0.1:8000",
        log_path=tmp_path / "vllm.log",
        health_timeout_sec=1,
        probe=lambda url: probe_result,
    )
    return server, spawned, signals


def test_server_start_and_clean_close(monkeypatch, tmp_path) -> None:
    server, spawned, signals = make_server(monkeypatch, tmp_path, polls=[None, None, 0, 0])

    server.start()
    handle = server.close()

    assert spawned["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "127.0.0.1" in spawned["env"]["NO_PROXY"]
    assert "::1" in spawned["env"]["no_proxy"]
    assert spawned["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert spawned["start_new_session"] is True
    assert (tmp_path / "vllm.log").exists()
    assert signals == [(4321, signal.SIGTERM)]
    assert handle["scope"] == "vllm_server"
    assert handle["identifier"] == "4321"
    assert handle["final_state"] == "terminated"
    assert handle["residual"] is False
    assert [op["operation"] for op in handle["operations"]] == ["spawn", "health", "terminate"]


def test_server_close_escalates_to_kill_after_grace_period(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("siete_rl.launcher.TERM_GRACE_SEC", 0.05)
    server, _, signals = make_server(monkeypatch, tmp_path, polls=[None])

    server.start()
    ticks = iter([0.0, 0.0, 0.1])
    monkeypatch.setattr("siete_rl.launcher.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("siete_rl.launcher.time.sleep", lambda _seconds: None)
    handle = server.close()

    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert handle["final_state"] == "residual"
    assert handle["residual"] is True


def test_server_early_exit_is_launcher_error(monkeypatch, tmp_path) -> None:
    server, _, _ = make_server(monkeypatch, tmp_path, polls=[1], probe_result=False)

    with pytest.raises(LauncherError, match="exited before becoming ready"):
        server.start()


def test_server_close_is_idempotent_without_start(tmp_path) -> None:
    server = VLLMServer(
        ["trl", "vllm-serve"],
        server_gpu="0",
        base_url="http://127.0.0.1:8000",
        log_path=tmp_path / "vllm.log",
    )
    handle = server.close()
    assert handle["final_state"] == "not_initialized"
    assert handle["residual"] is False
