from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from swe_agent.config import load_config
from swe_agent.launcher import (
    LauncherError,
    VLLMEndpoints,
    VLLMServer,
    allocate_vllm_endpoints,
    build_server_command,
    resolve_gpu_topology,
    split_visible_gpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


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


def test_split_visible_gpus_assigns_server_and_trainer(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server_gpu = split_visible_gpus(load_project_config())
    assert server_gpu == "0"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_split_visible_gpus_requires_two_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(LauncherError, match="at least"):
        split_visible_gpus(load_project_config())


def test_resolve_gpu_topology_does_not_mutate_environment(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert resolve_gpu_topology(load_project_config()) == ("2", "3")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_run_endpoints_are_distinct_and_override_fixed_config_port(monkeypatch) -> None:
    config = load_project_config()
    ports = iter((18421, 18422))
    monkeypatch.setattr("swe_agent.launcher._reserve_ephemeral_port", lambda host, excluded=None: next(ports))
    endpoints = allocate_vllm_endpoints(config)
    assert endpoints.host == "127.0.0.1"
    assert endpoints.server_port != endpoints.group_port
    command = build_server_command(config, endpoints)
    assert command[command.index("--port") + 1] == str(endpoints.server_port)


def test_build_server_command_matches_config() -> None:
    config = load_project_config()
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

    monkeypatch.setattr("swe_agent.launcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "swe_agent.launcher.os.killpg",
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
    assert spawned["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert spawned["start_new_session"] is True
    assert (tmp_path / "vllm.log").exists()
    assert signals == [(4321, signal.SIGTERM)]
    assert handle["scope"] == "vllm_server"
    assert handle["identifier"] == "4321"
    assert handle["final_state"] == "terminated"
    assert handle["residual"] is False
    assert [op["operation"] for op in handle["operations"]] == ["spawn", "health", "terminate"]


def test_server_close_escalates_to_kill(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("swe_agent.launcher.TERM_GRACE_SEC", 0.05)
    server, _, signals = make_server(monkeypatch, tmp_path, polls=[None])

    server.start()
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
