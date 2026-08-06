from __future__ import annotations

import json
import signal
from pathlib import Path

from siete_rl.config import load_config
from siete_rl.launcher import VLLMEndpoints
from siete_rl import supervisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def test_supervisor_owns_server_and_starts_isolated_worker(monkeypatch, tmp_path: Path) -> None:
    config, project_root, resolved_config_path = load_config(CONFIG_PATH)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        supervisor, "load_config", lambda path: (config, project_root, resolved_config_path)
    )
    monkeypatch.setattr(
        supervisor,
        "preflight",
        lambda loaded, root: {"missing_domain_modules": [], "status": "preflight_passed"},
    )
    monkeypatch.setattr(supervisor, "resolve_gpu_topology", lambda loaded: ("0", "1"))
    endpoints = VLLMEndpoints("127.0.0.1", 18421, 18422)
    monkeypatch.setattr(supervisor, "allocate_vllm_endpoints", lambda loaded: endpoints)
    monkeypatch.setattr(supervisor, "generate_run_id", lambda: "run-supervised")

    workspace = tmp_path / "run-supervised"
    workspace.mkdir()
    (workspace / ".swe-agent-supervisor-workspace").write_text("run-supervised", encoding="utf-8")
    monkeypatch.setattr(supervisor, "_prepare_workspace", lambda root, run_id: workspace)

    class FakeServer:
        pid = 441

        def __init__(self, command, **kwargs) -> None:
            calls["server_command"] = command
            calls["server_kwargs"] = kwargs

        def start(self, *, cancelled=None) -> None:
            del cancelled
            calls["server_started"] = True

        def close(self):
            calls["server_closed"] = True
            return {"scope": "vllm_server", "final_state": "terminated", "residual": False}

    class FakeWorker:
        pid = 442

        def poll(self):
            return 0

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr(supervisor, "VLLMServer", FakeServer)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda command, **kwargs: calls.update(worker_command=command, worker_kwargs=kwargs) or FakeWorker(),
    )
    monkeypatch.setattr(supervisor, "_sweep_run_containers", lambda run_id, state: None)
    expected = {"run_id": "run-supervised", "status": "completed"}
    monkeypatch.setattr(supervisor, "_load_worker_report", lambda output_dir: expected)

    report = supervisor.run(CONFIG_PATH)

    assert report == expected
    assert calls["server_started"] is True
    assert calls["server_closed"] is True
    command = calls["worker_command"]
    assert command[command.index("--server-port") + 1] == "18421"
    assert command[command.index("--group-port") + 1] == "18422"
    assert command[command.index("--trainer-gpu") + 1] == "1"
    assert calls["worker_kwargs"]["start_new_session"] is True
    worker_env = calls["worker_kwargs"]["env"]
    assert "127.0.0.1" in worker_env["NO_PROXY"]
    assert "localhost" in worker_env["no_proxy"]


def test_supervisor_marks_stale_worker_report_interrupted(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    run_path = output_dir / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "status": "running",
                "failure": None,
                "time": {
                    "started_at": "2026-07-26T00:00:00Z",
                    "finished_at": None,
                    "duration_seconds": None,
                },
            }
        ),
        encoding="utf-8",
    )

    supervisor._mark_interrupted_run(output_dir, signal.SIGTERM)

    report = json.loads(run_path.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["failure"]["type"] == "SupervisorTermination"
    assert "SIGTERM" in report["failure"]["message"]
    assert report["time"]["finished_at"] is not None
