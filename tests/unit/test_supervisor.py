from __future__ import annotations

import json
import signal
from pathlib import Path

from siete_rl.config import load_config
from siete_rl import supervisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1.yaml"


def test_supervisor_starts_four_gpu_colocated_worker(monkeypatch, tmp_path: Path) -> None:
    config, project_root, resolved_config_path = load_config(CONFIG_PATH)
    config = config.model_copy(
        update={
            "vllm": config.vllm.model_copy(
                update={
                    "mode": "colocate",
                    "enable_sleep_mode": True,
                    "tensor_parallel_size": 4,
                    "server_base_url": None,
                }
            ),
            "runtime": config.runtime.model_copy(update={"process_count": 4}),
        }
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        supervisor, "load_config", lambda path: (config, project_root, resolved_config_path)
    )
    monkeypatch.setattr(
        supervisor,
        "preflight",
        lambda loaded, root: {"missing_domain_modules": [], "status": "preflight_passed"},
    )
    monkeypatch.setattr(supervisor, "resolve_gpu_topology", lambda loaded: ("0,1,2,3", "0,1,2,3"))
    monkeypatch.setattr(
        supervisor,
        "allocate_vllm_endpoints",
        lambda loaded: type("Endpoints", (), {"ddp_port": 18423})(),
    )
    monkeypatch.setattr(supervisor, "generate_run_id", lambda: "run-supervised")

    expected_run_id = f"stage{config.dataset.stage}-run-supervised"
    workspace = tmp_path / expected_run_id
    workspace.mkdir()
    (workspace / ".swe-agent-supervisor-workspace").write_text(expected_run_id, encoding="utf-8")
    monkeypatch.setattr(supervisor, "_prepare_workspace", lambda root, run_id: workspace)

    class FakeServer:
        def __init__(self, command, **kwargs) -> None:
            del command, kwargs
            raise AssertionError("colocate mode must not start a separate vLLM server")

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
    expected = {"run_id": expected_run_id, "status": "completed"}
    monkeypatch.setattr(supervisor, "_load_worker_report", lambda output_dir: expected)

    report = supervisor.run(CONFIG_PATH)

    assert report == expected
    command = calls["worker_command"]
    assert command[:4] == [
        command[0], "-m", "accelerate.commands.launch", "--num_processes"
    ]
    assert command[command.index("--num_processes") + 1] == "4"
    assert "--multi_gpu" in command
    assert command[command.index("--main_process_port") + 1] == "18423"
    assert command[command.index("--tee") + 1] == "3"
    assert command[command.index("--log_dir") + 1] == (workspace / "worker_logs").as_posix()
    assert "--server-port" not in command
    assert "--group-port" not in command
    assert calls["worker_kwargs"]["start_new_session"] is True
    worker_env = calls["worker_kwargs"]["env"]
    assert worker_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_supervisor_marks_stale_worker_report_interrupted(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    run_path = output_dir / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "status": "running",
                "failure": None,
                "cleanup": {
                    "status": "pending",
                    "clean_release": None,
                    "residual_count": 0,
                },
                "time": {
                    "started_at": "2026-07-26T00:00:00Z",
                    "finished_at": None,
                    "duration_seconds": None,
                },
            }
        ),
        encoding="utf-8",
    )

    supervisor._mark_stale_worker_run(
        output_dir,
        returncode=143,
        interrupted_signum=signal.SIGTERM,
        cleanup_errors=[],
    )

    report = json.loads(run_path.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["failure"]["type"] == "SupervisorTermination"
    assert "SIGTERM" in report["failure"]["message"]
    assert report["time"]["finished_at"] is not None


def test_supervisor_marks_stale_worker_report_failed(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "running",
                "failure": None,
                "cleanup": {
                    "status": "pending",
                    "clean_release": None,
                    "residual_count": 0,
                },
                "time": {
                    "started_at": "2026-07-26T00:00:00Z",
                    "finished_at": None,
                    "duration_seconds": None,
                },
            }
        ),
        encoding="utf-8",
    )

    supervisor._mark_stale_worker_run(
        output_dir,
        returncode=1,
        interrupted_signum=None,
        cleanup_errors=[],
    )

    report = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "WorkerProcessError"
    assert report["cleanup"]["status"] == "completed"
