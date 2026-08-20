from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from siete_rl.config import load_config
from siete_rl import supervisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1.yaml"


def _process_is_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


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


def test_terminate_process_group_reaps_worker_after_leader_exits(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time; "
        "from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _process_is_alive(child_pid)
        process_groups = supervisor._descendant_process_groups(leader.pid)
        assert process_groups == {leader.pid, child_pid}

        assert supervisor._terminate_process_group(
            leader, signal.SIGTERM, 1.0, process_groups=process_groups
        )
        assert not _process_is_alive(child_pid)
    finally:
        for pgid in (leader.pid, child_pid):
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_supervisor_reports_residual_worker_process_group(tmp_path: Path) -> None:
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
                "time": {"started_at": "2026-07-26T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    supervisor._mark_stale_worker_run(
        output_dir,
        returncode=1,
        interrupted_signum=None,
        cleanup_errors=[],
        worker_group_residual=True,
    )

    report = json.loads(run_path.read_text(encoding="utf-8"))
    assert report["cleanup"] == {
        "status": "failed",
        "clean_release": False,
        "residual_count": 1,
    }
