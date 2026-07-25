from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_only_planned_launcher_exists() -> None:
    scripts = sorted(path.name for path in (PROJECT_ROOT / "scripts").iterdir())
    assert scripts == ["dry_run.sh", "grpo.sh", "prepare.sh", "qualify.sh"]
    launcher = (PROJECT_ROOT / "scripts/grpo.sh").read_text(encoding="utf-8")
    assert 'exec .venv/bin/swe_agent grpo --config "$config_path"' in launcher
    # vLLM server 生命周期、GPU 拆分与容器清扫均已内嵌 swe_agent.launcher
    assert "vllm-serve" not in launcher
    assert "setsid" not in launcher
    assert "docker rm" not in launcher
    assert "SWE_AGENT_RUN_ID" not in launcher
    assert "accelerate launch" not in launcher
    assert "--use_fsdp" not in launcher


def test_pull_script_is_pinned_to_dedicated_daemon() -> None:
    script = (PROJECT_ROOT / "scripts/prepare.sh").read_text(encoding="utf-8")
    # 拉取只允许指向专用 docker-swegym daemon，绝不触碰共享 socket
    assert 'DOCKER_HOST="unix:///run/docker-swegym/docker.sock"' in script
    assert "/var/run/docker.sock" not in script
    assert "snap.docker" not in script


def test_console_script_targets_cli_main() -> None:
    matches = [point for point in entry_points(group="console_scripts") if point.name == "swe_agent"]
    assert len(matches) == 1
    assert matches[0].value == "swe_agent.cli:main"


def test_package_has_no_legacy_control_plane() -> None:
    package_root = PROJECT_ROOT / "src/swe_agent"
    assert (package_root / "__init__.py").is_file()
    assert not (package_root / "core").exists()
    assert not (package_root / "runtime").exists()
    assert not (package_root / "training").exists()
    assert not (package_root / "workflow.py").exists()
