from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_only_planned_launcher_exists() -> None:
    scripts = sorted(path.name for path in (PROJECT_ROOT / "scripts").iterdir())
    assert scripts == ["grpo.sh"]
    launcher = (PROJECT_ROOT / "scripts/grpo.sh").read_text(encoding="utf-8")
    assert ".venv/bin/trl vllm-serve" in launcher
    assert "setsid env CUDA_VISIBLE_DEVICES=\"$server_gpu\"" in launcher
    assert 'CUDA_VISIBLE_DEVICES="$server_gpu"' in launcher
    assert 'CUDA_VISIBLE_DEVICES="$trainer_gpu"' in launcher
    assert 'curl --fail --silent --show-error "$server_url/health"' in launcher
    assert 'kill -TERM -- "-$server_pid"' in launcher
    assert 'label=swe_agent.run_id=$SWE_AGENT_RUN_ID' in launcher
    assert '.venv/bin/swe_agent grpo --config "$config_path"' in launcher
    assert "accelerate launch" not in launcher
    assert "--use_fsdp" not in launcher


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
