from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_only_planned_launcher_exists() -> None:
    scripts = sorted(path.name for path in (PROJECT_ROOT / "scripts").iterdir())
    assert scripts == ["grpo.sh"]
    launcher = (PROJECT_ROOT / "scripts/grpo.sh").read_text(encoding="utf-8")
    assert 'exec .venv/bin/swe-agent grpo --config "$1"' in launcher


def test_console_script_targets_cli_main() -> None:
    matches = [point for point in entry_points(group="console_scripts") if point.name == "swe-agent"]
    assert len(matches) == 1
    assert matches[0].value == "swe_agent.cli:main"


def test_package_has_no_legacy_control_plane() -> None:
    package_root = PROJECT_ROOT / "src/swe_agent"
    assert (package_root / "__init__.py").is_file()
    assert not (package_root / "core").exists()
    assert not (package_root / "runtime").exists()
    assert not (package_root / "training").exists()
    assert not (package_root / "workflow.py").exists()
