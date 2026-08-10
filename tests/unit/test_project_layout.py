from __future__ import annotations

import ast
from importlib.metadata import entry_points
from pathlib import Path
import re
import subprocess
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_launch_scripts_are_syntax_valid_and_delegate_to_package_entry_points() -> None:
    launchers = {
        "eval.sh": "python -m siete_rl.eval",
        "grpo.sh": ".venv/bin/siete-rl grpo",
        "qualify.sh": "python -m siete_rl.qualify",
    }
    for name, delegation in launchers.items():
        script = PROJECT_ROOT / "scripts" / name
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert delegation in script.read_text(encoding="utf-8")


def test_pull_script_is_pinned_to_dedicated_daemon() -> None:
    script = (PROJECT_ROOT / "scripts/prepare.sh").read_text(encoding="utf-8")
    # 拉取只允许指向专用 docker-swegym daemon，绝不触碰共享 socket
    assert 'DOCKER_HOST="unix:///run/docker-swegym/docker.sock"' in script
    assert "/var/run/docker.sock" not in script
    assert "snap.docker" not in script


def test_console_script_targets_cli_main() -> None:
    matches = [point for point in entry_points(group="console_scripts") if point.name == "siete-rl"]
    assert len(matches) == 1
    assert matches[0].value == "siete_rl.cli:main"


def test_runtime_code_has_no_external_openhands_dependency() -> None:
    forbidden = {"openhands", "openhands_aci", "litellm", "browsergym"}
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(name.split(".")[0] in forbidden for name in names), path

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
        for requirement in project["project"]["dependencies"]
    }
    assert forbidden.isdisjoint(dependency_names)
