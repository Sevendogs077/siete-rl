from __future__ import annotations

import json
from pathlib import Path

import pytest

from siete_rl.asset_generation import generate_task_assets, image_tag_for
from siete_rl.swegym import SWEGymContractError


IMAGE_ID = "sha256:" + "1" * 64


def _row(instance_id: str = "owner__repo-1") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "version": "4.2",
        "problem_statement": "problem",
        "patch": "gold-diff",
        "test_patch": "test-diff",
        "FAIL_TO_PASS": ["t1"],
        "PASS_TO_PASS": ["t2"],
        "eval_script": "make init\npytest -q\n",
    }


def test_generate_task_assets_is_idempotent(tmp_path: Path) -> None:
    first = generate_task_assets(_row(), tmp_path / "assets", image_id=IMAGE_ID)
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}
    second = generate_task_assets(_row(), tmp_path / "assets", image_id=IMAGE_ID)
    assert second == first
    assert {p.name: p.stat().st_mtime_ns for p in second} == mtimes
    manifest = json.loads((tmp_path / "assets/owner__repo-1/manifest.json").read_text())
    assert manifest == {
        "schema_version": "1",
        "task_id": "owner__repo-1",
        "repo_name": "owner/repo",
        "base_commit": "a" * 40,
        "image_name": "docker.io/xingyaoww/sweb.eval.x86_64.owner_s_repo-1:latest",
        "expected_image_id": IMAGE_ID,
    }


def test_generate_task_assets_rewrites_on_upstream_change(tmp_path: Path) -> None:
    row = _row()
    generate_task_assets(row, tmp_path / "assets", image_id=IMAGE_ID)
    root = tmp_path / "assets" / "owner__repo-1"
    assert (root / "gold.patch").read_text() == "gold-diff"

    row["patch"] = "gold-diff-v2"
    generate_task_assets(row, tmp_path / "assets", image_id=IMAGE_ID)
    assert (root / "gold.patch").read_text() == "gold-diff-v2"


def test_generate_task_assets_rejects_invalid_task_id(tmp_path: Path) -> None:
    with pytest.raises(SWEGymContractError, match="task_id"):
        generate_task_assets(_row("../escape"), tmp_path / "assets", image_id=IMAGE_ID)
    assert not (tmp_path / "escape").exists()


def test_image_tag_lowercases_only_the_repository_component() -> None:
    assert image_tag_for("Project-MONAI__MONAI-1") == (
        "docker.io/xingyaoww/sweb.eval.x86_64.project-monai_s_monai-1:latest"
    )
