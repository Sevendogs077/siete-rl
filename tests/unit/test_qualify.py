from __future__ import annotations

from pathlib import Path

import pytest

from siete_rl import qualify

# 整个文件都依赖私有 data/assets 下的真实数据集资产。
pytestmark = pytest.mark.external_assets
from siete_rl.config import load_config
from siete_rl.swegym import select_task_ids


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def test_qualify_main_passes_on_qualified_config() -> None:
    assert qualify.main(["--config", str(CONFIG_PATH), "--no-docker"]) == 0


def test_qualify_detects_revision_drift() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    drifted = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"official_revision": "0" * 40})}
    )
    results = qualify.check_dataset(drifted, project_root)
    assert results
    assert all(check.ok is False for check in results if check.name.endswith(".revisions"))


def test_qualify_covers_every_selected_task() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    task_ids = select_task_ids(config, project_root)
    results = qualify.check_dataset(config, project_root)
    checked = {
        check.name.removeprefix("dataset.").removesuffix(".official_row")
        for check in results
        if check.name.endswith(".official_row")
    }
    assert checked == set(task_ids)


def test_qualify_detects_cross_table_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    original = qualify._read_exact_row
    calls = 0

    def mismatched(path: Path, instance_id: str):
        nonlocal calls
        calls += 1
        row = dict(original(path, instance_id))
        if calls == 2:
            row["base_commit"] = "0" * 40
        return row

    monkeypatch.setattr(qualify, "_read_exact_row", mismatched)
    results = qualify.check_dataset(config, project_root)
    mismatched = [
        check for check in results if check.name.endswith(".cross_table_fields") and not check.ok
    ]
    assert len(mismatched) == 1
    assert "base_commit" in mismatched[0].detail


def test_qualify_detects_offline_transform_tampering() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    original_transform = qualify.transform_eval_script_offline
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        qualify,
        "transform_eval_script_offline",
        lambda script: original_transform(script) + "\n# tampered\n",
    )
    results = qualify.check_assets(config, project_root)
    assert any(
        check.name.endswith(".offline_transform") and check.ok is False for check in results
    )
    monkeypatch.undo()
