from __future__ import annotations

from pathlib import Path

import pytest

from swe_agent import qualify
from swe_agent.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


def test_qualify_main_passes_on_qualified_config() -> None:
    assert qualify.main(["--config", str(CONFIG_PATH), "--no-docker"]) == 0


def test_qualify_detects_revision_drift() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    drifted = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"official_revision": "0" * 40})}
    )
    results = {check.name: check for check in qualify.check_dataset(drifted, project_root)}
    assert results["dataset.revisions"].ok is False


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
    results = {check.name: check for check in qualify.check_dataset(config, project_root)}
    assert results["dataset.cross_table_fields"].ok is False
    assert "base_commit" in results["dataset.cross_table_fields"].detail


def test_qualify_detects_offline_transform_tampering() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    original_transform = qualify.transform_eval_script_offline
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        qualify,
        "transform_eval_script_offline",
        lambda script: original_transform(script) + "\n# tampered\n",
    )
    results = {check.name: check for check in qualify.check_assets(config, project_root)}
    assert results["assets.offline_transform"].ok is False
    monkeypatch.undo()
