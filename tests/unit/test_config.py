from __future__ import annotations

from pathlib import Path

import pytest

from siete_rl.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_7B = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def test_model_and_tokenizer_path_env_overrides_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PATH", "~/models/override-model")
    monkeypatch.setenv("TOKENIZER_PATH", "~/models/override-tokenizer")

    config, _, _ = load_config(CONFIG_7B)

    assert config.model.model_path == (Path.home() / "models/override-model").resolve().as_posix()
    assert config.model.tokenizer_path == (
        Path.home() / "models/override-tokenizer"
    ).resolve().as_posix()


def test_checked_in_config_loads_from_project_root() -> None:
    config, root, _ = load_config(CONFIG_7B)

    assert root == PROJECT_ROOT
    assert Path(config.model.model_path).is_absolute()
    assert Path(config.dataset.tasks_dir).is_absolute()
