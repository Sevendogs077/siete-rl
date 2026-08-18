from __future__ import annotations

from pathlib import Path

import pytest

from siete_rl.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_7B = PROJECT_ROOT / "configs/stage1.yaml"


def test_model_and_tokenizer_path_env_overrides_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PATH", "~/models/override-model")
    monkeypatch.setenv("TOKENIZER_PATH", "~/models/override-tokenizer")
    monkeypatch.setenv("MODEL_ADAPTER_PATH", "~/models/stage1-adapter")

    config, _, _ = load_config(CONFIG_7B)

    assert config.model.model_path == (Path.home() / "models/override-model").resolve().as_posix()
    assert config.model.tokenizer_path == (
        Path.home() / "models/override-tokenizer"
    ).resolve().as_posix()
    assert config.model.adapter_path == (
        Path.home() / "models/stage1-adapter"
    ).resolve().as_posix()


def test_checked_in_config_loads_from_project_root() -> None:
    config, root, _ = load_config(CONFIG_7B)

    assert root == PROJECT_ROOT
    assert Path(config.model.model_path).is_absolute()
    assert Path(config.dataset.train_path).is_absolute()
    assert Path(config.dataset.tasks_dir).is_absolute()


def test_gpu_count_selects_two_gpu_colocate_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_COUNT", "2")

    config, _, _ = load_config(CONFIG_7B)

    assert config.runtime.process_count == 2
    assert config.vllm.tensor_parallel_size == 2


def test_gpu_count_rejects_unsupported_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_COUNT", "3")

    with pytest.raises(ValueError, match="GPU_COUNT must be 2 or 4"):
        load_config(CONFIG_7B)
