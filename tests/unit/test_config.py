from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from siete_rl.config import (
    GenerationConfig,
    GRPOConfigValues,
    ProjectConfig,
    load_config,
)
from siete_rl.scoring import DEFAULT_LAMBDA


def _minimal_grpo_values(**overrides: object) -> GRPOConfigValues:
    """以 7B yaml 的 grpo 段为底，覆盖指定字段后重新校验。"""

    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")["grpo"]
    payload.update(overrides)
    return GRPOConfigValues.model_validate(payload)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_7B = CONFIG_DIR / "grpo_swegym_openhands_7b_lora.yaml"


def _minimal_generation_values(**overrides: object) -> GenerationConfig:
    """最小合法 GenerationConfig 载荷，覆盖指定字段后重新校验。"""
    payload = {
        "max_completion_length": 1024,
        "context_safety_margin": 0,
        "use_liger_kernel": True,
        "max_tool_calling_iterations": 40,
        "max_consecutive_protocol_errors": 5,
        "use_process_mask": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }
    payload.update(overrides)
    return GenerationConfig.model_validate(payload)


def test_parallel_workers_default_to_serial() -> None:
    cfg = _minimal_generation_values()
    assert cfg.reset_parallel_workers == 1
    assert cfg.tool_parallel_workers == 1
    assert cfg.verifier_parallel_workers == 1


def test_parallel_workers_reject_zero() -> None:
    for field in (
        "reset_parallel_workers",
        "tool_parallel_workers",
        "verifier_parallel_workers",
    ):
        with pytest.raises(ValidationError):
            _minimal_generation_values(**{field: 0})


def test_experiment_config_explicitly_declares_parallel_worker_fields() -> None:
    payload = yaml.safe_load(CONFIG_7B.read_text(encoding="utf-8"))

    assert {
        "reset_parallel_workers",
        "tool_parallel_workers",
        "verifier_parallel_workers",
    } <= set(payload["generation"])


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
    config_7b, root_7b, _ = load_config(CONFIG_7B)

    assert root_7b == PROJECT_ROOT
    assert Path(config_7b.model.model_path).is_absolute()
    assert Path(config_7b.dataset.tasks_dir).is_absolute()
    assert config_7b.model.training_mode == "lora"
    assert config_7b.runtime.runtime_qualified is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["peft"].update(target_modules=["q_proj"]),
            "target_modules must be exactly",
        ),
        (
            lambda payload: payload["generation"].update(
                max_completion_length=payload["model"]["context_length"]
            ),
            "exceed model context",
        ),
        (
            lambda payload: payload["vllm"].update(
                max_model_length=payload["model"]["context_length"] + 1
            ),
            "vLLM max model length exceeds",
        ),
        (
            lambda payload: payload["grpo"].update(generation_batch_size=17),
            "divisible by num_generations",
        ),
        (
            lambda payload: payload["grpo"].update(epsilon=0.2, epsilon_high=0.1),
            "epsilon_high",
        ),
        (
            lambda payload: payload["grpo"].update(
                vllm_importance_sampling_clip_min=2.0,
                vllm_importance_sampling_clip_max=1.0,
            ),
            "clip min",
        ),
    ],
    ids=[
        "lora-targets",
        "model-context",
        "vllm-context",
        "generation-groups",
        "epsilon-order",
        "importance-sampling-clip-order",
    ],
)
def test_cross_field_contract_rejects_inconsistent_values(mutate, message: str) -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        ProjectConfig.model_validate(payload)


def test_task_ids_and_max_tasks_count_mismatch_is_rejected() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["dataset"]["task_ids"] = ["a", "b"]
    payload["dataset"]["max_tasks"] = 3
    with pytest.raises(ValidationError, match="disagree"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("task_ids", "max_tasks"),
    [
        (["a", "b"], 2),
        (None, 8),
        (["a"], None),
    ],
)
def test_task_selection_accepts_consistent_combinations(
    task_ids: list[str] | None, max_tasks: int | None
) -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["dataset"].update(task_ids=task_ids, max_tasks=max_tasks)

    validated = ProjectConfig.model_validate(payload)

    assert validated.dataset.task_ids == (tuple(task_ids) if task_ids is not None else None)
    assert validated.dataset.max_tasks == max_tasks


@pytest.mark.parametrize(
    ("task_ids", "max_tasks", "match"),
    [
        (None, 0, "greater than or equal to 1"),
        ([], None, "at least 1 item"),
        (["a", "a"], None, "must not contain duplicates"),
    ],
)
def test_task_selection_rejects_invalid_values(
    task_ids: list[str] | None, max_tasks: int | None, match: str
) -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["dataset"].update(task_ids=task_ids, max_tasks=max_tasks)

    with pytest.raises(ValidationError, match=match):
        ProjectConfig.model_validate(payload)


def test_extra_fields_are_rejected() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        ProjectConfig.model_validate(payload)


def test_lora_cannot_enable_quantization() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["quantization"] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    with pytest.raises(ValidationError, match="LoRA mode"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("peft", "rank", 0),
        ("peft", "dropout", -0.1),
        ("peft", "dropout", 1.1),
        ("chat", "max_prompt_length", 0),
        ("generation", "max_completion_length", 0),
        ("generation", "temperature", 0.0),
        ("generation", "top_p", 0.0),
        ("generation", "top_p", 1.1),
        ("grpo", "num_generations", 1),
        ("grpo", "learning_rate", -1.0),
        ("vllm", "gpu_memory_utilization", 0.0),
        ("vllm", "gpu_memory_utilization", 1.1),
        ("vllm", "max_model_length", 0),
    ],
)
def test_steering_parameters_reject_out_of_range_values(
    section: str, field: str, value: object
) -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload[section][field] = value

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


def test_use_liger_kernel_rejects_sequence_token_importance_sampling() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["grpo"]["importance_sampling_level"] = "sequence_token"
    with pytest.raises(ValidationError, match="use_liger_kernel"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    "mode", ["sequence_mask", "sequence_truncate", "token_mask", "token_truncate"]
)
def test_vllm_importance_sampling_mode_accepts_trl_supported_values(mode: str) -> None:
    """与 TRL 1.8 GRPOTrainer 的运行时枚举保持一致（grpo_trainer.py:2477）。"""
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["grpo"]["vllm_importance_sampling_mode"] = mode

    validated = ProjectConfig.model_validate(payload)

    assert validated.grpo.vllm_importance_sampling_mode == mode


def test_grpo_config_accepts_layered_reward_type() -> None:
    values = _minimal_grpo_values(reward_type="layered")
    assert values.reward_type == "layered"
    assert values.layered_lambda == DEFAULT_LAMBDA


def test_grpo_config_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValidationError):
        _minimal_grpo_values(reward_type="layered", layered_lambda=0.0)


@pytest.mark.parametrize("mode", ["token", "sequence", "token_level", ""])
def test_vllm_importance_sampling_mode_rejects_unsupported_values(mode: str) -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["grpo"]["vllm_importance_sampling_mode"] = mode
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)
