from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swe_agent.config import LORA_TARGET_MODULES, ProjectConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_7B = CONFIG_DIR / "grpo_swegym_qwen2_5_coder_7b_lora.yaml"


def test_complete_config_parses_independently() -> None:
    config_7b, root_7b, _ = load_config(CONFIG_7B)

    assert root_7b == PROJECT_ROOT
    assert config_7b.model.training_mode == "lora"
    assert config_7b.quantization.load_in_4bit is False
    assert config_7b.runtime.runtime_qualified is True
    assert config_7b.vllm.mode == "server"
    assert config_7b.vllm.tensor_parallel_size is None
    assert config_7b.vllm.server_base_url == "http://127.0.0.1:8000"


def test_complete_config_has_independent_top_level_shape() -> None:
    payload = yaml.safe_load(CONFIG_7B.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "dataset",
        "docker",
        "model",
        "quantization",
        "peft",
        "chat",
        "generation",
        "grpo",
        "vllm",
        "runtime",
        "output",
    }
    assert not ({"include", "extends", "inherit", "overlay"} & set(payload))


def test_shared_training_contract_is_fixed() -> None:
    """只锁与调参值无关的结构不变量；具体数值（步数、组大小、学习率等）不属于契约。"""

    config, _, _ = load_config(CONFIG_7B)
    assert config.peft.target_modules == LORA_TARGET_MODULES
    # 上下文预算方程：prompt + completion + margin 必须恰好顶满模型上下文
    assert (
        config.chat.max_prompt_length
        + config.generation.max_completion_length
        + config.generation.context_safety_margin
        == config.model.context_length
    )
    # GRPO 组约束：generation batch 必须整除出完整的组；
    # 多任务数据集最多 100 行，RepeatSampler 每 batch 的 unique prompt 数不能超过数据集行数。
    assert config.grpo.num_generations >= 2
    assert config.grpo.generation_batch_size % config.grpo.num_generations == 0
    assert config.grpo.generation_batch_size // config.grpo.num_generations <= 100
    # TRL 节拍对齐：generation_batch_size == pdbs × steps_per_generation（缺省取 accum）
    steps_per_generation = (
        config.grpo.steps_per_generation or config.grpo.gradient_accumulation_steps
    )
    assert (
        config.grpo.generation_batch_size
        == config.grpo.per_device_train_batch_size * steps_per_generation
    )
    assert config.grpo.max_steps >= 1
    assert config.vllm.use_vllm is True
    assert config.vllm.enable_sleep_mode is (config.vllm.mode == "colocate")
    assert config.dataset.tasks_dir.endswith("assets/swegym")
    assert config.dataset.task_ids is None or len(config.dataset.task_ids) >= 1
    assert not hasattr(config.docker, "image")


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


def test_steering_parameters_accept_theoretical_boundaries() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["model"]["context_length"] = 32768
    payload["peft"].update(rank=1, alpha=1, dropout=1.0)
    payload["chat"].update(max_prompt_length=1, max_observation_chars=1)
    payload["generation"].update(
        max_completion_length=1,
        context_safety_margin=0,
        max_tool_calling_iterations=1,
        max_consecutive_format_errors=1,
        temperature=0.1,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.1,
        structured_outputs_regex=r"\{.*\}",
    )
    payload["grpo"].update(
        num_generations=2,
        num_iterations=1,
        epsilon=0.0,
        epsilon_high=0.0,
        delta=0.0,
        beta=0.0,
        mask_truncated_completions=True,
        router_aux_loss_coef=0.0,
        shuffle_dataset=False,
        vllm_importance_sampling_correction=False,
        vllm_importance_sampling_clip_max=1.0,
        vllm_importance_sampling_clip_min=0.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        steps_per_generation=1,
        max_steps=1,
        learning_rate=0.0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        gradient_checkpointing=False,
        bf16=False,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        log_completions=True,
    )
    payload["vllm"].update(gpu_memory_utilization=1.0, max_model_length=32768)

    validated = ProjectConfig.model_validate(payload)

    assert validated.peft.rank == 1
    assert validated.peft.dropout == 1.0
    assert validated.vllm.gpu_memory_utilization == 1.0
    assert validated.grpo.steps_per_generation == 1


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
