"""两条模型路径共用的严格、无继承 YAML 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class StrictConfig(BaseModel):
    """拒绝未知字段，避免配置拼写错误静默改变训练语义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictConfig):
    task_id: str = Field(min_length=1)
    official_path: str
    official_revision: str = Field(min_length=40, max_length=40)
    subset_path: str
    subset_revision: str = Field(min_length=40, max_length=40)
    assets_dir: str


class DockerConfig(StrictConfig):
    image: str = Field(min_length=1)
    expected_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_registry_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: Literal["linux/amd64"]
    pull_policy: Literal["never"]
    network_mode: Literal["none"]
    host_mounts: Literal[False]
    cpus: float = Field(gt=0)
    memory: str = Field(min_length=2)
    pids_limit: int = Field(gt=0)
    exec_timeout_sec: int = Field(gt=0)
    verifier_timeout_sec: int = Field(gt=0)


class ModelConfig(StrictConfig):
    provenance_id: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    tokenizer_path: str = Field(min_length=1)
    architecture: Literal["Qwen2ForCausalLM", "Qwen3MoeForCausalLM"]
    context_length: int = Field(ge=1)
    trust_remote_code: bool
    dtype: Literal["bfloat16"]
    training_mode: Literal["lora", "qlora"]


class QuantizationConfig(StrictConfig):
    load_in_4bit: bool
    bnb_4bit_quant_type: Literal["nf4"] | None
    bnb_4bit_compute_dtype: Literal["bfloat16"] | None
    bnb_4bit_use_double_quant: bool | None


class PeftConfig(StrictConfig):
    task_type: Literal["CAUSAL_LM"]
    rank: int = Field(ge=1)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"]
    target_modules: tuple[str, ...]
    modules_to_save: None


class ChatConfig(StrictConfig):
    native_tool_calling: Literal[True]
    add_response_schema: Literal[True]
    max_prompt_length: int = Field(ge=1)
    max_observation_chars: int = Field(ge=1)


class GenerationConfig(StrictConfig):
    max_completion_length: int = Field(ge=1)
    context_safety_margin: int = Field(ge=0)
    max_tool_calling_iterations: int = Field(ge=1)
    max_consecutive_format_errors: int = Field(ge=1)
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    repetition_penalty: float = Field(gt=0.0)
    structured_outputs_regex: str | None


class GRPOConfigValues(StrictConfig):
    reward_type: Literal["binary_verifier"]
    num_generations: int = Field(ge=2)
    num_iterations: int = Field(ge=1)
    loss_type: Literal["grpo", "dapo", "bnpo", "dr_grpo"]
    scale_rewards: Literal["group", "batch", "none"]
    multi_objective_aggregation: Literal["sum_then_normalize", "normalize_then_sum"]
    epsilon: float = Field(ge=0.0)
    epsilon_high: float | None = Field(ge=0.0)
    delta: float | None = Field(ge=0.0)
    beta: float = Field(ge=0.0)
    importance_sampling_level: Literal["token", "sequence", "sequence_token"]
    mask_truncated_completions: bool
    router_aux_loss_coef: float = Field(ge=0.0)
    shuffle_dataset: bool
    vllm_importance_sampling_correction: bool
    vllm_importance_sampling_mode: Literal["sequence_mask", "sequence", "token"]
    vllm_importance_sampling_clip_max: float = Field(gt=0.0)
    vllm_importance_sampling_clip_min: float | None = Field(ge=0.0)
    per_device_train_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    generation_batch_size: int = Field(ge=1)
    steps_per_generation: int | None = Field(ge=1)
    max_steps: int = Field(ge=1)
    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    max_grad_norm: float = Field(ge=0.0)
    gradient_checkpointing: bool
    bf16: bool
    logging_steps: int = Field(ge=1)
    save_strategy: Literal["steps", "epoch", "no"]
    save_steps: int = Field(ge=1)
    save_total_limit: int = Field(ge=1)
    log_completions: bool
    report_to: tuple[str, ...]


class VLLMConfig(StrictConfig):
    use_vllm: Literal[True]
    mode: Literal["colocate", "server"]
    model_impl: Literal["vllm"]
    enable_sleep_mode: bool
    tensor_parallel_size: int | None = Field(ge=1)
    server_base_url: str | None
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_model_length: int = Field(ge=1)


class RuntimeConfig(StrictConfig):
    runtime_qualified: bool
    process_count: int = Field(ge=1)
    base_seed: int = Field(ge=0)


class OutputConfig(StrictConfig):
    output_root: str = Field(min_length=1)
    run_id: str | None
    train_log: str = Field(min_length=1)


class ProjectConfig(StrictConfig):
    schema_version: Literal[1]
    dataset: DatasetConfig
    docker: DockerConfig
    model: ModelConfig
    quantization: QuantizationConfig
    peft: PeftConfig
    chat: ChatConfig
    generation: GenerationConfig
    grpo: GRPOConfigValues
    vllm: VLLMConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.peft.target_modules != LORA_TARGET_MODULES:
            raise ValueError(
                "peft.target_modules must be exactly q_proj,k_proj,v_proj,o_proj"
            )

        quantization = self.quantization
        if self.model.training_mode == "lora":
            if quantization.load_in_4bit or any(
                value is not None
                for value in (
                    quantization.bnb_4bit_quant_type,
                    quantization.bnb_4bit_compute_dtype,
                    quantization.bnb_4bit_use_double_quant,
                )
            ):
                raise ValueError("LoRA mode must not define a 4-bit quantization config")
        else:
            if (
                not quantization.load_in_4bit
                or quantization.bnb_4bit_quant_type != "nf4"
                or quantization.bnb_4bit_compute_dtype != "bfloat16"
                or quantization.bnb_4bit_use_double_quant is not True
            ):
                raise ValueError("QLoRA mode requires NF4 BF16 double-quant configuration")

        total_context = (
            self.chat.max_prompt_length
            + self.generation.max_completion_length
            + self.generation.context_safety_margin
        )
        if total_context > self.model.context_length:
            raise ValueError("prompt, completion and safety margin exceed model context")
        if self.vllm.max_model_length > self.model.context_length:
            raise ValueError("vLLM max model length exceeds model context")
        if (
            self.chat.max_prompt_length + self.generation.max_completion_length
            > self.vllm.max_model_length
        ):
            raise ValueError("prompt and completion exceed vLLM max model length")
        if self.grpo.generation_batch_size % self.grpo.num_generations != 0:
            raise ValueError("generation_batch_size must be divisible by num_generations")
        if (
            self.grpo.epsilon_high is not None
            and self.grpo.epsilon_high < self.grpo.epsilon
        ):
            raise ValueError("epsilon_high must be greater than or equal to epsilon")
        if (
            self.grpo.vllm_importance_sampling_clip_min is not None
            and self.grpo.vllm_importance_sampling_clip_min
            > self.grpo.vllm_importance_sampling_clip_max
        ):
            raise ValueError(
                "vLLM importance-sampling clip min must not exceed clip max"
            )

        if self.runtime.runtime_qualified:
            if (
                self.vllm.mode != "server"
                or self.vllm.tensor_parallel_size is not None
                or self.vllm.server_base_url != "http://127.0.0.1:8000"
                or self.vllm.enable_sleep_mode
            ):
                raise ValueError("qualified runtime requires a separate local vLLM server")
        elif self.vllm.tensor_parallel_size is not None or self.vllm.server_base_url is not None:
            raise ValueError("an unqualified runtime must not activate GPU topology")
        return self


def load_config(path: str | Path) -> tuple[ProjectConfig, Path, Path]:
    """严格读取配置并解析项目内路径，不检查尚未迁移的领域资产。"""

    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取配置 {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("配置顶层必须是对象")

    config = ProjectConfig.model_validate(payload)
    project_root = find_project_root(config_path)
    resolved = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "official_path": resolve_path(project_root, config.dataset.official_path),
                    "subset_path": resolve_path(project_root, config.dataset.subset_path),
                    "assets_dir": resolve_path(project_root, config.dataset.assets_dir),
                }
            ),
            "model": config.model.model_copy(
                update={
                    "model_path": resolve_path(project_root, config.model.model_path),
                    "tokenizer_path": resolve_path(project_root, config.model.tokenizer_path),
                }
            ),
            "output": config.output.model_copy(
                update={
                    "output_root": resolve_path(project_root, config.output.output_root)
                }
            ),
        }
    )
    return ProjectConfig.model_validate(resolved.model_dump(mode="python")), project_root, config_path


def find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"无法从配置路径定位项目根目录: {config_path}")


def resolve_path(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve().as_posix()
