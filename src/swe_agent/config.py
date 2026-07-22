"""两条模型路径共用的严格、无继承 YAML 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


FIXED_TASK_ID = "getmoto__moto-7023"
FIXED_IMAGE = "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7023:latest"
FIXED_IMAGE_ID = "sha256:8ce447e420f0511fe21b50bc5406b937411b4d829829e82b9b9c1619eeace9de"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class StrictConfig(BaseModel):
    """拒绝未知字段，避免配置拼写错误静默改变训练语义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictConfig):
    task_id: Literal["getmoto__moto-7023"]
    official_path: str
    official_revision: str = Field(min_length=40, max_length=40)
    subset_path: str
    subset_revision: str = Field(min_length=40, max_length=40)
    assets_dir: str


class DockerConfig(StrictConfig):
    image: Literal[
        "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7023:latest"
    ]
    expected_image_id: Literal[
        "sha256:8ce447e420f0511fe21b50bc5406b937411b4d829829e82b9b9c1619eeace9de"
    ]
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
    context_length: Literal[32768]
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
    rank: Literal[16]
    alpha: Literal[32]
    dropout: Literal[0.0]
    bias: Literal["none"]
    target_modules: tuple[str, ...]
    modules_to_save: None


class ChatConfig(StrictConfig):
    native_tool_calling: Literal[True]
    add_response_schema: Literal[True]
    submit_requires_final_response: Literal[True]
    max_prompt_length: Literal[8192]
    max_observation_chars: int = Field(ge=256)


class GenerationConfig(StrictConfig):
    max_completion_length: Literal[22528]
    context_safety_margin: Literal[2048]
    max_tool_calling_iterations: Literal[20]
    temperature: Literal[1.0]
    top_p: Literal[1.0]
    top_k: Literal[0]
    repetition_penalty: Literal[1.0]
    structured_outputs_regex: None


class GRPOConfigValues(StrictConfig):
    reward_type: Literal["binary_verifier"]
    num_generations: Literal[4]
    num_iterations: Literal[1]
    loss_type: Literal["dapo"]
    scale_rewards: Literal["group"]
    multi_objective_aggregation: Literal["sum_then_normalize"]
    epsilon: Literal[0.2]
    epsilon_high: None
    delta: None
    beta: Literal[0.0]
    importance_sampling_level: Literal["token"]
    mask_truncated_completions: Literal[False]
    router_aux_loss_coef: Literal[0.0]
    shuffle_dataset: Literal[True]
    vllm_importance_sampling_correction: Literal[True]
    vllm_importance_sampling_mode: Literal["sequence_mask"]
    vllm_importance_sampling_clip_max: Literal[3.0]
    vllm_importance_sampling_clip_min: None
    per_device_train_batch_size: Literal[1]
    gradient_accumulation_steps: Literal[1]
    generation_batch_size: Literal[4]
    steps_per_generation: None
    max_steps: Literal[1]
    learning_rate: Literal[0.000001]
    weight_decay: Literal[0.0]
    max_grad_norm: Literal[1.0]
    gradient_checkpointing: Literal[True]
    bf16: Literal[True]
    logging_steps: Literal[1]
    save_strategy: Literal["steps"]
    save_steps: Literal[1]
    save_total_limit: Literal[2]
    log_completions: Literal[False]
    report_to: tuple[str, ...]


class VLLMConfig(StrictConfig):
    use_vllm: Literal[True]
    mode: Literal["colocate", "server"]
    model_impl: Literal["vllm"]
    enable_sleep_mode: bool
    tensor_parallel_size: Literal[1] | None
    server_base_url: str | None
    gpu_memory_utilization: Literal[0.3]
    max_model_length: Literal[32768]


class RuntimeConfig(StrictConfig):
    runtime_qualified: bool
    process_count: Literal[1]
    base_seed: int = Field(ge=0)


class OutputConfig(StrictConfig):
    output_root: str = Field(min_length=1)
    run_id: str | None
    train_log: Literal["train.log"]


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
        if self.grpo.generation_batch_size % self.grpo.num_generations != 0:
            raise ValueError("generation_batch_size must be divisible by num_generations")

        if self.runtime.runtime_qualified:
            if self.model.training_mode != "lora":
                raise ValueError("only the first-stage LoRA profile may be runtime-qualified")
            if (
                self.vllm.mode != "server"
                or self.vllm.tensor_parallel_size is not None
                or self.vllm.server_base_url != "http://127.0.0.1:8000"
                or self.vllm.enable_sleep_mode
            ):
                raise ValueError("first-stage runtime requires a separate local vLLM server")
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
