from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swe_agent.config import LORA_TARGET_MODULES, ProjectConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_7B = CONFIG_DIR / "grpo_swegym_qwen2_5_coder_7b_lora.yaml"
CONFIG_30B = CONFIG_DIR / "grpo_swegym_qwen3_coder_30b_a3b_qlora.yaml"


def test_both_complete_configs_parse_independently() -> None:
    config_7b, root_7b, _ = load_config(CONFIG_7B)
    config_30b, root_30b, _ = load_config(CONFIG_30B)

    assert root_7b == root_30b == PROJECT_ROOT
    assert config_7b.model.training_mode == "lora"
    assert config_7b.quantization.load_in_4bit is False
    assert config_7b.runtime.runtime_qualified is True
    assert config_7b.vllm.mode == "server"
    assert config_7b.vllm.tensor_parallel_size is None
    assert config_7b.vllm.server_base_url == "http://127.0.0.1:8000"

    assert config_30b.model.training_mode == "qlora"
    assert config_30b.quantization.load_in_4bit is True
    assert config_30b.quantization.bnb_4bit_quant_type == "nf4"
    assert config_30b.quantization.bnb_4bit_use_double_quant is True
    assert config_30b.runtime.runtime_qualified is False
    assert config_30b.vllm.tensor_parallel_size is None
    assert config_30b.vllm.server_base_url is None


def test_complete_configs_have_same_independent_top_level_shape() -> None:
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in (CONFIG_7B, CONFIG_30B)]
    assert set(payloads[0]) == set(payloads[1]) == {
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
    for payload in payloads:
        assert not ({"include", "extends", "inherit", "overlay"} & set(payload))


@pytest.mark.parametrize("path", [CONFIG_7B, CONFIG_30B])
def test_shared_training_contract_is_fixed(path: Path) -> None:
    config, _, _ = load_config(path)
    assert config.peft.target_modules == LORA_TARGET_MODULES
    assert config.peft.rank == 16
    assert config.peft.alpha == 32
    assert config.peft.dropout == 0.0
    assert config.chat.max_prompt_length == 8192
    assert config.generation.max_completion_length == 22528
    assert config.generation.context_safety_margin == 2048
    assert (
        config.chat.max_prompt_length
        + config.generation.max_completion_length
        + config.generation.context_safety_margin
        == config.model.context_length
    )
    assert config.grpo.num_generations == 4
    assert config.grpo.generation_batch_size == 4
    assert config.grpo.max_steps == 1
    assert config.vllm.use_vllm is True
    assert config.vllm.enable_sleep_mode is (config.vllm.mode == "colocate")


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
