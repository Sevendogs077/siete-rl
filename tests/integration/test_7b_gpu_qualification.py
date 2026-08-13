from __future__ import annotations

import gc
from pathlib import Path

import pytest

from siete_rl.config import load_config
from siete_rl.train import _gpu_baseline, _require_single_visible_gpu


pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def test_gpu_baseline_validates_cuda_and_vllm_runtime() -> None:
    physical_device = _require_single_visible_gpu()

    baseline = _gpu_baseline(physical_device)

    assert baseline["physical_device"] == physical_device
    assert baseline["owner_pid"] > 0
    assert baseline["allocated"] >= 0
    assert baseline["reserved"] >= baseline["allocated"]


def test_openhands_7b_bf16_lora_forward_backward_save_reload(tmp_path: Path) -> None:
    _require_single_visible_gpu()

    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from siete_rl.train import build_peft_config

    config, _, _ = load_config(CONFIG_PATH)
    assert config.quantization.load_in_4bit is False
    assert config.model.dtype == "bfloat16"

    model = None
    peft_model = None
    reloaded = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_path,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_path,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
            dtype=torch.bfloat16,
        ).to("cuda")

        peft_model = get_peft_model(
            model,
            build_peft_config(config),
            autocast_adapter_dtype=False,
        )
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()

        trainable = {name for name, parameter in peft_model.named_parameters() if parameter.requires_grad}
        assert trainable
        assert all("lora_" in name for name in trainable)
        assert all(parameter.dtype == torch.bfloat16 for name, parameter in peft_model.named_parameters() if name in trainable)

        inputs = tokenizer("Fix the failing function.", return_tensors="pt")
        inputs = {name: value.to("cuda") for name, value in inputs.items()}
        outputs = peft_model(**inputs, labels=inputs["input_ids"], use_cache=False)
        outputs.loss.backward()
        nonzero_gradients = [
            name
            for name, parameter in peft_model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        ]
        assert nonzero_gradients

        adapter_dir = tmp_path / "adapter"
        peft_model.save_pretrained(adapter_dir)
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()

        model = peft_model.unload()
        peft_model = None
        reloaded = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        assert any(parameter.requires_grad for parameter in reloaded.parameters())
    finally:
        del reloaded, peft_model, model
        gc.collect()
        if "torch" in locals():
            torch.cuda.empty_cache()
