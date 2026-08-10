from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from siete_rl.config import load_config
from siete_rl.train import _gpu_baseline, _require_single_visible_gpu
from siete_rl.trainer import SWEGRPOTrainer


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

        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        assert set(config.peft.target_modules) <= module_names
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


def test_cuda_liger_credit_mask_preserves_fixed_g_and_accumulator() -> None:
    _require_single_visible_gpu()

    import torch
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

    def make_case():
        trainer = object.__new__(SWEGRPOTrainer)
        trainer._use_process_mask = False
        trainer.model = SimpleNamespace(training=True)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.beta = 0.0
        trainer.current_gradient_accumulation_steps = 1
        trainer.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=None),
            gather=lambda value: value,
        )
        trainer.liger_loss = LigerFusedLinearGRPOLoss(
            beta=0.0,
            compiled=False,
            use_ref_model=False,
            chunk_size=4,
            loss_type="grpo",
        )
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.5, -0.5]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.5, -0.5]],
            ],
            device="cuda",
        )
        trainer._get_last_hidden_state = lambda *args, **kwargs: hidden
        model = SimpleNamespace(lm_head=torch.nn.Linear(2, 3, bias=False).cuda())
        with torch.no_grad():
            model.lm_head.weight.copy_(
                torch.tensor(
                    [[0.2, -0.1], [0.1, 0.3], [-0.2, 0.4]], device="cuda"
                )
            )
        inputs = {
            "prompt_ids": torch.zeros((4, 1), dtype=torch.long, device="cuda"),
            "prompt_mask": torch.ones((4, 1), dtype=torch.long, device="cuda"),
            "completion_ids": torch.tensor(
                [[0, 1], [1, 2], [0, 1], [1, 2]],
                dtype=torch.long,
                device="cuda",
            ),
            "completion_mask": torch.ones((4, 2), device="cuda"),
            "advantages": torch.tensor(
                [1.0, 0.25, 1.0, 0.25], device="cuda"
            ),
        }
        return trainer, model, inputs

    full_trainer, full_model, full_inputs = make_case()
    full_inputs["token_weights"] = torch.ones((4, 2), device="cuda")
    full_loss = full_trainer.compute_liger_loss(full_model, full_inputs)
    full_loss.backward()
    full_grad = full_model.lm_head.weight.grad.detach().clone()

    censored_trainer, censored_model, censored_inputs = make_case()
    censored_inputs["token_weights"] = torch.tensor(
        [[1, 1], [1, 1], [0, 0], [0, 0]],
        dtype=torch.float32,
        device="cuda",
    )
    censored_loss = censored_trainer.compute_liger_loss(
        censored_model, censored_inputs
    )
    censored_loss.backward()
    accumulated = censored_model.lm_head.weight.grad.detach().clone()

    assert censored_loss.item() == pytest.approx(full_loss.item() / 2)
    assert torch.allclose(accumulated, full_grad / 2, atol=1e-6, rtol=1e-6)

    censored_inputs["token_weights"].zero_()
    zero_loss = censored_trainer.compute_liger_loss(censored_model, censored_inputs)
    zero_loss.backward()
    assert torch.equal(censored_model.lm_head.weight.grad, accumulated)
