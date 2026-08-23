from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl import GRPOTrainer

from siete_rl.trainer import SWEGRPOTrainer


def _model():
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    return get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=2,
            target_modules=["q_proj"],
        ),
    )


def _fill_adapter(model, adapter_name: str, value: float) -> None:
    for name, parameter in model.named_parameters():
        if f".{adapter_name}." in name:
            parameter.data.fill_(value)


def _adapter_tensors(model, adapter_name: str) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if f".{adapter_name}." in name
    ]


def _resume_trainer(monkeypatch, checkpoint: Path):
    model = PeftModel.from_pretrained(_model().unload(), checkpoint, is_trainable=True)

    def initialize_upstream(trainer, *args, **kwargs):
        del args
        trainer.model = kwargs["model"]
        trainer.model.add_adapter("ref", trainer.model.peft_config["default"])
        for name, parameter in trainer.model.named_parameters():
            if ".default." in name:
                trainer.model.get_parameter(name.replace(".default.", ".ref.")).data.copy_(
                    parameter.data
                )
        trainer.use_liger_kernel = True
        trainer.vllm_mode = "server"

    monkeypatch.setattr(GRPOTrainer, "__init__", initialize_upstream)
    return SWEGRPOTrainer(
        model=model,
        max_consecutive_protocol_errors=1,
        use_process_mask=True,
        preloaded_checkpoint=checkpoint,
    )


def test_stage1_resume_keeps_base_model_as_reference(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    model = _model()
    _fill_adapter(model, "default", 2.0)
    model.save_pretrained(checkpoint)

    resumed = _resume_trainer(monkeypatch, checkpoint)

    assert "ref" not in resumed.model.peft_config


def test_stage2_resume_restores_saved_reference_adapter(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    model = _model()
    _fill_adapter(model, "default", 2.0)
    model.add_adapter("ref", model.peft_config["default"])
    _fill_adapter(model, "ref", 1.0)
    expected = _adapter_tensors(model, "ref")
    model.save_pretrained(checkpoint)

    resumed = _resume_trainer(monkeypatch, checkpoint)

    actual = _adapter_tensors(resumed.model, "ref")
    assert len(actual) == len(expected)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
