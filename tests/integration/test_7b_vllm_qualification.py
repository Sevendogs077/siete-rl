from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest

from swe_agent.config import load_config
from swe_agent.train import _require_single_visible_gpu


pytestmark = [pytest.mark.gpu, pytest.mark.vllm]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"
TASK_ID = "getmoto__moto-7023"


class QualificationEnvironment:
    """只用于确认真实 Trainer 能绑定同步 environment tool。"""

    def reset(self, task_id: str, **kwargs: object) -> str:
        del kwargs
        return f"ready:{task_id}"

    def inspect_file(self, path: str) -> str:
        """Inspect one file in the qualification environment.

        Args:
            path: Repository-relative file path.
        """

        return f"contents:{path}"


def _zero_reward(completions: list[Any], **kwargs: object) -> list[float]:
    del kwargs
    return [0.0] * len(completions)


def test_qwen25_colocate_sleep_generate_and_peft_weight_sync(tmp_path: Path) -> None:
    """资格化计划固定的真实 Trainer、vLLM sleep/wake 与 merged LoRA 同步。"""

    _require_single_visible_gpu()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema

    from swe_agent.train import _release_trainer, build_trainer

    config, _, _ = load_config(CONFIG_PATH)
    assert config.vllm.mode == "colocate"
    assert config.vllm.tensor_parallel_size == 1
    assert config.vllm.enable_sleep_mode is True
    assert config.vllm.gpu_memory_utilization == 0.3

    tokenizer = add_response_schema(
        AutoTokenizer.from_pretrained(
            config.model.tokenizer_path,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
        )
    )
    rows = [
        {
            "task_id": TASK_ID,
            "prompt": [{"role": "user", "content": "Reply with OK."}],
        }
        for _ in range(config.grpo.num_generations)
    ]

    trainer = None
    backend = None
    lora_b = None
    original_push = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        trainer = build_trainer(
            config,
            output_dir=tmp_path / "trainer-output",
            seed=config.runtime.base_seed,
            train_dataset=Dataset.from_list(rows),
            environment_factory=QualificationEnvironment,
            reward_func=_zero_reward,
            processing_class=tokenizer,
        )
        backend = trainer.vllm_generation
        assert backend.mode == "colocate"
        assert backend.tensor_parallel_size == 1
        assert backend.enable_sleep_mode is True
        assert backend.gpu_memory_utilization == 0.3
        assert len(trainer._environment_pool[None]) == 1
        assert isinstance(trainer._environment_pool[None][0], QualificationEnvironment)

        lora_b_name, lora_b = next(
            (name, parameter)
            for name, parameter in trainer.model.named_parameters()
            if ".lora_B." in name
        )
        target_name = (
            lora_b_name.removeprefix("base_model.model.")
            .split(".lora_B.", maxsplit=1)[0]
            + ".weight"
        )
        pushed_checksums: list[float] = []
        original_push = backend._push_param_to_vllm

        def recording_push(name: str, parameter: torch.Tensor) -> None:
            if name == target_name:
                pushed_checksums.append(parameter.detach().float().sum().item())
            original_push(name, parameter)

        backend._push_param_to_vllm = recording_push
        backend.sync_weights()
        assert len(pushed_checksums) == 1
        checksum_before = pushed_checksums[-1]

        prompt_ids = tokenizer.apply_chat_template(
            rows[0]["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        configured_max_completion_length = backend.max_completion_length
        backend.max_completion_length = 8
        try:
            returned_prompt_ids, completion_ids, logprobs, logprob_token_ids = backend.generate(
                prompts=[prompt_ids],
                images=None,
                num_generations=1,
            )
        finally:
            backend.max_completion_length = configured_max_completion_length
        assert returned_prompt_ids == [prompt_ids]
        assert completion_ids and 0 < len(completion_ids[0]) <= 8
        assert logprobs is not None
        assert logprob_token_ids is not None

        with torch.no_grad():
            lora_b.fill_(0.01)
        backend.sync_weights()
        assert len(pushed_checksums) == 2
        assert pushed_checksums[-1] != pytest.approx(checksum_before)
        assert torch.count_nonzero(lora_b).item() == lora_b.numel()
    finally:
        if backend is not None and original_push is not None:
            backend._push_param_to_vllm = original_push
        original_push = None
        lora_b = None
        backend = None

        class QualificationRecorder:
            messages: list[str] = []

            def log(self, message: str) -> None:
                self.messages.append(message)

        qualification_recorder = QualificationRecorder()
        errors, handles = _release_trainer(trainer, qualification_recorder)
        assert errors == []
        assert {
            (handle["scope"], handle["final_state"])
            for handle in handles
        } == {
            ("vllm_engine", "closed"),
            ("model", "released"),
            ("trainer", "released"),
        }
        trainer = None
        del original_push, lora_b, backend, trainer, tokenizer
        gc.collect()
        if "torch" in locals():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
