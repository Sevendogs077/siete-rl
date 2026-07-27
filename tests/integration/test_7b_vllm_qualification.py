"""OpenHands 7B server-mode vLLM 配置与协议资格。"""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_agent.config import load_config
from swe_agent.models import Task
from swe_agent.prompts import build_prompt
from swe_agent.tool_protocol import parse_openhands_text
from swe_agent.train import build_processing_class


pytestmark = [pytest.mark.gpu, pytest.mark.vllm]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def test_openhands_server_mode_tokenizer_and_protocol_contract() -> None:
    config, _, _ = load_config(CONFIG_PATH)
    assert config.vllm.mode == "server"
    assert config.vllm.enable_sleep_mode is False
    assert config.vllm.tensor_parallel_size is None
    assert config.vllm.gpu_memory_utilization == 0.25
    tokenizer = build_processing_class(config)
    prompt = build_prompt(Task(task_id="probe", repo_name="owner/repo", base_commit="0" * 40, problem_statement="fix"))
    rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    assert "<tool_call>" not in rendered
    assert parse_openhands_text("<function=finish>\n</function>")["kind"] == "tool"
