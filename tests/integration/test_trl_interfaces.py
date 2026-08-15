"""固定 Transformers/TRL 版本下的 OpenHands tokenizer 接口回归。"""

from __future__ import annotations

import pytest

from siete_rl.config import load_config
from siete_rl.models import Task
from siete_rl.prompts import build_prompt
from siete_rl.train import build_processing_class
from siete_rl.trainer import SWEGRPOTrainer

# 需要本地模型 tokenizer，CI/无资产环境下默认不跑
pytestmark = pytest.mark.external_assets


CONFIG_PATH = "configs/stage1.yaml"


@pytest.fixture(scope="module")
def tokenizer():
    config, _, _ = load_config(CONFIG_PATH)
    return build_processing_class(config)


def test_real_tokenizer_renders_only_the_local_openhands_scaffold(tokenizer) -> None:
    prompt = build_prompt(Task(task_id="probe", repo_name="owner/repo", base_commit="0" * 40, problem_statement="fix"))
    rendered = tokenizer.apply_chat_template(prompt, tools=[{"type": "function", "function": {"name": "legacy"}}], tokenize=False, add_generation_prompt=True)
    assert getattr(tokenizer, "supports_tool_calling", False)
    assert getattr(tokenizer, "is_chat_template_prefix_preserving", False)
    assert tokenizer.response_template == "openhands-local-parser"
    assert tokenizer.response_schema is None
    assert rendered.count("---- BEGIN FUNCTION") == 3
    assert "<tool_call>" not in rendered


def test_real_tokenizer_parser_handles_openhands_function_text(tokenizer) -> None:
    text = "<function=finish>\n</function>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = tokenizer.parse_response(ids, prefix=[])
    assert parsed["tool_calls"][0]["function"] == {"name": "finish", "arguments": {}}


def test_real_tokenizer_parser_ignores_only_the_terminal_eos_token(tokenizer) -> None:
    text = "<function=finish>\n</function>" + tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)

    parsed = tokenizer.parse_response(ids, prefix=[])

    assert parsed["tool_calls"][0]["function"] == {"name": "finish", "arguments": {}}


def test_openhands_user_turn_suffix_is_a_plain_token_list(tokenizer) -> None:
    trainer = object.__new__(SWEGRPOTrainer)
    trainer._tokenizer = tokenizer
    trainer._get_tool_suffix_ids = lambda messages: (_ for _ in ()).throw(AssertionError(messages))

    suffix = trainer._get_openhands_suffix_ids([
        {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\npwd"}
    ])

    assert suffix and all(isinstance(token, int) for token in suffix)
