from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from trl import GRPOTrainer
from trl.chat_template_utils import (
    is_chat_template_prefix_preserving,
    parse_response,
    supports_tool_calling,
)

from swe_agent.config import load_config
from swe_agent.environment import SWEEnvironment
from swe_agent.models import Task
from swe_agent.prompts import build_prompt
from swe_agent.tool_protocol import ToolCallParseError, install_compatible_tool_call_parser
from swe_agent.tools import validate_tool_arguments
from swe_agent.train import build_processing_class


CONFIG_PATH = "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


@pytest.fixture(scope="module")
def tokenizer():
    config, _, _ = load_config(CONFIG_PATH)
    return build_processing_class(config)


@pytest.fixture(scope="module")
def actual_tools():
    environment = SWEEnvironment(
        task_context={},
        sandbox_factory=lambda *args, **kwargs: None,
        verifier_factory=lambda *args, **kwargs: None,
        output_limit_chars=12_000,
        max_timeout_sec=300,
    )
    return [
        member
        for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]


def test_formal_qwen25_template_renders_exact_swe_tools(tokenizer, actual_tools) -> None:
    assert supports_tool_calling(tokenizer) is True
    assert is_chat_template_prefix_preserving(tokenizer) is True
    assert tokenizer.response_template is not None

    expected_parameters = {
        "list_files": {"path", "max_entries"},
        "read_file": {"path", "start_line", "end_line"},
        "search_code": {"query", "path", "max_matches"},
        "edit_file": {"path", "operation", "old_text", "new_text", "content", "line"},
        "run_command": {"command", "timeout_sec"},
        "submit": set(),
    }
    assert {tool.__name__ for tool in actual_tools} == set(expected_parameters)

    prompt = build_prompt(
        Task(
            task_id="probe",
            repo_name="owner/repo",
            base_commit="0" * 40,
            problem_statement="Inspect the repository before editing.",
        )
    )
    rendered = tokenizer.apply_chat_template(
        prompt,
        tools=actual_tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.count('"type": "function"') == 6
    for name, parameters in expected_parameters.items():
        assert f'"name": "{name}"' in rendered
        for parameter in parameters:
            assert f'"{parameter}"' in rendered
    assert "For each function call" in rendered


def test_local_qwen25_native_tool_round_trip(tokenizer, actual_tools) -> None:
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "moto/api.py", "start_line": 1, "end_line": 2},
            },
        }
    ]
    validate_tool_arguments("read_file", tool_calls[0]["function"]["arguments"])
    prompt = [{"role": "user", "content": "Inspect the repository."}]
    messages = [
        *prompt,
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    prefix = tokenizer.apply_chat_template(
        prompt,
        tools=actual_tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=actual_tools,
        tokenize=True,
        return_dict=False,
    )
    assert rendered[: len(prefix)] == prefix

    parsed = parse_response(tokenizer, rendered[len(prefix) :], prefix=prefix)
    assert parsed["role"] == "assistant"
    assert parsed["content"] == ""
    assert parsed["tool_calls"] == tool_calls
    assert "<|im_end|>" not in str(parsed["tool_calls"])


def test_formal_tokenizer_installs_compatible_parser_once(tokenizer) -> None:
    assert tokenizer._swe_agent_compatible_tool_call_parser_installed is True
    wrapper = tokenizer.parse_response.__func__
    assert install_compatible_tool_call_parser(tokenizer) is tokenizer
    assert tokenizer.parse_response.__func__ is wrapper


@pytest.mark.parametrize(
    "response",
    [
        '<tool_call>\n{"name":"read_file","arguments":{"path":"README.md"}}\n</tool_call>',
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```',
        '{"name":"read_file","arguments":{"path":"README.md"}}',
    ],
)
def test_formal_tokenizer_normalizes_all_three_formats(tokenizer, response: str) -> None:
    parsed = tokenizer.parse_response(response, prefix="")
    assert parsed == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            }
        ],
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_formal_tokenizer_allows_delimiters_inside_compatible_arguments(
    tokenizer, wrapped: bool
) -> None:
    payload = (
        '{"name":"edit_file","arguments":{"content":'
        '"Example:\\n```python\\nprint(1)\\n```\\n<tool_call>demo</tool_call>"}}'
    )
    response = f"```json\n{payload}\n```" if wrapped else payload

    parsed = tokenizer.parse_response(response, prefix="")

    assert parsed["tool_calls"][0]["function"]["arguments"]["content"].startswith(
        "Example:"
    )


def test_formal_tokenizer_marks_ordinary_content_as_parse_error(tokenizer) -> None:
    response = "The documentation mentions <tool_call> and </tool_call> markers."

    parsed = tokenizer.parse_response(response, prefix="")

    assert parsed["role"] == "assistant"
    assert parsed["content"] == response
    assert parsed["parse_error"]


def test_formal_tokenizer_preserves_official_batch_interface(tokenizer) -> None:
    responses = [
        '{"name":"read_file","arguments":{}}',
        '{"name":"submit","arguments":{}}',
    ]

    parsed = tokenizer.parse_response(responses, prefix=["", ""])

    assert parsed == [
        {"role": "assistant", "content": responses[0]},
        {"role": "assistant", "content": responses[1]},
    ]


def test_formal_tokenizer_delegates_batch_parse_errors(tokenizer) -> None:
    responses = [
        '<tool_call>\n{"name":"read_file","arguments":}\n</tool_call>',
        "ordinary content",
    ]

    with pytest.raises(ValueError):
        tokenizer.parse_response(responses, prefix=["", ""])


@pytest.mark.parametrize(
    "response",
    [
        '<tool_call>\n{"name":"read_file","arguments":{}}',
        '</tool_call>\n{"name":"read_file","arguments":{}}',
        '<tool_call>\n{"name":"read_file","arguments":}\n</tool_call>',
        (
            '<tool_call>\n{"name":"read_file","arguments":{}}\n</tool_call>\n'
            '<tool_call>\n{"name":"submit","arguments":{}}\n</tool_call>'
        ),
        (
            'I will inspect first.\n<tool_call>\n'
            '{"name":"read_file","arguments":{}}\n</tool_call>'
        ),
    ],
)
def test_malformed_official_markers_return_parse_error(tokenizer, response: str) -> None:
    parsed = tokenizer.parse_response(response, prefix="")
    assert parsed["role"] == "assistant"
    assert parsed["content"] == response
    assert "tool_calls" not in parsed
    assert parsed["parse_error"]


def test_trl_outer_parser_preserves_malformed_official_as_parse_error(tokenizer, actual_tools) -> None:
    prompt = [{"role": "user", "content": "Inspect the repository."}]
    prefix = tokenizer.apply_chat_template(
        prompt,
        tools=actual_tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    malformed = '<tool_call>\n{"name":"read_file","arguments":{}}'
    completion_ids = tokenizer(
        malformed + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    parsed = parse_response(tokenizer, completion_ids, prefix=prefix)
    assert parsed["role"] == "assistant"
    assert "tool_calls" not in parsed
    assert "<tool_call>" in parsed["content"]
    assert parsed["parse_error"]


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```',
        '{"name":"read_file","arguments":{"path":"README.md"}}',
    ],
)
def test_trl_parser_normalizes_compatible_completion_ids(
    tokenizer, actual_tools, response: str
) -> None:
    prompt = [{"role": "user", "content": "Inspect the repository."}]
    prefix = tokenizer.apply_chat_template(
        prompt,
        tools=actual_tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    completion_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    parsed = parse_response(tokenizer, completion_ids, prefix=prefix)
    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "README.md"},
            },
        }
    ]


def test_trl_parser_marks_trailing_extra_brace_as_parse_error(tokenizer) -> None:
    parsed = tokenizer.parse_response(
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}}\n```',
        prefix="",
    )
    assert "tool_calls" not in parsed
    assert parsed["parse_error"]


def test_trl_outer_parser_passes_parse_error_marker_through(tokenizer, actual_tools) -> None:
    prompt = [{"role": "user", "content": "Inspect the repository."}]
    prefix = tokenizer.apply_chat_template(
        prompt,
        tools=actual_tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    malformed = '```json\n{"name":"read_file","arguments":}\n```'
    completion_ids = tokenizer(malformed + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    parsed = parse_response(tokenizer, completion_ids, prefix=prefix)
    assert parsed["role"] == "assistant"
    assert "tool_calls" not in parsed
    assert parsed["parse_error"]


def test_compatible_message_rerenders_as_qwen_history_with_tool_result(tokenizer, actual_tools) -> None:
    parsed = tokenizer.parse_response(
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```'
    )
    messages = [
        {"role": "user", "content": "Inspect the repository."},
        parsed,
        {"role": "tool", "name": "read_file", "content": "file contents"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=actual_tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "```json" not in rendered
    assert rendered.count('\n<tool_call>\n{"name": "read_file"') == 1
    assert (
        "\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>"
        in rendered
    )
    assert '"name": "read_file"' in rendered
    assert "<tool_response>\nfile contents\n</tool_response>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_tool_messages_preserve_prompt_prefix(tokenizer) -> None:
    tool_call = {
        "type": "function",
        "function": {"name": "read_file", "arguments": {"path": "moto/api.py"}},
    }
    before_tool = [
        {"role": "user", "content": "Inspect the repository."},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
    ]
    after_tool = before_tool + [
        {"role": "tool", "name": "read_file", "content": "file contents"}
    ]
    before_ids = tokenizer.apply_chat_template(before_tool, tokenize=True, return_dict=False)
    after_ids = tokenizer.apply_chat_template(
        after_tool, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    eos_positions = [index for index, token in enumerate(before_ids) if token == tokenizer.eos_token_id]
    assert eos_positions
    trimmed = before_ids[: eos_positions[-1] + 1]
    assert after_ids[: len(trimmed)] == trimmed


class FixtureEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.submitted = False
        self.policy_failed = False
        self.closed = False

    def reset(self, task_id: str, **kwargs: object) -> str:
        del kwargs
        self.submitted = False
        self.policy_failed = False
        self.events.append(f"reset:{task_id}")
        return "ready"

    def inspect(self, path: str) -> str:
        self.events.append(f"inspect:{path}")
        return f"contents:{path}"

    def submit(self) -> str:
        self.events.append("submit")
        self.submitted = True
        return "submitted; answer once without another tool call"

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def test_environment_factory_reset_and_cleanup_are_instance_local() -> None:
    events_a: list[str] = []
    events_b: list[str] = []
    env_a = FixtureEnvironment(events_a)
    env_b = FixtureEnvironment(events_b)
    try:
        assert env_a.reset(task_id="getmoto__moto-7023", prompt="ignored") == "ready"
        assert env_b.reset(task_id="getmoto__moto-7023", prompt="ignored") == "ready"
        assert env_a.inspect("a.py") == "contents:a.py"
        assert env_b.inspect("b.py") == "contents:b.py"
        assert events_a != events_b
    finally:
        env_a.close()
        env_b.close()
    assert env_a.closed and env_b.closed
    assert events_a[-1] == events_b[-1] == "close"


def test_trl_tool_loop_executes_multiple_sync_calls_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    trainer = object.__new__(GRPOTrainer)
    trainer.max_tool_calling_iterations = 2
    trainer.max_completion_length = 32
    trainer.use_vllm = False
    trainer.vllm_mode = "colocate"
    trainer._is_vlm = False
    trainer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))
    trainer._tokenizer = object()
    trainer._sync_tool_dicts = [
        {
            "inspect": lambda path: calls.append(f"inspect:{path}") or "contents",
            "submit": lambda: calls.append("submit") or "submitted",
        }
    ]
    trainer._async_tool_dicts = [{}]
    trainer._get_tool_suffix_ids = lambda messages: [90] * len(messages)
    trainer._generate_single_turn = lambda *args: ([[91]], None)
    monkeypatch.setattr(
        "trl.trainer.grpo_trainer.parse_response",
        lambda *args, **kwargs: {"role": "assistant", "content": "final", "tool_calls": []},
    )

    completions = [
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "inspect", "arguments": {"path": "moto/api.py"}},
                    },
                    {"type": "function", "function": {"name": "submit", "arguments": {}}},
                ],
            }
        ]
    ]
    result = trainer._tool_call_loop(
        prompts=[[{"role": "user", "content": "inspect"}]],
        prompt_ids=[[1]],
        completion_ids=[[2]],
        completions=completions,
        logprobs=None,
        images=None,
        multimodal_fields={},
    )

    tool_mask, returned_completions, completion_ids, _, call_count, failures, _ = result
    assert calls == ["inspect:moto/api.py", "submit"]
    assert call_count == 2
    assert failures == 0
    assert [message["content"] for message in returned_completions[0][1:3]] == [
        "contents",
        "submitted",
    ]
    assert returned_completions[0][-1]["content"] == "final"
    assert completion_ids[0] == [2, 90, 90, 91]
    assert tool_mask[0] == [1, 0, 0, 1]
