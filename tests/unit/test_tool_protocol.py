from __future__ import annotations

from typing import Any

import pytest

from swe_agent.tool_protocol import (
    ToolCallParseError,
    install_compatible_tool_call_parser,
    parse_compatible_tool_call,
)


STANDARD_READ_FILE = {
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


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```',
        '  \n```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```\n ',
        '{"name":"read_file","arguments":{"path":"README.md"}}',
        ' \n {"name":"read_file","arguments":{"path":"README.md"}} \n ',
    ],
)
def test_compatible_formats_are_normalized(text: str) -> None:
    parsed = parse_compatible_tool_call(text)
    assert parsed == STANDARD_READ_FILE


@pytest.mark.parametrize(
    "text",
    [
        "I will inspect the repository first.",
        "# Repository inspection\nUse the available tools.",
        (
            "I will inspect first.\n```json\n"
            '{"name":"read_file","arguments":{"path":"README.md"}}\n```'
        ),
        (
            "Example only: "
            '{"name":"read_file","arguments":{"path":"README.md"}}'
        ),
    ],
)
def test_non_whole_response_examples_are_rejected(text: str) -> None:
    with pytest.raises(ToolCallParseError):
        parse_compatible_tool_call(text)


@pytest.mark.parametrize(
    "text",
    [
        (
            '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```\n'
            "I will continue."
        ),
        (
            '```json\n{"name":"read_file","arguments":{"path":"README.md"}}\n```\n'
            '```json\n{"name":"submit","arguments":{}}\n```'
        ),
        '```\n{"name":"read_file","arguments":{}}\n```',
        '```json\n{"name":"read_file","arguments":{}}',
        '```python\n{"name":"read_file","arguments":{}}\n```',
        "```json\n[]\n```",
        '```json\n"read_file"\n```',
        "```json\n42\n```",
        "```json\ntrue\n```",
        "```json\nnull\n```",
        '```json\n{"name":"read_file","arguments":{}} {}\n```',
        "```json\n{'name':'read_file','arguments':{}}\n```",
        '```json\n{"name":"read_file","arguments":{},}\n```',
        '```json\n{"name":"read_file","arguments":[]}\n```',
        '```json\n{"arguments":{}}\n```',
        '```json\n{"name":"read_file"}\n```',
        '```json\n{"name":"read_file","arguments":{},"extra":1}\n```',
    ],
)
def test_malformed_or_ambiguous_fenced_json_is_rejected(text: str) -> None:
    with pytest.raises(ToolCallParseError):
        parse_compatible_tool_call(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"read_file","arguments":{}} trailing text',
        '{"name":"read_file","arguments":{}} {}',
        "{'name':'read_file','arguments':{}}",
        '{"name":"read_file","arguments":{},}',
        '{"name":"read_file","arguments":"{}"}',
        '{"name":"","arguments":{}}',
        '{"arguments":{}}',
        '{"name":"read_file"}',
        '{"name":"read_file","arguments":{},"extra":1}',
        '{"type":"function","function":{"name":"read_file","arguments":{}}}',
        '{"tool_calls":[]}',
        '{"name":"read_file","parameters":{}}',
    ],
)
def test_malformed_or_non_flat_bare_json_is_rejected(text: str) -> None:
    with pytest.raises(ToolCallParseError):
        parse_compatible_tool_call(text)


def test_protocol_layer_does_not_validate_tool_name_or_business_arguments() -> None:
    parsed = parse_compatible_tool_call(
        '{"name":"not_a_registered_tool","arguments":{"unexpected":"value"}}'
    )
    assert parsed is not None
    assert parsed["tool_calls"][0]["function"] == {
        "name": "not_a_registered_tool",
        "arguments": {"unexpected": "value"},
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_compatible_json_allows_protocol_delimiters_inside_arguments(wrapped: bool) -> None:
    payload = (
        '{"name":"edit_file","arguments":{"content":'
        '"Example:\\n```python\\nprint(1)\\n```\\n<tool_call>demo</tool_call>"}}'
    )
    text = f"```json\n{payload}\n```" if wrapped else payload

    parsed = parse_compatible_tool_call(text)

    assert parsed is not None
    assert parsed["tool_calls"][0]["function"]["arguments"]["content"].startswith("Example:")


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"read_file","arguments":{"path":"README.md"}}}',
        '{"name":"read_file","arguments":{"path":"README.md"}}}}  ',
        '```json\n{"name":"read_file","arguments":{"path":"README.md"}}}\n```',
    ],
)
def test_trailing_extra_braces_are_rejected(text: str) -> None:
    with pytest.raises(ToolCallParseError):
        parse_compatible_tool_call(text)


class FixtureTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    def parse_response(self, response: Any, schema: Any = None, *, prefix: Any = None) -> dict:
        self.calls.append((response, schema, prefix))
        return {"role": "assistant", "content": str(response)}

    def decode(self, response: Any, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return str(response)


def test_installation_is_idempotent_and_forwards_schema_and_prefix() -> None:
    tokenizer = FixtureTokenizer()
    installed = install_compatible_tool_call_parser(tokenizer)
    wrapper = tokenizer.parse_response.__func__
    original = tokenizer._swe_agent_original_parse_response

    assert install_compatible_tool_call_parser(tokenizer) is installed
    assert tokenizer.parse_response.__func__ is wrapper
    assert tokenizer._swe_agent_original_parse_response is original

    prefix = [1, 2, 3]
    parsed = tokenizer.parse_response(
        '{"name":"read_file","arguments":{"path":"README.md"}}',
        prefix=prefix,
    )
    assert parsed == STANDARD_READ_FILE
    assert tokenizer.calls == [
        ('{"name":"read_file","arguments":{"path":"README.md"}}', None, prefix)
    ]


def test_explicit_schema_is_delegated_without_compatibility_fallback() -> None:
    tokenizer = install_compatible_tool_call_parser(FixtureTokenizer())
    schema = {"schema": "sentinel"}
    response = '{"name":"read_file","arguments":{}}'

    parsed = tokenizer.parse_response(response, schema, prefix="prefix")

    assert parsed == {"role": "assistant", "content": response}
    assert tokenizer.calls == [(response, schema, "prefix")]


def test_batch_is_delegated_without_single_response_decoding() -> None:
    tokenizer = install_compatible_tool_call_parser(FixtureTokenizer())
    responses = [
        '{"name":"read_file","arguments":{}}',
        '{"name":"submit","arguments":{}}',
    ]

    parsed = tokenizer.parse_response(responses, prefix=["", ""])

    assert parsed == {"role": "assistant", "content": str(responses)}
    assert tokenizer.calls == [(responses, None, ["", ""])]


def test_malformed_compatible_attempt_returns_parse_error_marker() -> None:
    tokenizer = install_compatible_tool_call_parser(FixtureTokenizer())
    text = '```json\n{"name":"read_file","arguments":}\n```'

    parsed = tokenizer.parse_response(text)

    assert parsed["role"] == "assistant"
    assert parsed["content"] == text
    assert "tool_calls" not in parsed
    assert parsed["parse_error"]


@pytest.mark.parametrize(
    "text",
    [
        "I will inspect the repository.",
        'I will inspect first.\n```json\n{"name":"read_file","arguments":{}}\n```',
        '<tool_call>\n{"name":"read_file","arguments":{}',
        "",
    ],
)
def test_any_non_complete_tool_call_returns_parse_error_marker(text: str) -> None:
    tokenizer = install_compatible_tool_call_parser(FixtureTokenizer())

    parsed = tokenizer.parse_response(text)

    assert parsed["role"] == "assistant"
    assert parsed["content"] == text
    assert "tool_calls" not in parsed
    assert parsed["parse_error"]
