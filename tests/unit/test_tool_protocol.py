from __future__ import annotations

from typing import Any

import pytest

from siete_rl.tool_protocol import (
    FIXED_FAKE_USER,
    OPENHANDS_TOOL_SCHEMAS,
    install_openhands_tool_protocol,
    parse_openhands_text,
    render_observation,
    render_system_suffix,
)


def test_schema_has_the_trajectory_tool_order() -> None:
    assert [item["function"]["name"] for item in OPENHANDS_TOOL_SCHEMAS] == ["execute_bash", "finish", "str_replace_editor"]
    rendered = render_system_suffix()
    assert rendered.count("---- BEGIN FUNCTION") == 3
    assert "<function=example_function_name>" in rendered
    assert "<tool_call>" not in rendered


@pytest.mark.parametrize(("text", "name", "arguments"), [
    ("reasoning\n<function=execute_bash>\n<parameter=command>pwd\nls</parameter>\n</function>", "execute_bash", {"command": "pwd\nls"}),
    ("<function=finish>\n</function>", "finish", {}),
    ("<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/repo/a.py</parameter>\n<parameter=view_range>[1, -1]</parameter>\n</function>", "str_replace_editor", {"command": "view", "path": "/repo/a.py", "view_range": [1, -1]}),
    ("<function=str_replace_editor>\n<parameter=command>insert</parameter>\n<parameter=path>/repo/a.py</parameter>\n<parameter=insert_line>3</parameter>\n</function>", "str_replace_editor", {"command": "insert", "path": "/repo/a.py", "insert_line": 3}),
])
def test_parser_converts_supported_function_calls(text: str, name: str, arguments: dict[str, Any]) -> None:
    parsed = parse_openhands_text(text)
    assert parsed["kind"] == "tool"
    assert parsed["content"] == text
    assert parsed["tool_calls"][0]["function"] == {"name": name, "arguments": arguments}


@pytest.mark.parametrize("text", [
    "just an ordinary assistant message",
    "<function=execute_bash>\n<parameter=command>pwd</parameter>",
    "<function=unknown>\n</function>",
    "<function=execute_bash>\n</function>",
    "<function=str_replace_editor>\n<parameter=command>bad</parameter>\n<parameter=path>/repo</parameter>\n</function>",
    "<function=finish>\n</function> suffix",
    "<function=finish>\n</function><function=finish>\n</function>",
])
def test_plain_and_invalid_outputs_are_classified(text: str) -> None:
    parsed = parse_openhands_text(text)
    if text == "just an ordinary assistant message":
        assert parsed == {"kind": "message", "content": text}
    else:
        assert parsed["kind"] == "protocol_error"


def test_runtime_strings_are_exact_and_terminated() -> None:
    assert render_observation("execute_bash", "ok") == "EXECUTION RESULT of [execute_bash]:\nok"
    assert render_observation("str_replace_editor", "bad", error=True) == "EXECUTION RESULT of [str_replace_editor]:\nERROR:\nbad"
    assert FIXED_FAKE_USER.endswith("\n")


class FixtureTokenizer:
    def decode(self, ids: Any, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return str(ids)

    def apply_chat_template(self, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        return {"conversation": conversation, "args": args, "kwargs": kwargs}


def test_installation_is_idempotent_and_ignores_native_tools() -> None:
    tokenizer = install_openhands_tool_protocol(FixtureTokenizer())
    assert install_openhands_tool_protocol(tokenizer) is tokenizer
    rendered = tokenizer.apply_chat_template([], tools=[{"legacy": True}], tokenize=False)
    assert "tools" not in rendered["kwargs"]
    parsed = tokenizer.parse_response("<function=finish>\n</function>", prefix=[1])
    assert parsed["tool_calls"][0]["function"]["name"] == "finish"
