"""严格接受完整 Qwen tool call、bare JSON 或 fenced JSON 工具调用。"""

from __future__ import annotations

import json
from types import MethodType
from typing import Any


class ToolCallParseError(ValueError):
    """可预期的工具调用序列化或最小结构错误。"""


_INSTALLED_ATTR = "_swe_agent_compatible_tool_call_parser_installed"
_ORIGINAL_ATTR = "_swe_agent_original_parse_response"


def parse_compatible_tool_call(visible_text: str) -> dict[str, Any]:
    """严格解析完整 fenced/bare JSON；其他任意输出均为协议错误。"""

    if not isinstance(visible_text, str):
        raise TypeError("visible assistant content must be a string")
    text = visible_text.strip()
    if not text:
        raise ToolCallParseError("assistant response must contain one complete tool call")

    if text.startswith("```"):
        payload = _parse_fenced_object(text)
        return _standard_message(payload)

    if not text.startswith("{"):
        raise ToolCallParseError(
            "assistant response must be one complete Qwen tool call, JSON object, or fenced JSON block"
        )
    payload = _load_strict_object(text)
    return _standard_message(payload)


def install_compatible_tool_call_parser(tokenizer: Any) -> Any:
    """在官方 tokenizer parser 外幂等安装两个严格 JSON fallback。"""

    if getattr(tokenizer, _INSTALLED_ATTR, False):
        return tokenizer

    original = getattr(tokenizer, "parse_response", None)
    if not callable(original):
        raise TypeError("tokenizer must provide parse_response before compatibility installation")

    def compatible_parse_response(
        self: Any,
        response: Any,
        schema: list[Any] | dict[str, Any] | None = None,
        *,
        prefix: Any = None,
    ) -> Any:
        if schema is not None or _is_batched_response(response):
            return original(response, schema, prefix=prefix)

        official: Any = None
        try:
            official = original(response, schema, prefix=prefix)
        except (ValueError, TypeError):
            # 预期的官方解析失败仍需检查协议标记和两个严格兼容分支。
            official = None

        visible_text = _decode_response(self, response, skip_special_tokens=True)
        try:
            if isinstance(official, dict) and official.get("tool_calls"):
                _validate_official_message(official)
                _validate_official_envelope(visible_text)
                normalized_official = dict(official)
                normalized_official["content"] = ""
                return normalized_official
            compatible = parse_compatible_tool_call(visible_text)
        except ToolCallParseError as exc:
            return {
                "role": "assistant",
                "content": visible_text,
                "parse_error": str(exc),
            }
        return compatible

    compatible_parse_response.__name__ = "parse_response"
    compatible_parse_response.__qualname__ = type(tokenizer).__name__ + ".parse_response"
    setattr(tokenizer, _ORIGINAL_ATTR, original)
    setattr(tokenizer, "parse_response", MethodType(compatible_parse_response, tokenizer))
    setattr(tokenizer, _INSTALLED_ATTR, True)
    return tokenizer


def _validate_official_envelope(visible_text: str) -> None:
    text = visible_text.strip()
    if not text.startswith("<tool_call>") or not text.endswith("</tool_call>"):
        raise ToolCallParseError("official Qwen tool call must use one complete outer envelope")


def _validate_official_message(message: dict[str, Any]) -> None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ToolCallParseError("official response must contain exactly one tool call")
    call = calls[0]
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ToolCallParseError("official tool call type must be function")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ToolCallParseError("official tool call function must be an object")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError("official tool call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("official tool call arguments must be an object")
    content = message.get("content", "")
    if content is not None and (not isinstance(content, str) or content.strip()):
        raise ToolCallParseError("official tool call content must be empty")


def _parse_fenced_object(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if len(lines) < 3:
        raise ToolCallParseError("assistant response is not one complete fenced block")
    if lines[0] != "```json":
        raise ToolCallParseError("fenced tool call language must be json")
    if lines[-1] != "```":
        raise ToolCallParseError("assistant response is not one complete fenced block")
    return _load_strict_object("\n".join(lines[1:-1]))


def _is_batched_response(response: Any) -> bool:
    if isinstance(response, (list, tuple)):
        if not response:
            return False
        return isinstance(response[0], (str, list, tuple))
    ndim = getattr(response, "ndim", None)
    return isinstance(ndim, int) and ndim >= 2


def _load_strict_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError("tool call must be one complete valid JSON value") from exc
    if not isinstance(value, dict):
        raise ToolCallParseError("tool call JSON must be an object")
    if set(value) != {"name", "arguments"}:
        raise ToolCallParseError("tool call object must contain exactly name and arguments")
    name = value["name"]
    arguments = value["arguments"]
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError("tool call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("tool call arguments must be an object")
    return value


def _standard_message(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": payload["arguments"],
                },
            }
        ],
    }


def _decode_response(tokenizer: Any, response: Any, *, skip_special_tokens: bool) -> str:
    if isinstance(response, str):
        return response
    decoded = tokenizer.decode(response, skip_special_tokens=skip_special_tokens)
    if not isinstance(decoded, str):
        raise TypeError("tokenizer.decode must return a string for one response")
    return decoded
