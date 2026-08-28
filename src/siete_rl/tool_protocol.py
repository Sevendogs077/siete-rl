"""本地 OpenHands mock-function-calling 协议。"""

from __future__ import annotations

import json
import re
from types import MethodType
from typing import Any


class ToolCallParseError(ValueError):
    """模型输出不是一个可执行的 OpenHands function call。"""


SYSTEM_PROMPT_SUFFIX_TEMPLATE = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
"""

FN_REGEX_PATTERN = r"<function=([^>]+)>\n(.*?)</function>"
FN_PARAM_REGEX_PATTERN = r"<parameter=([^>]+)>(.*?)</parameter>"
FIXED_FAKE_USER = (
    "Please continue working on the task on whatever approach you think is suitable.\n"
    "If you think you have solved the task, please first send your answer to user through message and then finish the interaction.\n"
    "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
)


def _schema(name: str, description: str, properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}}


OPENHANDS_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _schema("execute_bash", """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
""", {"command": {"type": "string", "description": "The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process."}}, ["command"]),
    _schema("finish", "Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.", {}, []),
    _schema("str_replace_editor", """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
""", {
        "command": {"type": "string", "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
        "path": {"type": "string", "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`."},
        "file_text": {"type": "string", "description": "Required parameter of `create` command, with the content of the file to be created."},
        "old_str": {"type": "string", "description": "Required parameter of `str_replace` command containing the string in `path` to replace."},
        "new_str": {"type": "string", "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert."},
        "insert_line": {"type": "integer", "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`."},
        "view_range": {"type": "array", "items": {"type": "integer"}, "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file."},
    }, ["command", "path"]),
)
_BY_NAME = {item["function"]["name"]: item["function"] for item in OPENHANDS_TOOL_SCHEMAS}
_INSTALLED_ATTR = "_swe_agent_openhands_parser_installed"


def render_tool_descriptions(tools: tuple[dict[str, Any], ...] = OPENHANDS_TOOL_SCHEMAS) -> str:
    """使用锁定 converter 的简洁描述格式，保持工具顺序不变。"""
    rendered: list[str] = []
    for index, tool in enumerate(tools, 1):
        function = tool["function"]
        parameters = function["parameters"]
        rendered.extend((f"---- BEGIN FUNCTION #{index}: {function['name']} ----", f"Description: {function['description']}"))
        properties = parameters.get("properties", {})
        if properties:
            rendered.append("Parameters:")
            required = set(parameters.get("required", []))
            for number, (name, detail) in enumerate(properties.items(), 1):
                description = detail.get("description", "No description provided")
                if "enum" in detail:
                    description += "\nAllowed values: [" + ", ".join(f"`{value}`" for value in detail["enum"]) + "]"
                rendered.append(f"  ({number}) {name} ({detail.get('type', 'string')}, {'required' if name in required else 'optional'}): {description}")
        else:
            rendered.append("No parameters are required for this function.")
        rendered.append(f"---- END FUNCTION #{index} ----\n")
    return "\n".join(rendered)


def render_system_suffix() -> str:
    return SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=render_tool_descriptions())


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    value = f"<function={name}>\n"
    for key, argument in arguments.items():
        argument = json.dumps(argument) if isinstance(argument, (list, dict)) else str(argument)
        value += f"<parameter={key}>{argument}</parameter>\n"
    return value + "</function>"


def render_observation(name: str, content: str, *, error: bool = False) -> str:
    return f"EXECUTION RESULT of [{name}]:\n" + ("ERROR:\n" if error else "") + content


def _convert_params(name: str, body: str) -> dict[str, Any]:
    function = _BY_NAME[name]
    properties = function["parameters"]["properties"]
    values: dict[str, Any] = {}
    for match in re.finditer(FN_PARAM_REGEX_PATTERN, body, flags=re.DOTALL):
        key, value = match.group(1), match.group(2).strip()
        if key not in properties:
            raise ToolCallParseError(f"Parameter '{key}' is not allowed for function '{name}'.")
        if key in values:
            raise ToolCallParseError(f"Parameter '{key}' was supplied more than once.")
        expected = properties[key].get("type", "string")
        try:
            if expected == "integer":
                value = int(value)
            elif expected == "array":
                value = json.loads(value)
                if not isinstance(value, list):
                    raise ValueError("not an array")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ToolCallParseError(f"Parameter '{key}' is expected to be {expected}.") from exc
        if "enum" in properties[key] and value not in properties[key]["enum"]:
            raise ToolCallParseError(f"Parameter '{key}' has an invalid value.")
        values[key] = value
    missing = set(function["parameters"].get("required", [])) - values.keys()
    if missing:
        raise ToolCallParseError("Missing required parameter(s): " + ", ".join(sorted(missing)))
    return values


def parse_openhands_text(text: str) -> dict[str, Any]:
    """分类为 plain message、tool call 或 protocol error。

    function call 前的 reasoning 原样保留为 assistant content；function closing tag
    之后的文字不属于 checkpoint 支持的 call，按协议错误处理。
    """
    if not isinstance(text, str):
        raise TypeError("assistant response must be a string")
    matches = list(re.finditer(FN_REGEX_PATTERN, text, flags=re.DOTALL))
    if not matches:
        if "<function=" in text or "</function>" in text:
            return {"kind": "protocol_error", "reason": "incomplete function call", "content": text}
        return {"kind": "message", "content": text}
    if len(matches) != 1:
        return {"kind": "protocol_error", "reason": "only one function call is allowed", "content": text}
    match = matches[0]
    if text[match.end():].strip():
        return {"kind": "protocol_error", "reason": "function call has a suffix", "content": text}
    name = match.group(1)
    if name not in _BY_NAME:
        return {"kind": "protocol_error", "reason": f"unknown function: {name}", "content": text}
    try:
        arguments = _convert_params(name, match.group(2))
    except ToolCallParseError as exc:
        return {"kind": "protocol_error", "reason": str(exc), "content": text}
    return {"kind": "tool", "content": text, "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def install_openhands_tool_protocol(tokenizer: Any) -> Any:
    """幂等安装 TRL 1.8 所需的 parser 与不带 native tools 的 renderer。"""
    if getattr(tokenizer, _INSTALLED_ATTR, False):
        return tokenizer
    original_template = tokenizer.apply_chat_template
    tokenizer._swe_agent_original_apply_chat_template = original_template

    def apply_chat_template(self: Any, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("tools", None)
        return original_template(conversation, *args, **kwargs)

    def parse_response(self: Any, ids: Any, *, prefix: Any = None) -> dict[str, Any]:
        del prefix
        # vLLM 可能把 Qwen 的 turn-end EOS 放进 completion。TRL 会在调用
        # parser 后才清理它；因此这里先移除该唯一的 tokenizer 终止符，避免
        # ``</function><|im_end|>`` 被误判为真实文本后缀。
        text = self.decode(ids, skip_special_tokens=False)
        eos_token = getattr(self, "eos_token", None)
        if isinstance(eos_token, str):
            text = text.removesuffix(eos_token)
        parsed = parse_openhands_text(text)
        if parsed["kind"] == "protocol_error":
            return {"role": "assistant", "content": text, "parse_error": parsed["reason"]}
        if parsed["kind"] == "message":
            return {"role": "assistant", "content": text}
        return {"role": "assistant", "content": text, "tool_calls": parsed["tool_calls"]}

    tokenizer.apply_chat_template = MethodType(apply_chat_template, tokenizer)
    tokenizer.parse_response = MethodType(parse_response, tokenizer)
    # TRL 1.8.0 仅在这两个字段之一非空时调用 parser；这只是激活标志。
    tokenizer.response_template = "openhands-local-parser"
    tokenizer.response_schema = None
    # 该 tokenizer 本身不带 native schema；wrapper 已提供等价的本地协议能力。
    tokenizer.supports_tool_calling = True
    tokenizer.is_chat_template_prefix_preserving = True
    setattr(tokenizer, _INSTALLED_ATTR, True)
    return tokenizer
