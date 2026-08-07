"""OpenHands 三工具的领域校验与容器执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from siete_rl.docker import CommandResult, DockerSandbox
from siete_rl.models import Action, Observation
from siete_rl.openhands_editor import ContainerFileBackend, OpenHandsEditor, ToolError
from siete_rl.tool_protocol import OPENHANDS_TOOL_SCHEMAS


class ToolContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]

    @property
    def native_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": self.properties, "required": list(self.required)}}}


TOOL_SPECS = tuple(ToolSpec(item["function"]["name"], item["function"]["description"], item["function"]["parameters"]["properties"], tuple(item["function"]["parameters"].get("required", []))) for item in OPENHANDS_TOOL_SCHEMAS)
BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def native_tool_schemas() -> list[dict[str, Any]]:
    return [spec.native_schema for spec in TOOL_SPECS]


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    if name not in BY_NAME: raise ToolContractError(f"unsupported tool: {name}")
    if not isinstance(arguments, dict): raise ToolContractError("tool arguments must be an object")
    spec = BY_NAME[name]; missing = set(spec.required) - arguments.keys(); unknown = arguments.keys() - spec.properties.keys()
    if missing: raise ToolContractError("missing tool arguments: " + ", ".join(sorted(missing)))
    if unknown: raise ToolContractError("unknown tool arguments: " + ", ".join(sorted(unknown)))
    for key, value in arguments.items():
        schema = spec.properties[key]; expected = schema["type"]
        valid = isinstance(value, str) if expected == "string" else isinstance(value, int) and not isinstance(value, bool) if expected == "integer" else isinstance(value, list) if expected == "array" else False
        if not valid: raise ToolContractError(f"{key} must have type {expected}")
        if "enum" in schema and value not in schema["enum"]: raise ToolContractError(f"{key} is not an allowed value")


# 循环检测：措辞沿用 SFT 轨迹中 <NOTE> 的形态（参照 openhands_editor.CONTENT_TRUNCATED_NOTICE）
_REPEAT_NOTE = (
    "\n<NOTE>You have repeated this exact action {count} times without making progress. "
    "Try a different approach, or call finish if you believe the task is complete.</NOTE>"
)


def _action_signature(action: Action) -> tuple[object, ...] | None:
    """重复判定的动作签名；finish 不参与（同一 episode 只会成功执行一次）。"""

    a = action.arguments
    if action.tool_name == "execute_bash":
        return ("execute_bash", a.get("command", ""))
    if action.tool_name == "str_replace_editor":
        return ("str_replace_editor", a.get("command", ""), a.get("path", ""), (a.get("old_str") or "")[:200])
    return None


class ToolExecutor:
    def __init__(self, sandbox: DockerSandbox, *, output_limit_chars: int, max_timeout_sec: int, workspace: str, max_repeat_action: int | None = None) -> None:
        self.sandbox, self.output_limit_chars, self.max_timeout_sec, self.workspace = sandbox, output_limit_chars, max_timeout_sec, workspace
        self.max_repeat_action = max_repeat_action
        self._repeat_counts: dict[tuple[object, ...], int] = {}
        self.submitted_patch: str | None = None
        self.editor = OpenHandsEditor(ContainerFileBackend(sandbox, timeout_sec=max_timeout_sec))

    def execute(self, action: Action) -> Observation:
        repeat_count = self._count_repeat(action)
        try:
            if action.tool_name == "execute_bash": code, text, timed_out = self._execute_bash(action.arguments)
            elif action.tool_name == "str_replace_editor": code, text, timed_out = 0, self.editor(**action.arguments), False
            elif action.tool_name == "finish": code, text, timed_out = self._finish(action.arguments)
            else: raise ToolError(f"unsupported tool: {action.tool_name}")
            error_type = None if code == 0 else "tool_error"
        except (ToolError, ValueError) as exc:
            code, text, timed_out, error_type = 1, str(exc), False, "tool_error"
        text, truncated = self._bounded(text)
        if repeat_count == self.max_repeat_action:
            text += _REPEAT_NOTE.format(count=repeat_count)
        return Observation(text=text, exit_code=code, error_type=error_type, timed_out=timed_out, truncated=truncated)

    def _count_repeat(self, action: Action) -> int:
        if self.max_repeat_action is None:
            return 0
        signature = _action_signature(action)
        if signature is None:
            return 0
        count = self._repeat_counts.get(signature, 0) + 1
        self._repeat_counts[signature] = count
        return count

    def _execute_bash(self, arguments: dict[str, Any]) -> tuple[int, str, bool]:
        command = arguments["command"]
        result = self.sandbox.exec(["/bin/bash", "-lc", f"cd {self.workspace} && {command}"], timeout_sec=self.max_timeout_sec)
        output = _command_output(result) + f"\n[Command finished with exit code {result.exit_code}]"
        return result.exit_code, output, result.timed_out

    def _finish(self, arguments: dict[str, Any]) -> tuple[int, str, bool]:
        if arguments: raise ToolError("finish does not accept arguments")
        self.submitted_patch = self.sandbox.get_diff()
        return 0, "", False

    def _bounded(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.output_limit_chars: return text, False
        half = self.output_limit_chars // 2
        return text[:half] + "\n[... Observation truncated due to length ...]\n" + text[-half:], True


def _command_output(result: CommandResult) -> str:
    return "\n".join(value.rstrip() for value in (result.stdout, result.stderr) if value).strip() or "<no output>"
