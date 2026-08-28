from __future__ import annotations

import json
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from siete_rl.docker import CommandResult, DockerSandbox
from siete_rl.models import Action, Observation
from siete_rl.tool_protocol import OPENHANDS_TOOL_SCHEMAS


MAX_RESPONSE_LEN_CHAR = 16_000
SNIPPET_CONTEXT_WINDOW = 4
CONTENT_TRUNCATED_NOTICE = (
    "<response clipped><NOTE>To save on context only part of this file has been "
    "shown to you. You should retry this tool after you have searched inside "
    "the file with `grep -n` in order to find the line numbers of what you are "
    "looking for.</NOTE>"
)


class ToolError(RuntimeError):
    pass


class EditorToolParameterMissingError(ToolError):
    def __init__(self, command: str, parameter: str) -> None:
        super().__init__(f"Parameter `{parameter}` is required for command: {command}.")


class EditorToolParameterInvalidError(ToolError):
    def __init__(self, parameter: str, value: object, hint: str = "") -> None:
        super().__init__(f"Invalid `{parameter}` parameter: {value}. {hint}".rstrip())


_BACKEND_PROGRAM = r'''import json, os, pathlib, sys
op, path = sys.argv[1:3]
p = pathlib.Path(path)
if op == "stat":
    print("directory" if p.is_dir() else "file" if p.is_file() else "missing")
elif op == "read":
    print(p.read_text(encoding="utf-8"), end="")
elif op == "write":
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sys.stdin.read(), encoding="utf-8")
elif op == "list":
    if not p.is_dir(): raise SystemExit("not a directory")
    values = []
    for item in sorted(p.rglob("*")):
        relative = item.relative_to(p)
        if len(relative.parts) > 2 or any(part.startswith(".") for part in relative.parts): continue
        values.append(item.as_posix())
    print(json.dumps(values))
else: raise SystemExit("unknown operation")
'''


class ContainerFileBackend:
    def __init__(self, sandbox: DockerSandbox, *, timeout_sec: int) -> None:
        self.sandbox = sandbox
        self.timeout_sec = timeout_sec

    def _run(self, op: str, path: str, *, input_text: str | None = None) -> str:
        result = self.sandbox.exec(
            ["python", "-c", _BACKEND_PROGRAM, op, path],
            input_text=input_text,
            timeout_sec=self.timeout_sec,
        )
        if result.timed_out:
            raise ToolError("container file operation timed out")
        if result.exit_code != 0:
            raise ToolError(
                (
                    result.stderr
                    or result.stdout
                    or "container file operation failed"
                ).strip()
            )
        return result.stdout

    def stat(self, path: str) -> str:
        return self._run("stat", path).strip()

    def read_text(self, path: str) -> str:
        return self._run("read", path)

    def write_text(self, path: str, text: str) -> None:
        self._run("write", path, input_text=text)

    def list_two_levels(self, path: str) -> list[str]:
        try:
            return json.loads(self._run("list", path))
        except json.JSONDecodeError as exc:
            raise ToolError("container directory listing was invalid") from exc


class OpenHandsEditor:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._history: dict[str, list[str]] = defaultdict(list)

    def __call__(
        self,
        *,
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **_: Any,
    ) -> str:
        self._validate_path(command, path)
        if command == "view":
            return self.view(path, view_range)
        if command == "create":
            if not file_text:
                raise EditorToolParameterMissingError(command, "file_text")
            self.backend.write_text(path, file_text)
            self._history[path].append(file_text)
            return f"File created successfully at: {path}"
        if command == "str_replace":
            if not old_str:
                raise EditorToolParameterMissingError(command, "old_str")
            return self.str_replace(path, old_str, new_str)
        if command == "insert":
            if insert_line is None:
                raise EditorToolParameterMissingError(command, "insert_line")
            if not new_str:
                raise EditorToolParameterMissingError(command, "new_str")
            return self.insert(path, insert_line, new_str)
        if command == "undo_edit":
            return self.undo_edit(path)
        raise ToolError(
            "Unrecognized command "
            + command
            + ". The allowed commands for the oh_editor tool are: "
            "view, create, str_replace, insert, undo_edit"
        )

    def _validate_path(self, command: str, path: str) -> None:
        if not PurePosixPath(path).is_absolute():
            raise EditorToolParameterInvalidError(
                "path",
                path,
                "The path should be an absolute path, starting with `/`.",
            )
        state = self.backend.stat(path)
        if command == "create" and state != "missing":
            raise EditorToolParameterInvalidError(
                "path",
                path,
                f"File already exists at: {path}. "
                "Cannot overwrite files using command `create`.",
            )
        if command != "create" and state == "missing":
            raise EditorToolParameterInvalidError(
                "path", path, f"The path {path} does not exist. Please provide a valid path."
            )
        if command != "view" and state == "directory":
            raise EditorToolParameterInvalidError(
                "path",
                path,
                f"The path {path} is a directory and only the `view` command "
                "can be used on directories.",
            )

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        if self.backend.stat(path) == "directory":
            if view_range:
                raise EditorToolParameterInvalidError(
                    "view_range",
                    view_range,
                    "The `view_range` parameter is not allowed when `path` "
                    "points to a directory.",
                )
            values = "\n".join(self.backend.list_two_levels(path))
            return (
                "Here's the files and directories up to 2 levels deep in "
                f"{path}, excluding hidden items:\n{values}\n"
            )
        text = self.backend.read_text(path)
        start = 1
        if view_range:
            if len(view_range) != 2 or not all(
                isinstance(item, int) for item in view_range
            ):
                raise EditorToolParameterInvalidError(
                    "view_range", view_range, "It should be a list of two integers."
                )
            lines = text.split("\n")
            start, end = view_range
            if start < 1 or start > len(lines):
                raise EditorToolParameterInvalidError(
                    "view_range",
                    view_range,
                    f"Its first element `{start}` should be within the range "
                    f"of lines of the file: {[1, len(lines)]}.",
                )
            if end != -1 and (end < start or end > len(lines)):
                raise EditorToolParameterInvalidError(
                    "view_range",
                    view_range,
                    "Its second element is outside the permitted range.",
                )
            text = "\n".join(lines[start - 1 : None if end == -1 else end])
        return self._make_output(text, path, start)

    def str_replace(self, path: str, old_str: str, new_str: str | None) -> str:
        old_text = self.backend.read_text(path)
        old_str = old_str.expandtabs()
        new = (new_str or "").expandtabs()
        occurrences = old_text.count(old_str)
        if not occurrences:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` "
                f"did not appear verbatim in {path}."
            )
        if occurrences > 1:
            raise ToolError(
                "No replacement was performed. Multiple occurrences of old_str "
                f"`{old_str}`. Please ensure it is unique."
            )
        new_text = old_text.replace(old_str, new)
        self.backend.write_text(path, new_text)
        self._history[path].append(old_text)
        line = old_text.split(old_str)[0].count("\n")
        snippet = "\n".join(
            new_text.split("\n")[
                max(0, line - SNIPPET_CONTEXT_WINDOW) :
                line + SNIPPET_CONTEXT_WINDOW + new.count("\n") + 1
            ]
        )
        output = self._make_output(
            snippet,
            f"a snippet of {path}",
            max(1, line - SNIPPET_CONTEXT_WINDOW + 1),
        )
        return (
            f"The file {path} has been edited. {output}"
            "Review the changes and make sure they are as expected. "
            "Edit the file again if necessary."
        )

    def insert(self, path: str, insert_line: int, new_str: str) -> str:
        old_text = self.backend.read_text(path).expandtabs()
        new_str = new_str.expandtabs()
        lines = old_text.split("\n")
        if insert_line < 0 or insert_line > len(lines):
            raise EditorToolParameterInvalidError(
                "insert_line",
                insert_line,
                f"It should be within the range of lines of the file: {[0, len(lines)]}",
            )
        inserted = new_str.split("\n")
        new_lines = lines[:insert_line] + inserted + lines[insert_line:]
        new_text = "\n".join(new_lines)
        self.backend.write_text(path, new_text)
        self._history[path].append(old_text)
        snippet = "\n".join(
            new_lines[
                max(0, insert_line - SNIPPET_CONTEXT_WINDOW) :
                insert_line + len(inserted) + SNIPPET_CONTEXT_WINDOW
            ]
        )
        output = self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_CONTEXT_WINDOW + 1),
        )
        return (
            f"The file {path} has been edited. {output}"
            "Review the changes and make sure they are as expected "
            "(correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )

    def undo_edit(self, path: str) -> str:
        if not self._history[path]:
            raise ToolError(f"No edit history found for {path}.")
        text = self._history[path].pop()
        self.backend.write_text(path, text)
        return (
            f"Last edit to {path} undone successfully. "
            f"{self._make_output(text, path)}"
        )

    @staticmethod
    def _make_output(text: str, description: str, start_line: int = 1) -> str:
        if len(text) > MAX_RESPONSE_LEN_CHAR:
            text = text[:MAX_RESPONSE_LEN_CHAR] + CONTENT_TRUNCATED_NOTICE
        numbered = "\n".join(
            f"{index + start_line:6}\t{line}"
            for index, line in enumerate(text.expandtabs().split("\n"))
        )
        return f"Here's the result of running `cat -n` on {description}:\n{numbered}\n"


class ToolContractError(ValueError):
    pass


TOOL_PARAMETERS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in OPENHANDS_TOOL_SCHEMAS
}


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    if name not in TOOL_PARAMETERS:
        raise ToolContractError(f"unsupported tool: {name}")
    if not isinstance(arguments, dict):
        raise ToolContractError("tool arguments must be an object")
    parameters = TOOL_PARAMETERS[name]
    properties = parameters["properties"]
    missing = set(parameters.get("required", [])) - arguments.keys()
    unknown = arguments.keys() - properties.keys()
    if missing:
        raise ToolContractError(
            "missing tool arguments: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ToolContractError(
            "unknown tool arguments: " + ", ".join(sorted(unknown))
        )
    for key, value in arguments.items():
        schema = properties[key]
        expected = schema["type"]
        if expected == "string":
            valid = isinstance(value, str)
        elif expected == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected == "array":
            valid = isinstance(value, list)
        else:
            valid = False
        if not valid:
            raise ToolContractError(f"{key} must have type {expected}")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolContractError(f"{key} is not an allowed value")


class ToolExecutor:
    def __init__(
        self,
        sandbox: DockerSandbox,
        *,
        output_limit_chars: int,
        max_timeout_sec: int,
        workspace: str,
    ) -> None:
        self.sandbox = sandbox
        self.output_limit_chars = output_limit_chars
        self.max_timeout_sec = max_timeout_sec
        self.workspace = workspace
        self.editor = OpenHandsEditor(
            ContainerFileBackend(sandbox, timeout_sec=max_timeout_sec)
        )

    def execute(self, action: Action) -> Observation:
        try:
            if action.tool_name == "execute_bash":
                code, text, timed_out = self._execute_bash(action.arguments)
            elif action.tool_name == "str_replace_editor":
                code, text, timed_out = 0, self.editor(**action.arguments), False
            elif action.tool_name == "finish":
                code, text, timed_out = self._finish(action.arguments)
            else:
                raise ToolError(f"unsupported tool: {action.tool_name}")
            error_type = None if code == 0 else "tool_error"
        except (ToolError, ValueError) as exc:
            code, text, timed_out, error_type = 1, str(exc), False, "tool_error"
        text, truncated = self._bounded(text)
        return Observation(
            text=text,
            exit_code=code,
            error_type=error_type,
            timed_out=timed_out,
            truncated=truncated,
        )

    def _execute_bash(self, arguments: dict[str, Any]) -> tuple[int, str, bool]:
        command = arguments["command"]
        result = self.sandbox.exec(
            ["/bin/bash", "-lc", f"cd {self.workspace} && {command}"],
            timeout_sec=self.max_timeout_sec,
        )
        output = (
            _command_output(result)
            + f"\n[Command finished with exit code {result.exit_code}]"
        )
        return result.exit_code, output, result.timed_out

    def _finish(self, arguments: dict[str, Any]) -> tuple[int, str, bool]:
        if arguments:
            raise ToolError("finish does not accept arguments")
        return 0, "", False

    def _bounded(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.output_limit_chars:
            return text, False
        half = self.output_limit_chars // 2
        return (
            text[:half]
            + "\n[... Observation truncated due to length ...]\n"
            + text[-half:],
            True,
        )


def _command_output(result: CommandResult) -> str:
    return (
        "\n".join(
            value.rstrip() for value in (result.stdout, result.stderr) if value
        ).strip()
        or "<no output>"
    )
