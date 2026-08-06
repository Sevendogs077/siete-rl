"""容器内 OpenHands ACI editor 的最小本地实现。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any, Protocol

from siete_rl.docker import DockerSandbox

MAX_RESPONSE_LEN_CHAR = 16_000
SNIPPET_CONTEXT_WINDOW = 4
CONTENT_TRUNCATED_NOTICE = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"


class ToolError(RuntimeError):
    pass


class EditorToolParameterMissingError(ToolError):
    def __init__(self, command: str, parameter: str) -> None:
        super().__init__(f"Parameter `{parameter}` is required for command: {command}.")


class EditorToolParameterInvalidError(ToolError):
    def __init__(self, parameter: str, value: object, hint: str = "") -> None:
        super().__init__(f"Invalid `{parameter}` parameter: {value}. {hint}".rstrip())


class FileBackend(Protocol):
    def stat(self, path: str) -> str: ...
    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, text: str) -> None: ...
    def list_two_levels(self, path: str) -> list[str]: ...


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
    """所有 filesystem 访问均通过 rollout container，绝不读取宿主路径。"""
    def __init__(self, sandbox: DockerSandbox, *, timeout_sec: int) -> None:
        self.sandbox, self.timeout_sec = sandbox, timeout_sec

    def _run(self, op: str, path: str, *, input_text: str | None = None) -> str:
        result = self.sandbox.exec(["python", "-c", _BACKEND_PROGRAM, op, path], input_text=input_text, timeout_sec=self.timeout_sec)
        if result.timed_out:
            raise ToolError("container file operation timed out")
        if result.exit_code != 0:
            raise ToolError((result.stderr or result.stdout or "container file operation failed").strip())
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
    """五个 command 与 episode-local LIFO undo history。"""
    def __init__(self, backend: FileBackend) -> None:
        self.backend = backend
        self._history: dict[str, list[str]] = defaultdict(list)

    def __call__(self, *, command: str, path: str, file_text: str | None = None, view_range: list[int] | None = None, old_str: str | None = None, new_str: str | None = None, insert_line: int | None = None, **_: Any) -> str:
        self._validate_path(command, path)
        if command == "view": return self.view(path, view_range)
        if command == "create":
            if not file_text: raise EditorToolParameterMissingError(command, "file_text")
            self.backend.write_text(path, file_text); self._history[path].append(file_text)
            return f"File created successfully at: {path}"
        if command == "str_replace":
            if not old_str: raise EditorToolParameterMissingError(command, "old_str")
            return self.str_replace(path, old_str, new_str)
        if command == "insert":
            if insert_line is None: raise EditorToolParameterMissingError(command, "insert_line")
            if not new_str: raise EditorToolParameterMissingError(command, "new_str")
            return self.insert(path, insert_line, new_str)
        if command == "undo_edit": return self.undo_edit(path)
        raise ToolError("Unrecognized command " + command + ". The allowed commands for the oh_editor tool are: view, create, str_replace, insert, undo_edit")

    def _validate_path(self, command: str, path: str) -> None:
        if not PurePosixPath(path).is_absolute():
            raise EditorToolParameterInvalidError("path", path, "The path should be an absolute path, starting with `/`.")
        state = self.backend.stat(path)
        if command == "create" and state != "missing":
            raise EditorToolParameterInvalidError("path", path, f"File already exists at: {path}. Cannot overwrite files using command `create`.")
        if command != "create" and state == "missing":
            raise EditorToolParameterInvalidError("path", path, f"The path {path} does not exist. Please provide a valid path.")
        if command != "view" and state == "directory":
            raise EditorToolParameterInvalidError("path", path, f"The path {path} is a directory and only the `view` command can be used on directories.")

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        if self.backend.stat(path) == "directory":
            if view_range: raise EditorToolParameterInvalidError("view_range", view_range, "The `view_range` parameter is not allowed when `path` points to a directory.")
            values = "\n".join(self.backend.list_two_levels(path))
            return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{values}\n"
        text = self.backend.read_text(path); start = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(item, int) for item in view_range): raise EditorToolParameterInvalidError("view_range", view_range, "It should be a list of two integers.")
            lines = text.split("\n"); start, end = view_range
            if start < 1 or start > len(lines): raise EditorToolParameterInvalidError("view_range", view_range, f"Its first element `{start}` should be within the range of lines of the file: {[1, len(lines)]}.")
            if end != -1 and (end < start or end > len(lines)): raise EditorToolParameterInvalidError("view_range", view_range, "Its second element is outside the permitted range.")
            text = "\n".join(lines[start - 1 : None if end == -1 else end])
        return self._make_output(text, path, start)

    def str_replace(self, path: str, old_str: str, new_str: str | None) -> str:
        old_text = self.backend.read_text(path); old_str, new = old_str.expandtabs(), (new_str or "").expandtabs(); occurrences = old_text.count(old_str)
        if not occurrences: raise ToolError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.")
        if occurrences > 1: raise ToolError(f"No replacement was performed. Multiple occurrences of old_str `{old_str}`. Please ensure it is unique.")
        new_text = old_text.replace(old_str, new); self.backend.write_text(path, new_text); self._history[path].append(old_text)
        line = old_text.split(old_str)[0].count("\n"); snippet = "\n".join(new_text.split("\n")[max(0, line-SNIPPET_CONTEXT_WINDOW):line+SNIPPET_CONTEXT_WINDOW+new.count("\n")+1])
        return f"The file {path} has been edited. {self._make_output(snippet, f'a snippet of {path}', max(1, line-SNIPPET_CONTEXT_WINDOW+1))}Review the changes and make sure they are as expected. Edit the file again if necessary."

    def insert(self, path: str, insert_line: int, new_str: str) -> str:
        old_text = self.backend.read_text(path).expandtabs(); new_str = new_str.expandtabs(); lines = old_text.split("\n")
        if insert_line < 0 or insert_line > len(lines): raise EditorToolParameterInvalidError("insert_line", insert_line, f"It should be within the range of lines of the file: {[0, len(lines)]}")
        new_lines = lines[:insert_line] + new_str.split("\n") + lines[insert_line:]; new_text = "\n".join(new_lines); self.backend.write_text(path, new_text); self._history[path].append(old_text)
        snippet = "\n".join(new_lines[max(0, insert_line-SNIPPET_CONTEXT_WINDOW):insert_line+len(new_str.split("\n"))+SNIPPET_CONTEXT_WINDOW])
        return f"The file {path} has been edited. {self._make_output(snippet, 'a snippet of the edited file', max(1, insert_line-SNIPPET_CONTEXT_WINDOW+1))}Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

    def undo_edit(self, path: str) -> str:
        if not self._history[path]: raise ToolError(f"No edit history found for {path}.")
        text = self._history[path].pop(); self.backend.write_text(path, text)
        return f"Last edit to {path} undone successfully. {self._make_output(text, path)}"

    @staticmethod
    def _make_output(text: str, description: str, start_line: int = 1) -> str:
        if len(text) > MAX_RESPONSE_LEN_CHAR: text = text[:MAX_RESPONSE_LEN_CHAR] + CONTENT_TRUNCATED_NOTICE
        numbered = "\n".join(f"{index + start_line:6}\t{line}" for index, line in enumerate(text.expandtabs().split("\n")))
        return f"Here's the result of running `cat -n` on {description}:\n{numbered}\n"
