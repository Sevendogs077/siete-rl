"""固定六工具的领域校验与 Docker 执行器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from swe_agent.docker import CommandResult, DockerSandbox
from swe_agent.models import Action, Observation


class ToolContractError(ValueError):
    pass


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...] = ()

    @property
    def native_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolContractError("tool arguments must be an object")
        missing = set(self.required) - arguments.keys()
        unknown = arguments.keys() - self.properties.keys()
        if missing:
            raise ToolContractError("missing tool arguments: " + ", ".join(sorted(missing)))
        if unknown:
            raise ToolContractError("unknown tool arguments: " + ", ".join(sorted(unknown)))
        for name, value in arguments.items():
            schema = self.properties[name]
            expected = schema["type"]
            valid = (
                isinstance(value, str)
                if expected == "string"
                else isinstance(value, int) and not isinstance(value, bool)
            )
            if not valid:
                raise ToolContractError(f"{name} must have type {expected}")
            if expected == "string" and schema.get("minLength") == 1 and not value:
                raise ToolContractError(f"{name} must not be empty")
            if expected == "integer" and value < schema.get("minimum", value):
                raise ToolContractError(f"{name} is below the allowed minimum")
            if "enum" in schema and value not in schema["enum"]:
                raise ToolContractError(f"{name} is not an allowed value")
        if self.name == "edit_file":
            requirements = {
                "replace": {"path", "operation", "old_text", "new_text"},
                "insert": {"path", "operation", "line", "new_text"},
                "create": {"path", "operation", "content"},
            }
            operation = arguments.get("operation")
            if operation not in requirements:
                raise ToolContractError("operation is not an allowed value")
            expected_arguments = requirements[operation]
            missing_for_operation = expected_arguments - arguments.keys()
            incompatible = arguments.keys() - expected_arguments
            if missing_for_operation:
                raise ToolContractError(
                    "edit operation is missing arguments: "
                    + ", ".join(sorted(missing_for_operation))
                )
            if incompatible:
                raise ToolContractError(
                    "edit operation has incompatible arguments: "
                    + ", ".join(sorted(incompatible))
                )
        if (
            self.name == "read_file"
            and "start_line" in arguments
            and "end_line" in arguments
            and arguments["end_line"] < arguments["start_line"]
        ):
            raise ToolContractError("end_line must not be less than start_line")


def _string(description: str, *, allow_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if not allow_empty:
        schema["minLength"] = 1
    return schema


TOOL_SPECS = (
    ToolSpec(
        "list_files",
        "List repository files with bounded output.",
        {"path": _string("Repository-relative directory"), "max_entries": {"type": "integer", "minimum": 1}},
        ("path",),
    ),
    ToolSpec(
        "read_file",
        "Read a numbered window from a UTF-8 file.",
        {
            "path": _string("Repository-relative file"),
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        ("path",),
    ),
    ToolSpec(
        "search_code",
        "Search for exact text in repository files.",
        {
            "query": _string("Exact search text"),
            "path": _string("Repository-relative path"),
            "max_matches": {"type": "integer", "minimum": 1},
        },
        ("query",),
    ),
    ToolSpec(
        "edit_file",
        "Edit one repository file with replace, insert, or create.",
        {
            "path": _string("Repository-relative file"),
            "operation": {"type": "string", "enum": ["replace", "insert", "create"]},
            "old_text": _string("Text that must occur exactly once"),
            "new_text": _string("Replacement or inserted text", allow_empty=True),
            "content": _string("Complete new file content", allow_empty=True),
            "line": {"type": "integer", "minimum": 1},
        },
        ("path", "operation"),
    ),
    ToolSpec(
        "run_command",
        "Run a diagnostic or public test command inside /testbed.",
        {"command": _string("Shell command"), "timeout_sec": {"type": "integer", "minimum": 1}},
        ("command",),
    ),
    ToolSpec("submit", "Submit the current non-empty git diff.", {}),
)
BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def native_tool_schemas() -> list[dict[str, Any]]:
    return [spec.native_schema for spec in TOOL_SPECS]


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    try:
        spec = BY_NAME[name]
    except KeyError as exc:
        raise ToolContractError(f"unsupported tool: {name}") from exc
    spec.validate(arguments)


_FILE_PROGRAM = r"""
import pathlib, sys
root = pathlib.Path('/testbed').resolve()
operation, relative = sys.argv[1], sys.argv[2]
path = (root / relative).resolve()
if path != root and root not in path.parents:
    raise SystemExit('path escapes /testbed')
if '.git' in path.parts:
    raise SystemExit('access to .git is forbidden')
if operation == 'list':
    limit = int(sys.argv[3]); values = []
    if not path.is_dir(): raise SystemExit('directory does not exist')
    for item in sorted(path.rglob('*')):
        if item.is_file() and '.git' not in item.relative_to(root).parts:
            values.append(item.relative_to(root).as_posix())
    print('\n'.join(values[:limit]) if values else '<empty>')
    if len(values) > limit: print(f'<truncated: {limit}/{len(values)}>')
elif operation == 'read':
    start, end = int(sys.argv[3]), int(sys.argv[4])
    lines = path.read_text(encoding='utf-8').splitlines()
    for number in range(start, min(end, len(lines)) + 1): print(f'{number}: {lines[number-1]}')
elif operation == 'search':
    query, limit = sys.argv[3], int(sys.argv[4]); count = 0
    targets = [path] if path.is_file() else sorted(path.rglob('*'))
    for item in targets:
        if not item.is_file() or '.git' in item.relative_to(root).parts: continue
        try: lines = item.read_text(encoding='utf-8').splitlines()
        except (UnicodeDecodeError, OSError): continue
        for number, line in enumerate(lines, 1):
            if query in line:
                print(f'{item.relative_to(root).as_posix()}:{number}:{line}')
                count += 1
                if count >= limit: raise SystemExit(0)
elif operation == 'edit':
    edit_operation = sys.argv[3]
    if edit_operation == 'replace':
        old, new = sys.argv[4], sys.argv[5]
        text = path.read_text(encoding='utf-8')
        if text.count(old) != 1: raise SystemExit('old_text must match exactly once')
        path.write_text(text.replace(old, new, 1), encoding='utf-8')
    elif edit_operation == 'insert':
        line, new = int(sys.argv[4]), sys.argv[5]
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        if line > len(lines) + 1: raise SystemExit('insertion line is beyond the file')
        lines.insert(line - 1, new if new.endswith('\n') else new + '\n')
        path.write_text(''.join(lines), encoding='utf-8')
    elif edit_operation == 'create':
        if path.exists(): raise SystemExit('create requires the file not to exist')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sys.argv[4], encoding='utf-8')
"""


class ToolExecutor:
    def __init__(
        self, sandbox: DockerSandbox, *, output_limit_chars: int, max_timeout_sec: int
    ) -> None:
        self.sandbox = sandbox
        self.output_limit_chars = output_limit_chars
        self.max_timeout_sec = max_timeout_sec
        self.submitted_patch: str | None = None
        self.handlers: dict[str, Callable[[dict[str, Any]], tuple[int, str, bool]]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_code": self._search_code,
            "edit_file": self._edit_file,
            "run_command": self._run_command,
            "submit": self._submit,
        }

    def execute(self, action: Action) -> Observation:
        try:
            handler = self.handlers[action.tool_name]
            exit_code, text, timed_out = handler(action.arguments)
            error_type = None if exit_code == 0 else "tool_error"
        except (ToolError, KeyError) as exc:
            exit_code, text, timed_out, error_type = 1, str(exc), False, "tool_error"
        bounded, truncated = self._bounded(text)
        return Observation(
            text=bounded,
            exit_code=exit_code,
            error_type=error_type,
            timed_out=timed_out,
            truncated=truncated,
        )

    def _list_files(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        return self._python(["list", str(args["path"]), str(args.get("max_entries", 200))])

    def _read_file(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        start = int(args.get("start_line", 1))
        end = int(args.get("end_line", start + 199))
        return self._python(["read", str(args["path"]), str(start), str(end)])

    def _search_code(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        return self._python(
            ["search", str(args.get("path", ".")), str(args["query"]), str(args.get("max_matches", 50))]
        )

    def _edit_file(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        operation = str(args["operation"])
        values = ["edit", str(args["path"]), operation]
        if operation == "replace":
            values.extend([str(args["old_text"]), str(args["new_text"])])
        elif operation == "insert":
            values.extend([str(args["line"]), str(args["new_text"])])
        else:
            values.append(str(args["content"]))
        exit_code, output, timed_out = self._python(values)
        if exit_code == 0 and not timed_out:
            diff = self.sandbox.get_diff()
            if not diff.strip():
                raise ToolError("edit_file did not produce a git diff")
            output = "File edited successfully.\n\n" + diff
        return exit_code, output, timed_out

    def _run_command(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        command = str(args["command"]).strip()
        _enforce_command_policy(command)
        timeout = min(int(args.get("timeout_sec", self.max_timeout_sec)), self.max_timeout_sec)
        shell = (
            "source /opt/miniconda3/bin/activate && conda activate testbed "
            "&& cd /testbed && "
            + command
        )
        result = self.sandbox.exec(["/bin/bash", "-lc", shell], timeout_sec=timeout)
        return result.exit_code, _command_output(result), result.timed_out

    def _submit(self, args: dict[str, Any]) -> tuple[int, str, bool]:
        if args:
            raise ToolError("submit does not accept arguments")
        patch = self.sandbox.get_diff()
        if not patch.strip():
            raise ToolError("an empty patch cannot be submitted")
        self.submitted_patch = patch
        return 0, "Current git diff submitted.", False

    def _python(self, arguments: list[str]) -> tuple[int, str, bool]:
        result = self.sandbox.exec(["python", "-c", _FILE_PROGRAM, *arguments])
        return result.exit_code, _command_output(result), result.timed_out

    def _bounded(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.output_limit_chars:
            return value, False
        return (
            value[: self.output_limit_chars] + f"\n<truncated: {len(value)} chars>",
            True,
        )


def _enforce_command_policy(command: str) -> None:
    denied = (
        r"(^|[;&|]\s*)docker(?:\s|$)",
        r"(^|[;&|]\s*)podman(?:\s|$)",
        r"(^|[;&|]\s*)(?:sudo\s+)?apt(?:-get)?(?:\s|$)",
        r"(?:^|\s)(?:python\s+-m\s+)?pip\s+install(?:\s|$)",
        r"(?:^|\s)(?:rm|mv|cp|sed)\s+[^\n]*(?:/etc/|/var/run/docker\.sock)",
    )
    if any(re.search(pattern, command) for pattern in denied):
        raise ToolError("run_command violates the fixed execution boundary")


def _command_output(result: CommandResult) -> str:
    output = "\n".join(
        value.rstrip() for value in (result.stdout, result.stderr) if value
    ).strip()
    return output or "<no output>"
