from __future__ import annotations

import pytest

from siete_rl.tools import OpenHandsEditor, ToolError


class Backend:
    def __init__(self) -> None:
        self.files = {
            "/repo/a.py": "one\ntwo\nthree",
            "/repo/repeated.txt": "same\nsame",
        }
        self.directories = {"/repo"}

    def stat(self, path: str) -> str:
        if path in self.directories:
            return "directory"
        if path in self.files:
            return "file"
        return "missing"

    def read_text(self, path: str) -> str:
        return self.files[path]

    def write_text(self, path: str, text: str) -> None:
        self.files[path] = text

    def list_two_levels(self, path: str) -> list[str]:
        assert path == "/repo"
        return sorted(self.files)


def test_view_renders_directories_and_requested_line_range() -> None:
    editor = OpenHandsEditor(Backend())

    directory = editor(command="view", path="/repo")
    ranged = editor(command="view", path="/repo/a.py", view_range=[2, -1])

    assert "/repo/a.py" in directory
    assert "     2\ttwo" in ranged
    assert "     3\tthree" in ranged
    assert "one" not in ranged


def test_edit_history_is_lifo_and_scoped_to_editor_instance() -> None:
    backend = Backend()
    first_episode = OpenHandsEditor(backend)
    second_episode = OpenHandsEditor(backend)

    first_episode(
        command="str_replace", path="/repo/a.py", old_str="two", new_str="TWO"
    )
    first_episode(
        command="insert", path="/repo/a.py", insert_line=1, new_str="middle"
    )
    assert backend.files["/repo/a.py"] == "one\nmiddle\nTWO\nthree"

    with pytest.raises(ToolError, match="No edit history"):
        second_episode(command="undo_edit", path="/repo/a.py")

    first_episode(command="undo_edit", path="/repo/a.py")
    assert backend.files["/repo/a.py"] == "one\nTWO\nthree"
    first_episode(command="undo_edit", path="/repo/a.py")
    assert backend.files["/repo/a.py"] == "one\ntwo\nthree"


def test_create_writes_a_previously_missing_file() -> None:
    backend = Backend()
    editor = OpenHandsEditor(backend)

    result = editor(command="create", path="/repo/new.py", file_text="value = 1\n")

    assert backend.files["/repo/new.py"] == "value = 1\n"
    assert "created successfully" in result


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"command": "view", "path": "a.py"}, "absolute path"),
        ({"command": "view", "path": "/repo/missing.py"}, "does not exist"),
        (
            {"command": "create", "path": "/repo/a.py", "file_text": "new"},
            "already exists",
        ),
        (
            {"command": "str_replace", "path": "/repo/a.py", "old_str": "missing"},
            "did not appear",
        ),
        (
            {
                "command": "str_replace",
                "path": "/repo/repeated.txt",
                "old_str": "same",
            },
            "Multiple occurrences",
        ),
        (
            {"command": "insert", "path": "/repo/a.py", "insert_line": 99, "new_str": "x"},
            "range of lines",
        ),
    ],
    ids=[
        "relative-path",
        "missing-path",
        "create-existing",
        "missing-replacement",
        "ambiguous-replacement",
        "insert-out-of-range",
    ],
)
def test_editor_rejects_invalid_operations(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ToolError, match=message):
        OpenHandsEditor(Backend())(**arguments)
