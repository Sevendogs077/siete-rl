from __future__ import annotations

from siete_rl.openhands_editor import OpenHandsEditor, ToolError


class Backend:
    def __init__(self) -> None:
        self.files = {"/repo/a.py": "one\ntwo\nthree"}; self.directories = {"/repo"}
    def stat(self, path): return "directory" if path in self.directories else "file" if path in self.files else "missing"
    def read_text(self, path): return self.files[path]
    def write_text(self, path, text): self.files[path] = text
    def list_two_levels(self, path): return ["/repo/a.py"]


def test_view_replace_insert_and_undo_are_episode_local() -> None:
    backend = Backend(); editor = OpenHandsEditor(backend)
    assert "one" in editor(command="view", path="/repo/a.py")
    assert "edited" in editor(command="str_replace", path="/repo/a.py", old_str="two", new_str="TWO")
    editor(command="insert", path="/repo/a.py", insert_line=1, new_str="middle")
    assert backend.files["/repo/a.py"] == "one\nmiddle\nTWO\nthree"
    editor(command="undo_edit", path="/repo/a.py")
    assert backend.files["/repo/a.py"] == "one\nTWO\nthree"


def test_editor_validates_absolute_paths_and_unique_replacement() -> None:
    editor = OpenHandsEditor(Backend())
    try: editor(command="view", path="a.py")
    except ToolError as exc: assert "absolute" in str(exc)
    else: raise AssertionError("relative path must fail")
    try: editor(command="str_replace", path="/repo/a.py", old_str="missing")
    except ToolError as exc: assert "did not appear" in str(exc)
    else: raise AssertionError("missing replacement must fail")
