from swe_agent.models import Task
from swe_agent.prompts import build_prompt


def test_prompt_requires_native_repository_inspection_without_textual_tool_requests() -> None:
    prompt = build_prompt(
        Task(
            task_id="task-1",
            repo_name="owner/repo",
            base_commit="0" * 40,
            problem_statement="Repair the bug.",
        )
    )

    system = prompt[0]["content"]
    assert "Use the provided tools for all repository operations" in system
    assert "tool-call format specified by the Tools instructions" in system
    assert "Do not describe, quote, simulate, or wrap a tool call" in system
    assert "first assistant response must call list_files, read_file, or search_code" in system
    assert "JSON object" not in system
    assert "XML text" not in system
    assert "<tool_call>" not in system
    assert prompt[1] == {"role": "user", "content": "Repair the bug."}
