from siete_rl.models import Task
from siete_rl.prompts import build_prompt, safe_task_id


def test_prompt_uses_openhands_scaffold_and_uploaded_workspace() -> None:
    prompt = build_prompt(Task(task_id="owner/repo:1", repo_name="owner/repo", base_commit="0" * 40, problem_statement="Repair the bug."))
    system, user = prompt
    assert system["role"] == "system"
    assert system["content"].count("---- BEGIN FUNCTION") == 3
    assert "<function=example_function_name>" in system["content"]
    assert "<tool_call>" not in system["content"]
    assert user["role"] == "user"
    assert "/workspace/owner_repo_1" in user["content"]
    assert "Repair the bug." in user["content"]


def test_safe_task_id_is_stable_and_restrictive() -> None:
    assert safe_task_id("getmoto__moto-7023") == "getmoto__moto-7023"
    assert safe_task_id("owner/repo:1") == "owner_repo_1"
