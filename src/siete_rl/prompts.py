"""OpenHands SWE-Gym prompt builder。"""

from __future__ import annotations

import re

from siete_rl.models import Task
from siete_rl.tool_protocol import render_system_suffix


# 该 base 与训练轨迹的固定 system message 相同；工具 blocks 由本地 schema 追加。
SYSTEM_PROMPT_BASE = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>
"""


def safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)


def build_prompt(task: Task) -> list[dict[str, str]]:
    """渲染 checkpoint 所见的 system scaffold 与 SWE 问题消息。"""
    workspace = safe_task_id(task.task_id)
    issue = (
        "<uploaded_files>\n"
        f"/workspace/{workspace}\n"
        "</uploaded_files>\n"
        f"I've uploaded a python code repository in the directory {workspace}. Consider the following PR description:\n\n"
        "<pr_description>\n"
        f"{task.problem_statement}\n"
        "</pr_description>\n\n"
        "Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n"
        "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        "Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE + render_system_suffix()},
        {"role": "user", "content": issue},
    ]
