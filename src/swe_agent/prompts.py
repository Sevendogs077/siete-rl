"""固定 SWE 任务的唯一 prompt builder。"""

from __future__ import annotations

from swe_agent.models import Task


SYSTEM_PROMPT = """You are a software repair agent working inside an isolated repository container.
Inspect the repository, diagnose the reported issue, make the smallest correct source change, run useful public checks when available, and submit a non-empty git diff.
Use the provided tools for all repository operations. Multiple tool calls in one assistant response are allowed and execute in order.
When a tool is needed, invoke the appropriate provided tool using exactly the tool-call format specified by the Tools instructions supplied with the conversation.
Do not describe, quote, simulate, or wrap a tool call in ordinary assistant content or a Markdown code fence.
Your first assistant response must call list_files, read_file, or search_code before making any edit.
Do not access hidden tests or verifier assets. Do not install dependencies, access the network, or invoke Docker.
When the patch is ready, call submit exactly once. A successful submit immediately ends the episode; do not call another tool afterwards.
"""


def build_prompt(task: Task) -> list[dict[str, str]]:
    """只从公开问题描述生成 TRL conversational prompt。"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.problem_statement},
    ]
