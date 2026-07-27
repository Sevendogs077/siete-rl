from __future__ import annotations

import inspect
from collections import deque

from swe_agent.docker import CommandResult
from swe_agent.environment import SWEEnvironment
from swe_agent.models import Environment, Evaluation, Sample, Task, Verification


class Sandbox:
    def __init__(self, sample, episode_id, scope) -> None:
        self.environment = sample.environment; self.episode_id = episode_id; self.scope = scope
        self.container_id = "a" * 64; self.container_name = "fake"; self.responses = deque(); self.diff = ""; self.closed = 0
    def open(self): return self
    def exec(self, command, *, input_text=None, timeout_sec=None):
        if command[:2] == ["python", "-c"] and len(command) == 4:
            return CommandResult(argv=[], exit_code=0, stdout="", stderr="", duration_sec=0, timed_out=False)
        return self.responses.popleft()
    def get_diff(self): return self.diff
    def close(self): self.closed += 1; self.container_id = None
    def drain_cleanup_operations(self): return []


class Verifier:
    _active_sandbox = None
    def __init__(self): self.calls = 0
    def verify(self, patch):
        self.calls += 1
        return Verification(result="resolved", patch_apply_status="applied", pytest_started=True, exit_code=0, stdout="", stderr="")
    def close(self): pass
    def drain_cleanup_events(self): return []


def harness():
    task = Task(task_id="owner/repo", repo_name="owner/repo", base_commit="0" * 40, problem_statement="fix")
    environment = Environment(environment_id="id", task_id=task.task_id, image_name="image", expected_image_id="sha256:" + "0" * 64, expected_registry_digest="sha256:" + "0" * 64, workdir="/testbed", cpus=1, memory="1g", pids_limit=1, exec_timeout_sec=1, verifier_timeout_sec=1)
    sandboxes = []; verifiers = []
    def make_sandbox(*args):
        value = Sandbox(*args); sandboxes.append(value); return value
    def make_verifier(*args):
        value = Verifier(); verifiers.append(value); return value
    return SWEEnvironment(task_context={task.task_id: (Sample(task=task, environment=environment), Evaluation(offline_eval_script="echo"))}, sandbox_factory=make_sandbox, verifier_factory=make_verifier, output_limit_chars=30000, max_timeout_sec=10), task.task_id, sandboxes, verifiers


def test_only_three_public_tools_and_reset_is_silent() -> None:
    env, task_id, sandboxes, _ = harness()
    methods = [name for name, member in inspect.getmembers(env, predicate=inspect.ismethod) if not name.startswith("_") and name != "reset"]
    assert methods == ["execute_bash", "finish", "str_replace_editor"]
    assert env.reset(task_id, prompt="ignored") is None
    assert sandboxes


def test_finish_freezes_empty_patch_without_verifier() -> None:
    env, task_id, _, verifiers = harness(); env.reset(task_id)
    assert env.finish() == ""
    assert env.terminated
    assert env.frozen_patch == ""
    assert env._finalize([]) == 0.0
    assert not verifiers


def test_finish_nonempty_patch_verifies_once() -> None:
    env, task_id, sandboxes, verifiers = harness(); env.reset(task_id); sandboxes[0].diff = "diff --git a/x b/x\n"
    env.finish()
    assert env._finalize([]) == 1.0
    assert env._finalize([]) == 1.0
    assert verifiers[0].calls == 1
