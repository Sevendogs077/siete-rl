from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from swe_agent.config import load_config
from swe_agent.docker import DockerSandbox, SubprocessDockerClient, inspect_image
from swe_agent.models import Action
from swe_agent.swegym import load_task_instance
from swe_agent.tools import ToolExecutor, validate_tool_arguments
from swe_agent.verifier import SWEGymVerifier


pytestmark = pytest.mark.docker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"
TASK_ID = "getmoto__moto-7023"


@pytest.fixture(scope="module")
def domain():
    config, project_root, _ = load_config(CONFIG_PATH)
    sample, evaluation = load_task_instance(config, project_root, TASK_ID)
    return config, sample, evaluation


def sandbox_for(domain, scope: str) -> DockerSandbox:
    _, sample, _ = domain
    return DockerSandbox(
        client=SubprocessDockerClient(),
        task=sample.task,
        environment=sample.environment,
        run_id="docker-qualification",
        episode_id=f"{scope}-{uuid.uuid4().hex[:8]}",
        scope=scope,  # type: ignore[arg-type]
    )


def test_fixed_local_image_identity_and_base_evaluator(domain) -> None:
    _, sample, evaluation = domain
    metadata = inspect_image(SubprocessDockerClient(), sample.environment)
    assert metadata["image_id"] == sample.environment.expected_image_id
    assert metadata["os"] == "linux"
    assert metadata["architecture"] == "amd64"
    assert metadata["size_bytes"] == 2_849_787_451

    sandbox = sandbox_for(domain, "verifier")
    try:
        sandbox.open()
        evaluated = sandbox.exec(
            ["/bin/bash", "-s"],
            input_text=evaluation.offline_eval_script,
            timeout_sec=sample.environment.verifier_timeout_sec,
        )
        assert evaluated.timed_out is False
        assert evaluated.exit_code != 0
        assert "+ pytest" in evaluated.stdout + evaluated.stderr
    finally:
        sandbox.close()
    assert sandbox.container_id is None


def test_real_six_tool_executor_and_nonempty_submit(domain) -> None:
    sandbox = sandbox_for(domain, "rollout")
    try:
        sandbox.open()
        executor = ToolExecutor(sandbox, output_limit_chars=12_000, max_timeout_sec=300)
        actions = [
            Action(tool_name="list_files", arguments={"path": "moto/lakeformation", "max_entries": 20}),
            Action(
                tool_name="read_file",
                arguments={"path": "moto/lakeformation/models.py", "start_line": 50, "end_line": 65},
            ),
            Action(
                tool_name="search_code",
                arguments={"query": "deregister_resource", "path": "moto/lakeformation", "max_matches": 20},
            ),
            Action(tool_name="run_command", arguments={"command": "git status --short", "timeout_sec": 30}),
            Action(
                tool_name="edit_file",
                arguments={
                    "path": "moto/lakeformation/models.py",
                    "operation": "replace",
                    "old_text": "    def deregister_resource(self, resource_arn: str) -> None:\n        del self.resources[resource_arn]\n",
                    "new_text": "    def deregister_resource(self, resource_arn: str) -> None:\n        if resource_arn not in self.resources:\n            raise EntityNotFound\n        del self.resources[resource_arn]\n",
                },
            ),
            Action(tool_name="submit", arguments={}),
        ]
        observations = []
        for action in actions:
            validate_tool_arguments(action.tool_name, action.arguments)
            observations.append(executor.execute(action))
        assert all(observation.error_type is None for observation in observations)
        assert executor.submitted_patch is not None
        assert "raise EntityNotFound" in executor.submitted_patch
    finally:
        sandbox.close()
    assert sandbox.container_id is None


def test_gold_patch_resolves_in_fresh_verifier_container(domain) -> None:
    config, _, evaluation = domain
    patch = (Path(config.dataset.tasks_dir) / TASK_ID / "gold.patch").read_text(encoding="utf-8")
    verifier = SWEGymVerifier(
        sandbox_factory=lambda: sandbox_for(domain, "verifier"),
        evaluation=evaluation,
    )
    verified = verifier.verify(patch)
    assert verified.result == "resolved"
    assert verified.patch_apply_status == "applied"
    assert verified.pytest_started is True
    assert verifier.drain_cleanup_events()[0]["residual"] is False
