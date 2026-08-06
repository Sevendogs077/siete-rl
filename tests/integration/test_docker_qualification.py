from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from siete_rl.config import load_config
from siete_rl.docker import DockerSandbox, SubprocessDockerClient, inspect_image
from siete_rl.environment import SWEEnvironment
from siete_rl.models import Environment, Evaluation, Sample, Task
from siete_rl.swegym import load_task_instance


pytestmark = pytest.mark.docker
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"
TASK_ID = "getmoto__moto-7023"


@pytest.fixture(scope="module")
def domain():
    config, root, _ = load_config(CONFIG_PATH)
    return config, load_task_instance(config, root, TASK_ID)


def test_fixed_local_image_identity(domain) -> None:
    _, (sample, _) = domain
    metadata = inspect_image(SubprocessDockerClient(), sample.environment)
    assert metadata["image_id"] == sample.environment.expected_image_id
    assert metadata["os"] == "linux" and metadata["architecture"] == "amd64"


def test_real_openhands_three_tool_episode(domain) -> None:
    config, (sample, evaluation) = domain
    sandboxes = []
    def make_sandbox(sample, episode_id, scope):
        value = DockerSandbox(client=SubprocessDockerClient(), task=sample.task, environment=sample.environment, run_id="docker-qualification", episode_id=f"{scope}-{uuid.uuid4().hex[:8]}", scope=scope)
        sandboxes.append(value); return value
    env = SWEEnvironment(task_context={TASK_ID: (sample, evaluation)}, sandbox_factory=make_sandbox, verifier_factory=lambda *args: None, output_limit_chars=30_000, max_timeout_sec=config.docker.exec_timeout_sec)
    try:
        assert env.reset(TASK_ID) is None
        aliases = env.execute_bash("readlink /repo; readlink /workspace/getmoto__moto-7023; pwd")
        assert aliases.count("/testbed") >= 2 and "/workspace/getmoto__moto-7023" in aliases
        viewed = env.str_replace_editor("view", "/repo/moto/lakeformation/models.py")
        assert "deregister_resource" in viewed
        edited = env.str_replace_editor("str_replace", "/repo/moto/lakeformation/models.py", old_str="        del self.resources[resource_arn]", new_str="        if resource_arn not in self.resources:\n            raise EntityNotFound\n        del self.resources[resource_arn]")
        assert "edited" in edited
        assert env.finish() == ""
        assert env.frozen_patch and "raise EntityNotFound" in env.frozen_patch
    finally:
        env._close()
    assert all(sandbox.container_id is None for sandbox in sandboxes)
