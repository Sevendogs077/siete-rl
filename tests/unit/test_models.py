from __future__ import annotations

import pytest
from pydantic import ValidationError

from siete_rl.models import (
    Action,
    Environment,
    Evaluation,
    Observation,
    Sample,
    Step,
    Task,
    Trajectory,
    Verification,
)


EXPECTED_FIELDS = {
    Task: {"task_id", "repo_name", "base_commit", "problem_statement"},
    Environment: {
        "environment_id",
        "task_id",
        "image_name",
        "expected_image_id",
        "expected_registry_digest",
        "workdir",
        "cpus",
        "memory",
        "pids_limit",
        "exec_timeout_sec",
        "verifier_timeout_sec",
    },
    Evaluation: {"offline_eval_script", "fail_to_pass", "pass_to_pass"},
    Sample: {"task", "environment"},
    Action: {"tool_name", "arguments"},
    Observation: {"text", "exit_code", "error_type", "timed_out", "truncated"},
    Step: {"index", "action", "observation"},
    Trajectory: {"task_id", "environment_id", "steps", "termination"},
    Verification: {"result", "patch_apply_status", "pytest_started", "exit_code", "stdout", "stderr"},
}


@pytest.fixture
def task() -> Task:
    return Task(
        task_id="getmoto__moto-7023",
        repo_name="getmoto/moto",
        base_commit="447710c6a68e7d5ea7ad6d7df93c663de32ac7f1",
        problem_statement="Fix the bug.",
    )


@pytest.fixture
def environment() -> Environment:
    return Environment(
        environment_id="swegym:getmoto__moto-7023",
        task_id="getmoto__moto-7023",
        image_name="image",
        expected_image_id="sha256:" + "1" * 64,
        expected_registry_digest="sha256:" + "2" * 64,
        workdir="/testbed",
        cpus=4,
        memory="16g",
        pids_limit=512,
        exec_timeout_sec=300,
        verifier_timeout_sec=3600,
    )


def test_nine_core_models_have_exact_planned_fields() -> None:
    assert len(EXPECTED_FIELDS) == 9
    for model, expected in EXPECTED_FIELDS.items():
        assert set(model.model_fields) == expected
        assert "metadata" not in model.model_fields


def test_sample_only_validates_task_environment_pair(task: Task, environment: Environment) -> None:
    sample = Sample(task=task, environment=environment)
    assert sample.task is task
    with pytest.raises(ValidationError, match="Environment.task_id"):
        Sample(task=task, environment=environment.model_copy(update={"task_id": "other"}))


def test_action_step_and_trajectory_have_no_wire_or_probability_fields(
    task: Task, environment: Environment
) -> None:
    action = Action(tool_name="read_file", arguments={"path": "moto/api.py"})
    observation = Observation(text="contents", exit_code=0)
    steps = [Step(index=0, action=action, observation=observation)]
    trajectory = Trajectory(
        task_id=task.task_id,
        environment_id=environment.environment_id,
        steps=steps,
        termination="format_exhausted",
    )
    dumped = trajectory.model_dump(mode="json")
    assert dumped["steps"][0]["action"] == {
        "tool_name": "read_file",
        "arguments": {"path": "moto/api.py"},
    }
    assert not ({"tool_call_id", "logprobs", "messages", "patch", "reward"} & set(dumped))
    with pytest.raises(ValidationError, match="contiguous"):
        Trajectory(
            task_id=task.task_id,
            environment_id=environment.environment_id,
            steps=[Step(index=1, action=action, observation=observation)],
            termination="submitted",
        )


def test_verification_requires_real_attributable_evidence() -> None:
    Verification(
        result="resolved",
        patch_apply_status="applied",
        pytest_started=True,
        exit_code=0,
        stdout="passed",
        stderr="",
    )
    Verification(
        result="unresolved",
        patch_apply_status="check_failed",
        pytest_started=False,
        exit_code=1,
        stdout="",
        stderr="does not apply",
    )
    with pytest.raises(ValidationError, match="successful pytest"):
        Verification(
            result="resolved",
            patch_apply_status="applied",
            pytest_started=True,
            exit_code=1,
            stdout="failed",
            stderr="",
        )


def test_all_models_reject_extra_fields(task: Task) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Task.model_validate({**task.model_dump(), "metadata": {}})
