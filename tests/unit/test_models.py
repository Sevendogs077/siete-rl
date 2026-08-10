from __future__ import annotations

import pytest
from pydantic import ValidationError

from siete_rl.models import (
    Action,
    Environment,
    Evaluation,
    Observation,
    Sample,
    Settlement,
    Step,
    Task,
    TerminalEvent,
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
    TerminalEvent: {"kind", "step_index"},
    Settlement: {"status", "detail"},
    Trajectory: {"task_id", "environment_id", "steps", "termination", "settlement"},
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


def test_persisted_domain_models_have_expected_fields() -> None:
    for model, expected in EXPECTED_FIELDS.items():
        assert set(model.model_fields) == expected


def test_sample_rejects_task_environment_mismatch(task: Task, environment: Environment) -> None:
    assert Sample(task=task, environment=environment).task is task
    with pytest.raises(ValidationError, match="Environment.task_id"):
        Sample(task=task, environment=environment.model_copy(update={"task_id": "other"}))


def test_trajectory_serialization_excludes_transport_and_training_state(
    task: Task, environment: Environment
) -> None:
    action = Action(
        tool_name="str_replace_editor",
        arguments={"command": "view", "path": "/repo/moto/api.py"},
    )
    observation = Observation(text="contents", exit_code=0)
    steps = [Step(index=0, action=action, observation=observation)]
    trajectory = Trajectory(
        task_id=task.task_id,
        environment_id=environment.environment_id,
        steps=steps,
        termination="format_exhausted",
        settlement=Settlement(status="unresolved"),
    )
    dumped = trajectory.model_dump(mode="json")
    assert dumped["steps"][0]["action"] == {
        "tool_name": "str_replace_editor",
        "arguments": {"command": "view", "path": "/repo/moto/api.py"},
    }
    assert not ({"tool_call_id", "logprobs", "messages", "patch", "reward"} & set(dumped))


def test_trajectory_requires_contiguous_step_indexes(
    task: Task, environment: Environment
) -> None:
    action = Action(tool_name="execute_bash", arguments={"command": "pytest -q"})
    observation = Observation(text="contents", exit_code=0)
    with pytest.raises(ValidationError, match="contiguous"):
        Trajectory(
            task_id=task.task_id,
            environment_id=environment.environment_id,
            steps=[Step(index=1, action=action, observation=observation)],
            termination="submitted",
            settlement=Settlement(status="unresolved"),
        )


def test_trajectory_records_orthogonal_termination_and_settlement() -> None:
    trajectory = Trajectory(
        task_id="task",
        environment_id="env",
        steps=[],
        termination="context_overlong",
        settlement=Settlement(status="unresolved"),
    )

    assert trajectory.termination == "context_overlong"
    assert trajectory.settlement == Settlement(status="unresolved", detail=None)


def test_settlement_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        Settlement(status="timeout")


@pytest.mark.parametrize(
    ("result", "patch_apply_status", "pytest_started", "exit_code"),
    [
        ("resolved", "check_failed", True, 0),
        ("resolved", "applied", False, 0),
        ("resolved", "applied", True, 1),
        ("unresolved", "applied", False, 1),
    ],
)
def test_verification_rejects_results_without_attributable_evidence(
    result: str, patch_apply_status: str, pytest_started: bool, exit_code: int
) -> None:
    with pytest.raises(ValidationError, match="pytest evidence|real pytest evidence"):
        Verification(
            result=result,
            patch_apply_status=patch_apply_status,
            pytest_started=pytest_started,
            exit_code=exit_code,
            stdout="failed",
            stderr="",
        )


def test_domain_models_reject_unknown_fields(task: Task) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Task.model_validate({**task.model_dump(), "metadata": {}})
