from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from siete_rl.config import load_config
from siete_rl.models import (
    Action,
    Observation,
    Settlement,
    Step,
    Trajectory,
    Verification,
)
from siete_rl.recording import RunRecorder, generate_run_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"


def configured_for(tmp_path: Path, *, run_id: str | None = None):
    config, _, _ = load_config(CONFIG_PATH)
    output = config.output.model_copy(
        update={"output_root": (tmp_path / "outputs").as_posix(), "run_id": run_id}
    )
    return config.model_copy(update={"output": output})


def verification(result: str = "resolved") -> Verification:
    return Verification(
        result=result,
        patch_apply_status="applied",
        pytest_started=True,
        exit_code=0 if result == "resolved" else 1,
        stdout="+ pytest\n",
        stderr="",
    )


def trajectory(
    termination: str = "submitted", settlement: str = "resolved"
) -> Trajectory:
    return Trajectory.model_validate(
        {
            "task_id": "getmoto__moto-7023",
            "environment_id": "getmoto__moto-7023",
            "steps": [
                Step(
                    index=0,
                    action=Action(
                        tool_name="execute_bash", arguments={"command": "sed -n '1,40p' README.md"}
                    ),
                    observation=Observation(text="1: hello", exit_code=0),
                )
            ],
            "termination": termination,
            "settlement": {"status": settlement},
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_run_id_and_initial_files_have_exact_contract(tmp_path: Path) -> None:
    generated = generate_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{4}", generated)

    config = configured_for(tmp_path, run_id="fixed-run")
    recorder = RunRecorder(
        config=config,
        seed=123,
        dependency_versions={"trl": "1.8.0"},
        code_commit="abc123",
        code_dirty=False,
        model_revision="Revision:master",
    )
    assert recorder.output_dir == (tmp_path / "outputs/fixed-run").resolve()
    assert {path.name for path in recorder.output_dir.iterdir()} == {
        "config.yaml",
        "run.json",
        "cleanup.json",
        "metrics.jsonl",
        "train.log",
    }
    assert yaml.safe_load(
        (recorder.output_dir / "config.yaml").read_text()
    ) == config.model_dump(mode="json")

    run = load_json(recorder.output_dir / "run.json")
    assert set(run) == {
        "run_id",
        "status",
        "failure",
        "results",
        "train",
        "artifacts",
        "time",
        "config",
        "cleanup",
        "provenance",
    }
    assert run["run_id"] == "fixed-run"
    assert run["status"] == "running"
    assert run["failure"] is None
    assert run["results"] == {
        "reward": {
            "successes": 0,
            "attempts": 0,
            "mean": None,
            "last_group_mean": None,
            "ema": None,
            "degenerate_groups": 0,
            "nondegenerate_groups": 0,
            "nondegenerate_rate": None,
        },
        "evaluation": None,
    }
    assert run["train"]["steps_completed"] == 0
    assert run["train"]["steps_target"] == config.grpo.max_steps
    assert run["train"]["last_metrics"] == {
        "loss": None,
        "grad_norm": None,
        "kl": None,
        "entropy": None,
        "importance_sampling_ratio_mean": None,
    }
    assert run["artifacts"] == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "plot": None,
        "plot_status": "pending",
        "train_log": "train.log",
        "vllm_log": "vllm.log",
        "checkpoints": [],
        "final_model": None,
        "cleanup_details": "cleanup.json",
    }
    assert run["cleanup"] == {
        "status": "pending",
        "clean_release": None,
        "residual_count": 0,
    }
    with pytest.raises(FileExistsError):
        RunRecorder(config=config, seed=123)


def test_supervisor_workspace_preserves_vllm_log_and_records_runtime_endpoints(tmp_path: Path) -> None:
    config = configured_for(tmp_path, run_id="supervised-run")
    output_dir = tmp_path / "outputs/supervised-run"
    output_dir.mkdir(parents=True)
    (output_dir / ".swe-agent-supervisor-workspace").write_text(
        "supervised-run", encoding="utf-8"
    )
    (output_dir / "vllm.log").write_text("server started\n", encoding="utf-8")

    recorder = RunRecorder(config=config, seed=7, workspace_prepared=True)
    recorder.set_vllm_endpoints(server_url="http://127.0.0.1:18421", group_port=18422)

    assert (output_dir / "vllm.log").read_text(encoding="utf-8") == "server started\n"
    run = load_json(output_dir / "run.json")
    assert run["provenance"]["vllm_endpoints"] == {
        "server_url": "http://127.0.0.1:18421",
        "group_port": 18422,
    }


def test_group_rollouts_rewards_metrics_and_consumption(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7, run_id="run-a")
    prompt = [{"role": "user", "content": "repair it"}]
    rollout_dirs = recorder.begin_group(
        prompt, 4, task_id="getmoto__moto-7023"
    )
    assert [path.name for path in rollout_dirs] == ["0000", "0001", "0002", "0003"]
    with pytest.raises(RuntimeError, match="previous group must complete"):
        recorder.begin_group(prompt, 4, task_id="getmoto__moto-7023")

    batch_path = recorder.output_dir / "rollouts/batch-0000/batch.json"
    group_path = recorder.output_dir / "rollouts/batch-0000/group-0000/group.json"
    batch = load_json(batch_path)
    group = load_json(group_path)
    assert batch["state"] == "running"
    assert batch["task_id"] == "getmoto__moto-7023"
    assert batch["global_step_at_generation"] == 0
    assert group["task_id"] == "getmoto__moto-7023"

    messages = prompt + [{"role": "assistant", "content": "done"}]
    first_verification = verification()
    recorder.write_rollout(
        0,
        messages=messages,
        trajectory=trajectory(),
        patch="diff --git a/x b/x\n",
        verification=first_verification,
    )
    assert load_json(rollout_dirs[0] / "messages.json") == messages
    assert load_json(rollout_dirs[0] / "trajectory.json") == trajectory().model_dump(
        mode="json"
    )
    assert (rollout_dirs[0] / "final_patch.diff").read_text() == (
        "diff --git a/x b/x\n"
    )
    assert load_json(
        rollout_dirs[0] / "verifier.json"
    ) == first_verification.model_dump(mode="json")

    recorder.write_rollout(
        1,
        messages=messages,
        trajectory=trajectory("format_exhausted"),
        patch=None,
        verification=None,
    )
    assert {path.name for path in rollout_dirs[1].iterdir()} == {
        "messages.json",
        "trajectory.json",
    }

    recorder.complete_group(
        episode_ids=["e0", "e1", "e2", "e3"],
        rewards=[1.0, 0.0, 0.0, 0.0],
        verifications=[first_verification, verification("unresolved"), None, None],
        settlements=[
            Settlement(status="resolved"),
            Settlement(status="unresolved"),
            Settlement(status="empty_patch"),
            Settlement(status="agent_error"),
        ],
    )
    completed_group = load_json(group_path)
    assert completed_group["state"] == "completed"
    assert completed_group["rewards"] == [1, 0, 0, 0]
    assert completed_group["reward_mean"] == 0.25
    assert completed_group["reward_std"] == pytest.approx(0.4330127019)
    assert completed_group["degenerate"] is False
    assert completed_group["group_size"] == 4
    assert completed_group["scorable_count"] == 4
    assert completed_group["censored_count"] == 0
    assert completed_group["fully_censored"] is False
    assert completed_group["settlement_counts"] == {
        "resolved": 1,
        "unresolved": 1,
        "empty_patch": 1,
        "agent_error": 1,
        "infra_error": 0,
    }
    assert completed_group["verification_counts"] == {
        "resolved": 1,
        "unresolved": 1,
        "not_run": 2,
    }

    run = load_json(recorder.output_dir / "run.json")
    assert run["results"]["reward"] == {
        "successes": 1,
        "attempts": 4,
        "mean": 0.25,
        "last_group_mean": 0.25,
        "ema": 0.25,
        "degenerate_groups": 0,
        "nondegenerate_groups": 1,
        "nondegenerate_rate": 1.0,
    }
    assert run["train"]["groups_generated"] == 1
    assert run["train"]["rollouts_generated"] == 4

    assert recorder.record_metrics(
        step=1,
        logs={
            "loss": 0.4,
            "grad_norm": 1.25,
            "kl": 0.02,
            "entropy": 0.8,
            "sampling/importance_sampling_ratio/mean": 1.01,
            "sampling/importance_sampling_ratio/max": 1.2,
            "num_tokens": 4096,
        },
    )
    assert not recorder.record_metrics(step=1, logs={"loss": 999})
    rows = metric_rows(recorder.metrics_path)
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["rollouts_cumulative"] == 4
    assert rows[0]["reward_mean_group"] == 0.25
    assert rows[0]["train_pass_rate_cumulative"] == 0.25
    assert rows[0]["group_degenerate"] is False
    assert rows[0]["importance_sampling_ratio_mean"] == 1.01
    run = load_json(recorder.output_dir / "run.json")
    assert run["train"]["tokens_generated"] == 4096
    assert run["train"]["last_metrics"] == {
        "loss": 0.4,
        "grad_norm": 1.25,
        "kl": 0.02,
        "entropy": 0.8,
        "importance_sampling_ratio_mean": 1.01,
    }

    recorder.complete_batch(1)
    completed_batch = load_json(batch_path)
    assert completed_batch["state"] == "completed"
    assert completed_batch["finished_at"] is not None
    assert completed_batch["consumed_by_global_steps"] == [1]
    assert load_json(recorder.output_dir / "run.json")["train"][
        "steps_completed"
    ] == 1


def test_complete_group_accepts_fractional_rewards(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7)
    recorder.begin_group(
        [{"role": "user", "content": "repair it"}], 3, task_id="getmoto__moto-7023"
    )
    recorder.complete_group(
        episode_ids=["e0", "e1", "e2"],
        rewards=[0.0, 0.018, 1.0],
        verifications=[
            verification("unresolved"),
            verification("unresolved"),
            verification(),
        ],
        settlements=[
            Settlement(status="unresolved"),
            Settlement(status="unresolved"),
            Settlement(status="resolved"),
        ],
    )

    group = load_json(recorder.output_dir / "rollouts/batch-0000/group-0000/group.json")
    assert group["rewards"] == [0.0, 0.018, 1.0]
    assert group["reward_mean"] == pytest.approx(1.018 / 3)
    assert group["degenerate"] is False

    reward = load_json(recorder.output_dir / "run.json")["results"]["reward"]
    # 部分得分不算成功，只有 reward == 1.0 计入 successes
    assert reward["successes"] == 1
    assert reward["attempts"] == 3
    assert reward["mean"] == pytest.approx(1.018 / 3)
    assert reward["last_group_mean"] == pytest.approx(1.018 / 3)


def test_complete_group_excludes_censored_rewards_from_statistics(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7)
    recorder.begin_group(
        [{"role": "user", "content": "repair it"}], 3, task_id="getmoto__moto-7023"
    )
    recorder.complete_group(
        episode_ids=["e0", "e1", "e2"],
        rewards=[None, 0.0, 1.0],
        verifications=[None, verification("unresolved"), verification()],
        settlements=[
            Settlement(status="infra_error"),
            Settlement(status="unresolved"),
            Settlement(status="resolved"),
        ],
    )

    group = load_json(recorder.output_dir / "rollouts/batch-0000/group-0000/group.json")
    assert group["rewards"] == [None, 0.0, 1.0]
    assert group["reward_mean"] == 0.5
    assert group["reward_std"] == 0.5
    assert group["degenerate"] is False
    assert group["group_size"] == 3
    assert group["scorable_count"] == 2
    assert group["censored_count"] == 1
    assert group["fully_censored"] is False
    assert group["settlement_counts"]["infra_error"] == 1

    reward = load_json(recorder.output_dir / "run.json")["results"]["reward"]
    assert reward["successes"] == 1
    assert reward["attempts"] == 2
    assert reward["mean"] == 0.5
    assert recorder.run["train"]["rollouts_generated"] == 3


def test_complete_group_records_fully_censored_without_reward_statistics(
    tmp_path: Path,
) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7)
    recorder.begin_group(
        [{"role": "user", "content": "repair it"}],
        4,
        task_id="getmoto__moto-7023",
    )
    recorder.complete_group(
        episode_ids=["e0", "e1", "e2", "e3"],
        rewards=[None, None, None, None],
        verifications=[None, None, None, None],
        settlements=[Settlement(status="infra_error")] * 4,
    )

    group = load_json(
        recorder.output_dir / "rollouts/batch-0000/group-0000/group.json"
    )
    assert group["group_size"] == 4
    assert group["scorable_count"] == 0
    assert group["censored_count"] == 4
    assert group["fully_censored"] is True
    assert group["reward_mean"] is None
    assert group["reward_std"] is None
    assert group["degenerate"] is None
    assert group["settlement_counts"]["infra_error"] == 4

    reward = recorder.run["results"]["reward"]
    assert reward["attempts"] == 0
    assert reward["mean"] is None
    assert reward["ema"] is None
    assert reward["degenerate_groups"] == 0
    assert reward["nondegenerate_groups"] == 0
    assert reward["nondegenerate_rate"] is None
    assert recorder.run["train"]["groups_generated"] == 1
    assert recorder.run["train"]["fully_censored_groups"] == 1


@pytest.mark.parametrize(
    "rewards",
    [[0.0, 1.2, 1.0], [0.0, -0.1, 1.0]],
    ids=["above-one", "below-zero"],
)
def test_complete_group_rejects_out_of_range_rewards(
    tmp_path: Path, rewards: list[float]
) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7)
    recorder.begin_group(
        [{"role": "user", "content": "repair it"}], 3, task_id="getmoto__moto-7023"
    )
    with pytest.raises(ValueError, match=r"group rewards must be within \[0, 1\]"):
        recorder.complete_group(
            episode_ids=["e0", "e1", "e2"],
            rewards=rewards,
            verifications=[None, None, None],
            settlements=[Settlement(status="unresolved")] * 3,
        )


def test_native_policy_path_is_internal_only(tmp_path: Path) -> None:
    recorder = RunRecorder(
        config=configured_for(tmp_path), seed=1, run_id="native-path-run"
    )

    recorder.observe_native_policy_path(False)
    recorder.observe_native_policy_path(True)
    recorder.observe_native_policy_path(False)

    assert recorder.native_policy_path_reached is True
    serialized = json.dumps(load_json(recorder.output_dir / "run.json"))
    assert "native_policy_path" not in serialized


def test_sync_trainer_state_keeps_partial_progress_and_checkpoints(
    tmp_path: Path,
) -> None:
    recorder = RunRecorder(
        config=configured_for(tmp_path), seed=1, run_id="partial-run"
    )
    (recorder.output_dir / "checkpoint-8").mkdir()
    recorder.sync_trainer_state(
        global_step=8,
        log_history=[
            {
                "step": 8,
                "loss": 0.7,
                "grad_norm": 2.0,
                "sampling/importance_sampling_ratio/mean": 0.98,
            }
        ],
    )

    run = load_json(recorder.output_dir / "run.json")
    assert run["train"]["steps_completed"] == 8
    assert run["train"]["last_metrics"]["loss"] == 0.7
    assert run["artifacts"]["checkpoints"] == ["checkpoint-8"]
    assert metric_rows(recorder.metrics_path)[0]["step"] == 8


def test_failure_and_interruption_finish_active_indexes(tmp_path: Path) -> None:
    recorder = RunRecorder(
        config=configured_for(tmp_path), seed=1, run_id="failed-run"
    )
    recorder.begin_group("prompt", 4, task_id="getmoto__moto-7023")
    recorder.fail(
        category="docker",
        primary_type="DockerRuntimeError",
        message="daemon unavailable",
        stage="generation",
    )
    run = load_json(recorder.output_dir / "run.json")
    assert run["status"] == "failed"
    assert run["failure"] == {
        "category": "docker",
        "type": "DockerRuntimeError",
        "message": "daemon unavailable",
        "stage": "generation",
        "log": "train.log",
    }
    assert run["time"]["finished_at"] is not None
    assert run["time"]["duration_seconds"] is not None
    assert load_json(recorder._batch_dir / "batch.json")["state"] == "failed"
    assert load_json(recorder._group_dir / "group.json")["state"] == "failed"

    interrupted = RunRecorder(
        config=configured_for(tmp_path), seed=2, run_id="interrupted-run"
    )
    interrupted.begin_group("prompt", 4, task_id="getmoto__moto-7023")
    interrupted.fail(
        category="trainer",
        primary_type="KeyboardInterrupt",
        message="interrupted",
        stage="train",
        interrupted=True,
    )
    run = load_json(interrupted.output_dir / "run.json")
    assert run["status"] == "interrupted"
    assert run["failure"]["category"] == "interrupted"


def test_cleanup_details_are_separate_and_completion_is_gated(tmp_path: Path) -> None:
    recorder = RunRecorder(
        config=configured_for(tmp_path), seed=1, run_id="cleanup-run"
    )
    failed_operation = {
        "sequence": 1,
        "at": "2026-07-20T00:00:00Z",
        "operation": "remove",
        "result": "failed",
        "error": "busy",
    }
    success_operation = {
        "sequence": 2,
        "at": "2026-07-20T00:00:01Z",
        "operation": "remove",
        "result": "success",
        "error": None,
    }
    common = {
        "scope": "rollout",
        "task_id": "getmoto__moto-7023",
        "episode_id": "episode-1",
        "container_name": "rollout-1",
        "container_id": "a" * 64,
    }
    recorder.merge_cleanup_events(
        [{**common, "operations": [failed_operation], "residual": True}]
    )
    recorder.merge_cleanup_events(
        [{**common, "operations": [success_operation], "residual": False}]
    )
    recorder.set_processes([])
    recorder.set_runtime_handles([])
    recorder.set_gpu_diagnostics([{"device": "2", "diagnostic_only": True}])
    recorder.finalize_cleanup()

    run_cleanup = load_json(recorder.output_dir / "run.json")["cleanup"]
    details = load_json(recorder.cleanup_path)
    assert run_cleanup == {
        "status": "completed",
        "clean_release": True,
        "residual_count": 0,
    }
    assert details["status"] == "completed"
    assert details["residuals"] == []
    assert details["containers"][0]["final_state"] == "removed"
    assert [
        item["result"] for item in details["containers"][0]["operations"]
    ] == ["failed", "success"]
    recorder.complete()
    assert load_json(recorder.output_dir / "run.json")["status"] == "completed"

    residual = RunRecorder(
        config=configured_for(tmp_path), seed=2, run_id="residual-run"
    )
    residual.merge_cleanup_events(
        [{**common, "operations": [failed_operation], "residual": True}]
    )
    residual.finalize_cleanup()
    run_cleanup = load_json(residual.output_dir / "run.json")["cleanup"]
    details = load_json(residual.cleanup_path)
    assert run_cleanup == {
        "status": "failed",
        "clean_release": False,
        "residual_count": 1,
    }
    assert details["residuals"] == ["a" * 64]
    with pytest.raises(RuntimeError, match="cannot complete"):
        residual.complete()
