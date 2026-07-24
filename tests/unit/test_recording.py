from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from swe_agent.config import load_config
from swe_agent.models import Action, Observation, Step, Trajectory, Verification
from swe_agent.recording import RunRecorder, generate_run_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


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


def trajectory(termination: str = "submitted") -> Trajectory:
    return Trajectory.model_validate(
        {
            "task_id": "getmoto__moto-7023",
            "environment_id": "getmoto__moto-7023",
            "steps": [
                Step(
                    index=0,
                    action=Action(tool_name="read_file", arguments={"path": "README.md"}),
                    observation=Observation(text="1: hello", exit_code=0),
                )
            ],
            "termination": termination,
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
        "train.log",
    }
    assert yaml.safe_load((recorder.output_dir / "config.yaml").read_text()) == config.model_dump(
        mode="json"
    )

    run = load_json(recorder.output_dir / "run.json")
    assert set(run) == {
        "schema_version",
        "identity",
        "provenance",
        "lifecycle",
        "failure",
        "training",
        "cleanup",
    }
    assert set(run["identity"]) == {"run_id", "output_dir", "config_file"}
    assert set(run["provenance"]) == {
        "started_at",
        "finished_at",
        "code_commit",
        "code_dirty",
        "dependency_versions",
        "model_path",
        "resolved_model_path",
        "model_revision",
        "generation_backend",
        "official_dataset_revision",
        "subset_dataset_revision",
        "task_id",
        "image_tag",
        "image_id",
        "image_platform",
        "seed",
    }
    assert run["lifecycle"] == {"state": "running"}
    assert run["failure"] is None
    assert run["training"] == {
        "system_closed_loop": "pending",
        "native_policy_path_reached": None,
        "trainer_group_consumed": None,
        "global_step": 0,
        "groups_generated": 0,
        "rollouts_generated": 0,
        "reward_mean": None,
        "reward_std": None,
        "loss": None,
        "grad_norm": None,
        "frac_reward_zero_std": None,
        "checkpoints": [],
        "final_model_ref": None,
        "observations": {
            "reward_degenerate": None,
            "nonzero_advantage_observed": None,
            "nonzero_gradient_observed": None,
            "nonzero_parameter_update_observed": None,
            "all_sequences_masked_by_is": None,
        },
    }
    assert run["cleanup"] == {
        "state": "pending",
        "clean_release": None,
        "residuals": [],
        "containers": [],
        "processes": [],
        "runtime_handles": [],
        "gpu_diagnostics": [],
    }
    with pytest.raises(FileExistsError):
        RunRecorder(config=config, seed=123)


def test_first_group_rollout_and_consumption_files(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=7, run_id="run-a")
    prompt = [{"role": "user", "content": "repair it"}]
    rollout_dirs = recorder.begin_group(prompt, 4)
    assert [path.name for path in rollout_dirs] == ["0000", "0001", "0002", "0003"]
    with pytest.raises(RuntimeError, match="previous group must complete"):
        recorder.begin_group(prompt, 4)

    batch_path = recorder.output_dir / "rollouts/batch-0000/batch.json"
    group_path = recorder.output_dir / "rollouts/batch-0000/group-0000/group.json"
    batch = load_json(batch_path)
    group = load_json(group_path)
    assert set(batch) == {
        "schema_version",
        "batch_index",
        "batch_id",
        "state",
        "task_id",
        "generation_backend",
        "global_step_at_generation",
        "started_at",
        "finished_at",
        "groups",
        "consumed_by_global_steps",
    }
    assert batch["state"] == "running"
    assert batch["global_step_at_generation"] == 0
    assert batch["groups"] == ["group-0000"]
    assert batch["consumed_by_global_steps"] == []
    assert set(group) == {
        "schema_version",
        "group_index",
        "group_id",
        "state",
        "task_id",
        "prompt_sha256",
        "rollout_dirs",
        "episode_ids",
        "rewards",
        "reward_mean",
        "reward_std",
        "degenerate",
        "verification_counts",
    }

    messages = prompt + [{"role": "assistant", "content": "done"}]
    first_verification = verification()
    recorder.write_rollout(
        0,
        messages=messages,
        trajectory=trajectory(),
        patch="diff --git a/x b/x\n",
        verification=first_verification,
    )
    first = rollout_dirs[0]
    assert load_json(first / "messages.json") == messages
    assert load_json(first / "trajectory.json") == trajectory().model_dump(mode="json")
    assert (first / "final_patch.diff").read_text() == "diff --git a/x b/x\n"
    assert load_json(first / "verifier.json") == first_verification.model_dump(mode="json")

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

    checks = [first_verification, verification("unresolved"), None, None]
    recorder.complete_group(
        episode_ids=["e0", "e1", "e2", "e3"],
        rewards=[1.0, 0.0, 0.0, 0.0],
        verifications=checks,
    )
    completed_group = load_json(group_path)
    assert completed_group["state"] == "completed"
    assert completed_group["episode_ids"] == ["e0", "e1", "e2", "e3"]
    assert completed_group["rewards"] == [1, 0, 0, 0]
    assert completed_group["reward_mean"] == 0.25
    assert completed_group["degenerate"] is False
    assert load_json(recorder.output_dir / "run.json")["training"]["observations"][
        "reward_degenerate"
    ] is False
    assert completed_group["verification_counts"] == {
        "resolved": 1,
        "unresolved": 1,
        "not_run": 2,
    }
    recorder.complete_batch(1)
    completed_batch = load_json(batch_path)
    assert completed_batch["state"] == "completed"
    assert completed_batch["finished_at"] is not None
    assert completed_batch["consumed_by_global_steps"] == [1]
    assert load_json(recorder.output_dir / "run.json")["training"]["global_step"] == 1


def test_failure_and_interruption_finish_active_indexes(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=1, run_id="failed-run")
    recorder.begin_group("prompt", 4)
    recorder.update_training(
        system_closed_loop="failed",
        native_policy_path_reached=False,
        trainer_group_consumed=False,
    )
    recorder.fail(
        category="docker",
        primary_type="DockerRuntimeError",
        message="daemon unavailable",
        stage="generation",
    )
    run = load_json(recorder.output_dir / "run.json")
    assert run["lifecycle"]["state"] == "failed"
    assert run["failure"]["category"] == "docker"
    assert run["failure"]["traceback_log_ref"] == "train.log"
    assert run["provenance"]["finished_at"] is not None
    assert load_json(recorder._batch_dir / "batch.json")["state"] == "failed"
    assert load_json(recorder._group_dir / "group.json")["state"] == "failed"

    interrupted = RunRecorder(
        config=configured_for(tmp_path), seed=2, run_id="interrupted-run"
    )
    interrupted.begin_group("prompt", 4)
    interrupted.fail(
        category="trainer",
        primary_type="KeyboardInterrupt",
        message="interrupted",
        stage="train",
        interrupted=True,
    )
    run = load_json(interrupted.output_dir / "run.json")
    assert run["lifecycle"]["state"] == "interrupted"
    assert run["failure"]["category"] == "interrupted"


def test_cleanup_history_residuals_and_completion_gate(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=1, run_id="cleanup-run")
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
    recorder.set_gpu_diagnostics(
        [
            {
                "device": "2",
                "owner_pid": 123,
                "allocated_bytes_before": 0,
                "reserved_bytes_before": 0,
                "allocated_bytes_after": 0,
                "reserved_bytes_after": 0,
                "baseline_allocated_bytes": 0,
                "baseline_reserved_bytes": 0,
                "observed_at": "2026-07-20T00:00:02Z",
                "diagnostic_only": True,
                "note": "allocator remains above baseline",
            }
        ]
    )
    recorder.finalize_cleanup()
    cleanup = load_json(recorder.output_dir / "run.json")["cleanup"]
    assert cleanup["state"] == "completed"
    assert cleanup["clean_release"] is True
    assert cleanup["residuals"] == []
    assert cleanup["containers"][0]["final_state"] == "removed"
    assert [item["result"] for item in cleanup["containers"][0]["operations"]] == [
        "failed",
        "success",
    ]
    recorder.complete()
    assert load_json(recorder.output_dir / "run.json")["lifecycle"]["state"] == "completed"

    residual = RunRecorder(config=configured_for(tmp_path), seed=2, run_id="residual-run")
    residual.merge_cleanup_events(
        [{**common, "operations": [failed_operation], "residual": True}]
    )
    residual.finalize_cleanup()
    cleanup = load_json(residual.output_dir / "run.json")["cleanup"]
    assert cleanup["state"] == "failed"
    assert cleanup["clean_release"] is False
    assert cleanup["residuals"] == ["a" * 64]
    with pytest.raises(RuntimeError, match="cannot complete"):
        residual.complete()


def test_unknown_training_field_is_rejected(tmp_path: Path) -> None:
    recorder = RunRecorder(config=configured_for(tmp_path), seed=1, run_id="run-b")
    with pytest.raises(ValueError, match="unknown run training fields"):
        recorder.update_training(unknown=True)
