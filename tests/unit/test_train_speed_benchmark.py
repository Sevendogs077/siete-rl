from __future__ import annotations

from copy import deepcopy

from scripts.train_speed_benchmark import (
    ArmResult,
    ArmSpec,
    build_arm_config,
    candidate_passes,
    executable_path,
    reset_worker_arms,
    screening_arms,
)
from scripts.docker_parallel_benchmark import WORKER_ORDERS, timing_summary


def _base_config() -> dict:
    return {
        "dataset": {
            "tasks_dir": "assets/swegym",
            "task_ids": None,
            "max_tasks": None,
            "official_path": "data/official.parquet",
            "subset_path": "data/subset.parquet",
        },
        "model": {
            "model_path": "~/.cache/model",
            "tokenizer_path": "~/.cache/model",
        },
        "generation": {
            "reset_parallel_workers": 1,
            "tool_parallel_workers": 16,
            "verifier_parallel_workers": 16,
        },
        "grpo": {
            "num_generations": 16,
            "gradient_accumulation_steps": 16,
            "generation_batch_size": 16,
            "max_steps": 100,
            "gradient_checkpointing": True,
            "save_strategy": "steps",
        },
        "vllm": {"gpu_memory_utilization": 0.45},
        "output": {"output_root": "outputs", "run_id": None},
    }


def test_build_arm_config_is_single_variable_and_does_not_mutate_base(tmp_path) -> None:
    base = _base_config()
    original = deepcopy(base)
    spec = ArmSpec("screen-vllm60", 0.60, generations=4)

    result = build_arm_config(
        base,
        spec,
        task_id="getmoto__moto-5189",
        shared_root=tmp_path / "shared",
        output_root=tmp_path / "results",
    )

    assert base == original
    assert result["dataset"]["task_ids"] == ["getmoto__moto-5189"]
    assert result["dataset"]["max_tasks"] == 1
    assert result["grpo"]["num_generations"] == 4
    assert result["grpo"]["generation_batch_size"] == 4
    assert result["grpo"]["gradient_accumulation_steps"] == 4
    assert result["grpo"]["max_steps"] == 1
    assert result["grpo"]["save_strategy"] == "steps"
    assert result["grpo"]["save_steps"] == 1
    assert result["grpo"]["save_total_limit"] == 1
    assert result["generation"]["reset_parallel_workers"] == 1
    assert result["vllm"]["gpu_memory_utilization"] == 0.60
    assert result["output"]["run_id"] == "screen-vllm60"
    assert result["output"]["output_root"] == str(tmp_path / "results")
    assert result["dataset"]["tasks_dir"] == str(tmp_path / "shared/assets/swegym")


def test_screening_order_controls_warmup_with_aba() -> None:
    arms = screening_arms()

    assert [arm.name for arm in arms] == [
        "screen-baseline-a1",
        "screen-vllm60-b1",
        "screen-baseline-a2",
    ]
    assert [arm.vllm_memory_utilization for arm in arms] == [0.45, 0.60, 0.45]
    assert {arm.generations for arm in arms} == {4}


def test_reset_worker_arms_change_only_reset_concurrency() -> None:
    arms = reset_worker_arms()

    assert [arm.name for arm in arms] == ["reset-workers8", "reset-workers16"]
    assert [arm.reset_workers for arm in arms] == [8, 16]
    assert {arm.vllm_memory_utilization for arm in arms} == {0.45}
    assert {arm.generations for arm in arms} == {16}
    assert {arm.tool_workers for arm in arms} == {16}
    assert {arm.verifier_workers for arm in arms} == {16}


def test_candidate_requires_completed_runs_and_five_percent_speedup() -> None:
    baselines = [
        ArmResult("a1", 100.0, "completed", 1, 4, 0),
        ArmResult("a2", 104.0, "completed", 1, 4, 0),
    ]

    assert candidate_passes(
        baselines,
        ArmResult("b", 95.0, "completed", 1, 4, 0),
    )
    assert not candidate_passes(
        baselines,
        ArmResult("b", 98.0, "completed", 1, 4, 0),
    )
    assert not candidate_passes(
        baselines,
        ArmResult("b", 90.0, "failed", 0, 0, 1),
    )


def test_timing_summary_reports_median_and_range() -> None:
    assert timing_summary([3.0, 1.0, 2.0]) == {
        "runs": 3,
        "median_seconds": 2.0,
        "min_seconds": 1.0,
        "max_seconds": 3.0,
    }


def test_docker_worker_order_rotates_all_reset_candidates() -> None:
    candidates = {1, 4, 8, 16}

    assert len(WORKER_ORDERS) == 4
    assert all(set(order) == candidates for order in WORKER_ORDERS)
    assert all(
        {order[position] for order in WORKER_ORDERS} == candidates
        for position in range(4)
    )


def test_executable_path_preserves_virtualenv_symlink(tmp_path) -> None:
    target = tmp_path / "python-real"
    target.touch()
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)

    assert executable_path(venv_python) == venv_python.absolute()
