#!/usr/bin/env python3
"""在隔离 worktree 中运行短时、单变量的 GRPO 训练基准。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Sequence

import yaml


EXPECTED_BASE_COMMIT = "4699d00973bc07b763ae25d210d27ead645b76bf"
DEFAULT_TASK = "getmoto__moto-5189"
DEFAULT_DOCKER_HOST = "unix:///run/docker-swegym/docker.sock"
RESET_IMPLEMENTATION_FILES = frozenset(
    {
        "configs/grpo_swegym_openhands_7b_lora.yaml",
        "src/siete_rl/config.py",
        "src/siete_rl/environment.py",
        "src/siete_rl/recording.py",
        "src/siete_rl/train.py",
        "src/siete_rl/trainer.py",
    }
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    vllm_memory_utilization: float
    generations: int
    reset_workers: int = 1
    gradient_checkpointing: bool = True
    tool_workers: int = 16
    verifier_workers: int = 16


@dataclass(frozen=True, slots=True)
class ArmResult:
    name: str
    wall_seconds: float
    status: str
    global_step: int
    rollouts: int
    infra_errors: int
    reset_time_seconds: float | None = None
    step_time_seconds: float | None = None


def screening_arms() -> tuple[ArmSpec, ...]:
    """A/B/A 排序消除首次加载和文件缓存的单向偏差。"""

    return (
        ArmSpec("screen-baseline-a1", 0.45, generations=4),
        ArmSpec("screen-vllm60-b1", 0.60, generations=4),
        ArmSpec("screen-baseline-a2", 0.45, generations=4),
    )


def production_arms() -> tuple[ArmSpec, ...]:
    return (
        ArmSpec("full-baseline-a", 0.45, generations=16),
        ArmSpec("full-vllm60-b", 0.60, generations=16),
    )


def reset_worker_arms() -> tuple[ArmSpec, ...]:
    """Docker 重复筛选后的两个 GPU finalist；其余训练参数完全相同。"""

    return (
        ArmSpec("reset-workers8", 0.45, generations=16, reset_workers=8),
        ArmSpec("reset-workers16", 0.45, generations=16, reset_workers=16),
    )


def build_arm_config(
    base: dict[str, Any],
    spec: ArmSpec,
    *,
    task_id: str,
    shared_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """从正式配置产生一份可审计的单步配置，不修改输入对象。"""

    payload = deepcopy(base)
    dataset = payload["dataset"]
    dataset["tasks_dir"] = str(shared_root / "assets/swegym")
    dataset["official_path"] = str(
        shared_root
        / "data/swegym/SWE-Gym__SWE-Gym/"
        "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb/data/train-00000-of-00001.parquet"
    )
    dataset["subset_path"] = str(
        shared_root
        / "data/swegym/SumanthRH__SWE-Gym-Subset/"
        "3f22e68f673027edbaebe3424e4c20ae580563fd/data/train-00000-of-00001.parquet"
    )
    dataset["task_ids"] = [task_id]
    dataset["max_tasks"] = 1

    grpo = payload["grpo"]
    grpo["num_generations"] = spec.generations
    grpo["generation_batch_size"] = spec.generations
    grpo["gradient_accumulation_steps"] = spec.generations
    grpo["max_steps"] = 1
    grpo["gradient_checkpointing"] = spec.gradient_checkpointing
    # train.py 的成功契约要求至少一个 checkpoint-<step>。
    grpo["save_strategy"] = "steps"
    grpo["save_steps"] = 1
    grpo["save_total_limit"] = 1

    generation = payload["generation"]
    generation["reset_parallel_workers"] = spec.reset_workers
    generation["tool_parallel_workers"] = spec.tool_workers
    generation["verifier_parallel_workers"] = spec.verifier_workers
    payload["vllm"]["gpu_memory_utilization"] = spec.vllm_memory_utilization
    payload["output"]["output_root"] = str(output_root)
    payload["output"]["run_id"] = spec.name
    return payload


def candidate_passes(
    baselines: Sequence[ArmResult],
    candidate: ArmResult,
    *,
    minimum_speedup: float = 0.05,
) -> bool:
    healthy_baselines = [
        row.wall_seconds
        for row in baselines
        if row.status == "completed" and row.global_step == 1 and row.infra_errors == 0
    ]
    if len(healthy_baselines) != len(baselines) or not healthy_baselines:
        return False
    if (
        candidate.status != "completed"
        or candidate.global_step != 1
        or candidate.infra_errors != 0
    ):
        return False
    reference = statistics.median(healthy_baselines)
    return candidate.wall_seconds <= reference * (1.0 - minimum_speedup)


def executable_path(path: Path) -> Path:
    """返回绝对入口路径但不解引用 venv symlink。"""

    return path.expanduser().absolute()


def _git_output(worktree: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=worktree, text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_worktree(
    worktree: Path, *, allow_reset_implementation: bool = False
) -> None:
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, "HEAD"],
        cwd=worktree,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(f"benchmark branch must descend from {EXPECTED_BASE_COMMIT}")
    source_diff = _git_output(
        worktree, "diff", "--name-only", f"{EXPECTED_BASE_COMMIT}..HEAD", "--", "src", "configs"
    )
    changed_source = set(source_diff.splitlines()) if source_diff else set()
    if changed_source and not allow_reset_implementation:
        raise RuntimeError(
            "benchmark branch changes training source/config relative to the pinned base: "
            + source_diff.replace("\n", ", ")
        )
    unexpected = changed_source - RESET_IMPLEMENTATION_FILES
    if allow_reset_implementation and unexpected:
        raise RuntimeError(
            "reset benchmark has unexpected source/config changes: "
            + ", ".join(sorted(unexpected))
        )
    dirty = _git_output(worktree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("benchmark requires a clean tracked worktree")


def _sample_gpus(stop: threading.Event, destination: Path) -> None:
    fields = (
        "timestamp,index,memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw"
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields.split(","))
        while not stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    "0,1",
                    f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    writer.writerow([value.strip() for value in line.split(",")])
                handle.flush()
            stop.wait(2.0)


def _read_result(run_dir: Path, arm: ArmSpec, wall_seconds: float) -> ArmResult:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return ArmResult(arm.name, wall_seconds, "missing_run", 0, 0, 1)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    metrics = []
    if metrics_path.is_file():
        metrics = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    last = metrics[-1] if metrics else {}
    failure = run.get("failure")
    infra_errors = int(
        bool(failure)
        and str(failure.get("category", "")).lower()
        in {"docker", "verifier", "infrastructure", "oom"}
    )
    return ArmResult(
        arm.name,
        wall_seconds,
        str(run.get("status", "unknown")),
        int(run.get("train", {}).get("global_step", last.get("step", 0)) or 0),
        int(last.get("rollouts_cumulative", 0) or 0),
        infra_errors,
        (
            float(last["reset_time_seconds"])
            if last.get("reset_time_seconds") is not None
            else None
        ),
        (
            float(last["step_time_seconds"])
            if last.get("step_time_seconds") is not None
            else None
        ),
    )


def run_arm(
    arm: ArmSpec,
    *,
    worktree: Path,
    shared_root: Path,
    python: Path,
    task_id: str,
    results_root: Path,
) -> ArmResult:
    configs_dir = results_root / "configs"
    runs_dir = results_root / "runs"
    telemetry_dir = results_root / "telemetry"
    for directory in (configs_dir, runs_dir, telemetry_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(
        (worktree / "configs/grpo_swegym_openhands_7b_lora.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload = build_arm_config(
        base,
        arm,
        task_id=task_id,
        shared_root=shared_root,
        output_root=runs_dir,
    )
    config_path = configs_dir / f"{arm.name}.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    env = dict(os.environ)
    model_path = Path.home() / ".cache/modelscope/hub/models/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent"
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "DOCKER_HOST": env.get("DOCKER_HOST", DEFAULT_DOCKER_HOST),
            "PYTHONPATH": str(worktree / "src"),
            "MODEL_PATH": str(model_path),
            "TOKENIZER_PATH": str(model_path),
        }
    )
    log_path = telemetry_dir / f"{arm.name}.launcher.log"
    gpu_path = telemetry_dir / f"{arm.name}.gpu.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpus, args=(stop, gpu_path), daemon=True)
    started = time.monotonic()
    sampler.start()
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(python), "-m", "siete_rl.cli", "grpo", "--config", str(config_path)],
                cwd=worktree,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30 * 60,
                check=False,
                text=True,
            )
    finally:
        stop.set()
        sampler.join(timeout=5)
    wall_seconds = time.monotonic() - started
    result = _read_result(runs_dir / arm.name, arm, wall_seconds)
    summary = {
        **asdict(arm),
        **asdict(result),
        "exit_code": completed.returncode if completed is not None else None,
    }
    (telemetry_dir / f"{arm.name}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("screen", "full", "reset"), default="screen"
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument(
        "--worktree", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path("/home/2025user/zyp/work/2607_siete_rl"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/2025user/zyp/work/2607_siete_rl/.venv/bin/python"),
    )
    parser.add_argument("--results-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    worktree = args.worktree.resolve()
    validate_worktree(
        worktree, allow_reset_implementation=args.phase == "reset"
    )
    results_root = (
        args.results_root.resolve()
        if args.results_root
        else worktree / "benchmark_results"
    )
    if args.phase == "screen":
        arms = screening_arms()
    elif args.phase == "full":
        arms = production_arms()
    else:
        arms = reset_worker_arms()
    manifest = {
        "base_commit": EXPECTED_BASE_COMMIT,
        "harness_commit": _git_output(worktree, "rev-parse", "HEAD"),
        "phase": args.phase,
        "task_id": args.task_id,
        "arms": [asdict(arm) for arm in arms],
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / f"manifest-{args.phase}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for arm in arms:
        print(f"running {arm.name}", flush=True)
        result = run_arm(
            arm,
            worktree=worktree,
            shared_root=args.shared_root.resolve(),
            python=executable_path(args.python),
            task_id=args.task_id,
            results_root=results_root,
        )
        results.append(result)
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        if result.status != "completed":
            break
    if args.phase == "screen" and len(results) == 3:
        passed = candidate_passes([results[0], results[2]], results[1])
        print(json.dumps({"vllm60_passes": passed}), flush=True)
    return 0 if results and all(row.status == "completed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
