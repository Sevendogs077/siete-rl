#!/usr/bin/env python3
"""用真实 SWEEnvironment 测量 reset 与工具调用并发收益。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from types import MappingProxyType
from typing import Callable, Sequence

from siete_rl.config import load_config
from siete_rl.docker import DockerSandbox, SubprocessDockerClient, sweep_run_containers
from siete_rl.environment import SWEEnvironment
from siete_rl.swegym import load_task_instance


DEFAULT_TASK = "getmoto__moto-5189"
WORKER_ORDERS = (
    (1, 4, 8, 16),
    (4, 8, 16, 1),
    (8, 16, 1, 4),
    (16, 1, 4, 8),
)


def timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timings must not be empty")
    return {
        "runs": len(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _parallel_call(
    functions: Sequence[Callable[[], object]], workers: int
) -> None:
    if workers == 1:
        for function in functions:
            function()
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(functions))) as pool:
        list(pool.map(lambda function: function(), functions))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path("/home/2025user/zyp/work/2607_siete_rl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/docker_parallel.json"),
    )
    args = parser.parse_args()
    if args.environments < 1 or args.repeats < 1:
        raise ValueError("environments and repeats must be positive")

    config, project_root, _ = load_config(
        args.shared_root.resolve() / "configs/grpo_swegym_openhands_7b_lora.yaml"
    )
    pair = load_task_instance(config, project_root, args.task_id)
    task_context = MappingProxyType({args.task_id: pair})
    client = SubprocessDockerClient()
    run_id = f"speed-docker-{int(time.time())}"

    def make_environment(
        reset_executor: ThreadPoolExecutor | None = None,
    ) -> SWEEnvironment:
        def sandbox_factory(sample, episode_id: str, scope: str) -> DockerSandbox:
            return DockerSandbox(
                client=client,
                task=sample.task,
                environment=sample.environment,
                run_id=run_id,
                episode_id=episode_id,
                scope=scope,
            )

        return SWEEnvironment(
            task_context=task_context,
            sandbox_factory=sandbox_factory,
            verifier_factory=lambda *unused: None,
            output_limit_chars=config.chat.max_observation_chars,
            max_timeout_sec=config.docker.exec_timeout_sec,
            reward_type=config.grpo.reward_type,
            layered_lambda=config.grpo.layered_lambda,
            reset_executor=reset_executor,
        )

    reset_times: dict[int, list[float]] = {1: [], 4: [], 8: [], 16: []}
    tool_times: dict[int, list[float]] = {1: [], 4: [], 8: [], 16: []}
    try:
        for repeat in range(args.repeats):
            order = WORKER_ORDERS[repeat % len(WORKER_ORDERS)]
            for workers in order:
                reset_executor = (
                    None
                    if workers == 1
                    else ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="swe-reset-benchmark"
                    )
                )
                environments = [
                    make_environment(reset_executor)
                    for _ in range(args.environments)
                ]
                try:
                    started = time.perf_counter()
                    for environment in environments:
                        environment.reset(task_id=args.task_id)
                    for environment in environments:
                        environment._await_reset()
                    reset_times[workers].append(time.perf_counter() - started)
                finally:
                    _parallel_call(
                        [lambda env=env: env._close() for env in environments], 16
                    )
                    if reset_executor is not None:
                        reset_executor.shutdown(wait=True, cancel_futures=False)

            environments = [make_environment() for _ in range(args.environments)]
            try:
                _parallel_call(
                    [lambda env=env: env.reset(task_id=args.task_id) for env in environments],
                    16,
                )
                for workers in order:
                    started = time.perf_counter()
                    _parallel_call(
                        [
                            lambda env=env: env.execute_bash("git status --short >/dev/null")
                            for env in environments
                        ],
                        workers,
                    )
                    tool_times[workers].append(time.perf_counter() - started)
            finally:
                _parallel_call(
                    [lambda env=env: env._close() for env in environments], 16
                )
    finally:
        removed = sweep_run_containers(client, run_id)

    payload = {
        "task_id": args.task_id,
        "environments": args.environments,
        "repeats": args.repeats,
        "reset": {str(k): timing_summary(v) for k, v in reset_times.items()},
        "tool": {str(k): timing_summary(v) for k, v in tool_times.items()},
        "orphan_containers_removed": removed,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
