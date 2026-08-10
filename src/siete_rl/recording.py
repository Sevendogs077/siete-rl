"""单主进程、原子写入的 run/batch/group/rollout 记录器。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

import yaml

from siete_rl.config import ProjectConfig
from siete_rl.models import Trajectory, Verification


RunState = Literal["running", "completed", "failed", "interrupted"]
IndexState = Literal["running", "completed", "failed", "interrupted"]

REWARD_EMA_ALPHA = 0.2
DEPENDENCIES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "vllm",
    "datasets",
    "pyarrow",
    "pydantic",
    "matplotlib",
)
LAST_METRIC_KEYS = {
    "loss": "loss",
    "grad_norm": "grad_norm",
    "kl": "kl",
    "entropy": "entropy",
    "importance_sampling_ratio_mean": "sampling/importance_sampling_ratio/mean",
}
STEP_METRIC_KEYS = {
    **LAST_METRIC_KEYS,
    "importance_sampling_ratio_max": "sampling/importance_sampling_ratio/max",
    "clip_ratio": "clip_ratio/region_mean",
    "completion_length_mean": "completions/mean_length",
    "completion_clipped_ratio": "completions/clipped_ratio",
    "tool_call_frequency": "tools/call_frequency",
    "tool_failure_frequency": "tools/failure_frequency",
    "step_time_seconds": "step_time",
    "process_mask_candidate_turns": "process_mask/candidate_turns",
    "process_mask_applied_turns": "process_mask/applied_turns",
    "process_mask_retained_negative_turns": "process_mask/retained_negative_turns",
    "process_mask_masked_token_frac": "process_mask/masked_token_frac",
    "process_mask_governance_masked": "process_mask/governance_masked",
    "reward_zero_std_frac": "frac_reward_zero_std",
    "settlement_recovered_positive_rows": "settlement/recovered_positive_rows",
    "settlement_recovered_positive_active_tokens": (
        "settlement/recovered_positive_active_tokens"
    ),
}


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(2)}"


def installed_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class RunRecorder:
    """持有内存状态并作为所有项目 JSON 文件的唯一写者。"""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        seed: int,
        run_id: str | None = None,
        dependency_versions: dict[str, str] | None = None,
        code_commit: str | None = None,
        code_dirty: bool = True,
        model_revision: str | None = None,
        workspace_prepared: bool = False,
    ) -> None:
        self.config = config
        self.run_id = run_id or config.output.run_id or generate_run_id()
        self.output_dir = (Path(config.output.output_root) / self.run_id).resolve()
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        workspace_marker = self.output_dir / ".swe-agent-supervisor-workspace"
        if workspace_prepared:
            try:
                marker_run_id = workspace_marker.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ValueError(
                    f"supervisor workspace is missing its ownership marker: {self.output_dir}"
                ) from exc
            if marker_run_id != self.run_id:
                raise ValueError(
                    f"supervisor workspace belongs to {marker_run_id!r}, not {self.run_id!r}"
                )
        else:
            self.output_dir.mkdir(exist_ok=False)
        self.rollouts_root = self.output_dir / "rollouts"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.cleanup_path = self.output_dir / "cleanup.json"
        self.log_path = self.output_dir / config.output.train_log
        self._batch_dir: Path | None = None
        self._group_dir: Path | None = None
        self.batch: dict[str, Any] | None = None
        self.group: dict[str, Any] | None = None
        self._all_rewards: list[float] = []
        self._degenerate_groups = 0
        self._reward_ema: float | None = None
        self._last_recorded_step = 0
        self._native_policy_path_reached = False
        self._started_at = datetime.now(UTC)
        started_at = _format_utc(self._started_at)

        self.cleanup_details: dict[str, Any] = {
            "status": "pending",
            "clean_release": None,
            "residuals": [],
            "containers": [],
            "processes": [],
            "runtime_handles": [],
            "gpu_diagnostics": [],
        }
        self.run: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "failure": None,
            "results": {
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
            },
            "train": {
                "steps_completed": 0,
                "steps_target": config.grpo.max_steps,
                "groups_generated": 0,
                "rollouts_generated": 0,
                "tokens_generated": 0,
                "model_updated": None,
                "last_metrics": {
                    "loss": None,
                    "grad_norm": None,
                    "kl": None,
                    "entropy": None,
                    "importance_sampling_ratio_mean": None,
                },
            },
            "artifacts": {
                "config": "config.yaml",
                "metrics": "metrics.jsonl",
                "plot": None,
                "plot_status": "pending",
                "train_log": config.output.train_log,
                "vllm_log": "vllm.log",
                "checkpoints": [],
                "final_model": None,
                "cleanup_details": "cleanup.json",
            },
            "time": {
                "started_at": started_at,
                "finished_at": None,
                "duration_seconds": None,
            },
            "config": {
                "model": config.model.provenance_id,
                "algorithm": "grpo",
                "reward": config.grpo.reward_type,
                "max_steps": config.grpo.max_steps,
                "num_generations": config.grpo.num_generations,
                "gradient_accumulation_steps": config.grpo.gradient_accumulation_steps,
                "learning_rate": config.grpo.learning_rate,
                "beta": config.grpo.beta,
                "max_completion_length": config.generation.max_completion_length,
                "use_liger_kernel": config.generation.use_liger_kernel,
                "training_mode": config.model.training_mode,
            },
            "cleanup": {
                "status": "pending",
                "clean_release": None,
                "residual_count": 0,
            },
            "provenance": {
                "code_commit": code_commit,
                "code_dirty": code_dirty,
                "seed": seed,
                "model_path": config.model.model_path,
                "resolved_model_path": Path(config.model.model_path).resolve().as_posix(),
                "model_revision": model_revision,
                "generation_backend": "vllm",
                "official_dataset_revision": config.dataset.official_revision,
                "subset_dataset_revision": config.dataset.subset_revision,
                "image_platform": config.docker.platform,
                "dependency_versions": dependency_versions or installed_dependency_versions(),
                "vllm_endpoints": None,
            },
        }
        _atomic_write_yaml(
            self.output_dir / "config.yaml", config.model_dump(mode="json")
        )
        _atomic_write_json(self.output_dir / "run.json", self.run)
        _atomic_write_json(self.cleanup_path, self.cleanup_details)
        self.metrics_path.touch(exist_ok=False)
        self.log_path.touch(exist_ok=False)

    @property
    def native_policy_path_reached(self) -> bool:
        return self._native_policy_path_reached

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message.rstrip()}\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def set_vllm_endpoints(self, *, server_url: str, group_port: int) -> None:
        """记录本 run 实际使用的 vLLM 通信端点，而非 YAML 中的模板端口。"""

        self.run["provenance"]["vllm_endpoints"] = {
            "server_url": server_url,
            "group_port": group_port,
        }
        self.flush_run()

    def begin_group(
        self, prompt: object, rollout_count: int, *, task_id: str
    ) -> list[Path]:
        """开始一个新的 generation batch（单 group）；上一个 batch 随之完成。"""

        if rollout_count < 1:
            raise ValueError("group requires at least one rollout")
        if self.group is not None and self.group["state"] != "completed":
            raise RuntimeError("previous group must complete before beginning the next")
        if self.batch is not None and self.batch["state"] == "running":
            self._finalize_batch(consumed_by=[self.batch["batch_index"] + 1])
        batch_index = 0 if self.batch is None else self.batch["batch_index"] + 1
        self._batch_dir = self.rollouts_root / f"batch-{batch_index:04d}"
        self._group_dir = self._batch_dir / "group-0000"
        self._group_dir.mkdir(parents=True, exist_ok=False)
        rollout_dirs = []
        for index in range(rollout_count):
            path = self._group_dir / f"{index:04d}"
            path.mkdir(exist_ok=False)
            rollout_dirs.append(path)
        self.batch = {
            "batch_index": batch_index,
            "batch_id": f"batch-{batch_index:04d}",
            "state": "running",
            "task_id": task_id,
            "generation_backend": "vllm",
            "global_step_at_generation": batch_index,
            "started_at": _utc_now(),
            "finished_at": None,
            "groups": ["group-0000"],
            "consumed_by_global_steps": [],
        }
        self.group = {
            "group_index": 0,
            "group_id": "group-0000",
            "state": "running",
            "task_id": task_id,
            "prompt_sha256": _payload_sha256(prompt),
            "rollout_dirs": [f"{index:04d}" for index in range(rollout_count)],
            "episode_ids": [],
            "rewards": [],
            "reward_mean": None,
            "reward_std": None,
            "degenerate": None,
            "verification_counts": {"resolved": 0, "unresolved": 0, "not_run": 0},
        }
        self._write_batch()
        self._write_group()
        return rollout_dirs

    def write_rollout(
        self,
        index: int,
        *,
        messages: object,
        trajectory: Trajectory | None,
        patch: str | None,
        verification: Verification | None,
    ) -> None:
        rollout_dir = self._rollout_dir(index)
        _atomic_write_json(rollout_dir / "messages.json", messages)
        if trajectory is not None:
            _atomic_write_json(
                rollout_dir / "trajectory.json", trajectory.model_dump(mode="json")
            )
        if patch is not None and patch.strip():
            _atomic_write_text(rollout_dir / "final_patch.diff", patch)
        if verification is not None:
            _atomic_write_json(
                rollout_dir / "verifier.json", verification.model_dump(mode="json")
            )

    def complete_group(
        self,
        *,
        episode_ids: list[str],
        rewards: list[float | None],
        verifications: list[Verification | None],
    ) -> None:
        if self.group is None:
            raise ValueError("no active group to complete")
        expected = len(self.group["rollout_dirs"])
        if len(episode_ids) != expected or len(rewards) != expected or len(verifications) != expected:
            raise ValueError("completed group requires aligned rollouts")
        # 支持 [0, 1] 内的浮点奖励（layered 模式），仅校验取值范围
        recorded_rewards: list[float | None] = []
        float_rewards: list[float] = []
        for reward in rewards:
            if reward is None:
                recorded_rewards.append(None)
                continue
            if not 0.0 <= reward <= 1.0:
                raise ValueError("group rewards must be within [0, 1]")
            value = float(reward)
            recorded_rewards.append(value)
            float_rewards.append(value)
        if not float_rewards:
            raise ValueError("completed group requires at least one scorable reward")
        resolved = sum(v is not None and v.result == "resolved" for v in verifications)
        unresolved = sum(v is not None and v.result == "unresolved" for v in verifications)
        reward_mean = fmean(float_rewards)
        reward_std = pstdev(float_rewards) if len(float_rewards) > 1 else 0.0
        degenerate = math.isclose(reward_std, 0.0)
        self.group.update(
            {
                "state": "completed",
                "episode_ids": list(episode_ids),
                "rewards": recorded_rewards,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "degenerate": degenerate,
                "verification_counts": {
                    "resolved": resolved,
                    "unresolved": unresolved,
                    "not_run": expected - resolved - unresolved,
                },
            }
        )

        train = self.run["train"]
        result = self.run["results"]["reward"]
        train["groups_generated"] += 1
        train["rollouts_generated"] += expected
        self._all_rewards.extend(float_rewards)
        self._degenerate_groups += int(degenerate)
        self._reward_ema = (
            reward_mean
            if self._reward_ema is None
            else REWARD_EMA_ALPHA * reward_mean
            + (1.0 - REWARD_EMA_ALPHA) * self._reward_ema
        )
        nondegenerate_groups = train["groups_generated"] - self._degenerate_groups
        result.update(
            {
                # successes 只统计 reward == 1.0（完全解决），部分得分不算成功
                "successes": sum(reward == 1.0 for reward in self._all_rewards),
                "attempts": len(self._all_rewards),
                "mean": fmean(self._all_rewards),
                "last_group_mean": reward_mean,
                "ema": self._reward_ema,
                "degenerate_groups": self._degenerate_groups,
                "nondegenerate_groups": nondegenerate_groups,
                "nondegenerate_rate": nondegenerate_groups / train["groups_generated"],
            }
        )
        self._write_group()
        self.flush_run()

    def record_metrics(self, *, step: int, logs: dict[str, Any]) -> bool:
        """保存一个新的 optimizer step；重复或倒序 step 不重复写入。"""

        if step < 1 or step <= self._last_recorded_step:
            return False
        train = self.run["train"]
        reward = self.run["results"]["reward"]
        train["steps_completed"] = max(train["steps_completed"], step)
        num_tokens = _numeric(logs.get("num_tokens"))
        if num_tokens is not None:
            train["tokens_generated"] = int(num_tokens)
        for target, source in LAST_METRIC_KEYS.items():
            value = _numeric(logs.get(source))
            if value is not None:
                train["last_metrics"][target] = value

        group_reward_mean = None if self.group is None else self.group.get("reward_mean")
        group_reward_std = None if self.group is None else self.group.get("reward_std")
        group_degenerate = None if self.group is None else self.group.get("degenerate")
        row: dict[str, Any] = {
            "recorded_at": _utc_now(),
            "step": step,
            "rollouts_cumulative": train["rollouts_generated"],
            "groups_cumulative": train["groups_generated"],
            "train_successes_cumulative": reward["successes"],
            # 累计平均奖励；二元奖励下等价于 resolved 通过率
            "train_pass_rate_cumulative": reward["mean"],
            "reward_mean_group": group_reward_mean,
            "reward_std_group_population": group_reward_std,
            "group_degenerate": group_degenerate,
            "nondegenerate_group_rate_cumulative": reward["nondegenerate_rate"],
            "reward_mean_ema": reward["ema"],
        }
        for target, source in STEP_METRIC_KEYS.items():
            row[target] = _numeric(logs.get(source))
        _append_json_line(self.metrics_path, row)
        self._last_recorded_step = step
        self.flush_run()
        return True

    def sync_trainer_state(
        self, *, global_step: int, log_history: list[dict[str, Any]]
    ) -> None:
        """异常退出前同步 Trainer 的真实进度和最后可用日志。"""

        if global_step < 0:
            raise ValueError("global_step must not be negative")
        latest_for_step = next(
            (
                row
                for row in reversed(log_history)
                if int(_numeric(row.get("step")) or 0) == global_step
            ),
            None,
        )
        if global_step > self._last_recorded_step:
            self.record_metrics(step=global_step, logs=latest_for_step or {})
        self.run["train"]["steps_completed"] = max(
            self.run["train"]["steps_completed"], global_step
        )
        self.refresh_checkpoints()
        self.flush_run()

    def complete_batch(self, global_step: int) -> None:
        if self.batch is None or self.group is None or self.group["state"] != "completed":
            raise RuntimeError("group must complete before its batch can be consumed")
        if global_step < 1:
            raise ValueError("public Trainer global_step must be positive")
        self._finalize_batch(consumed_by=[global_step])
        self.run["train"]["steps_completed"] = max(
            self.run["train"]["steps_completed"], global_step
        )
        self.flush_run()

    def _finalize_batch(self, *, consumed_by: list[int]) -> None:
        if self.batch is None:
            raise RuntimeError("no active batch to finalize")
        self.batch.update(
            {
                "state": "completed",
                "finished_at": _utc_now(),
                "consumed_by_global_steps": list(consumed_by),
            }
        )
        self._write_batch()

    def observe_native_policy_path(self, reached: bool) -> None:
        """保留内部诊断，但不把实现概念写入 run.json。"""

        if not isinstance(reached, bool):
            raise ValueError("native policy path observation must be a bool")
        self._native_policy_path_reached = (
            self._native_policy_path_reached or reached
        )

    def set_model_updated(self, updated: bool | None) -> None:
        self.run["train"]["model_updated"] = updated
        self.flush_run()

    def refresh_checkpoints(self) -> list[str]:
        checkpoints = sorted(
            (
                path.name
                for path in self.output_dir.glob("checkpoint-*")
                if path.is_dir()
            ),
            key=lambda name: int(name.removeprefix("checkpoint-"))
            if name.removeprefix("checkpoint-").isdigit()
            else math.inf,
        )
        self.run["artifacts"]["checkpoints"] = checkpoints
        return checkpoints

    def set_final_model(self, filename: str) -> None:
        self.run["artifacts"]["final_model"] = filename
        self.flush_run()

    def set_plot(self, filename: str) -> None:
        self.run["artifacts"]["plot"] = filename
        self.run["artifacts"]["plot_status"] = "generated"
        self.run["artifacts"].pop("plot_error", None)
        self.flush_run()

    def set_plot_skipped(self, reason: str) -> None:
        self.run["artifacts"]["plot"] = None
        self.run["artifacts"]["plot_status"] = reason
        self.run["artifacts"].pop("plot_error", None)
        self.flush_run()

    def set_plot_error(self, error: BaseException) -> None:
        self.run["artifacts"]["plot"] = None
        self.run["artifacts"]["plot_status"] = "failed"
        self.run["artifacts"]["plot_error"] = (
            f"{type(error).__name__}: {error}"
        )
        self.flush_run()

    def merge_cleanup_events(self, events: list[dict[str, object]]) -> None:
        containers: list[dict[str, Any]] = self.cleanup_details["containers"]
        for event in events:
            name = event.get("container_name")
            scope = event.get("scope")
            task_id = event.get("task_id")
            if (
                not isinstance(name, str)
                or scope not in {"rollout", "verifier"}
                or not isinstance(task_id, str)
                or not task_id
            ):
                raise ValueError("invalid container cleanup event")
            existing = next(
                (item for item in containers if item["container_name"] == name),
                None,
            )
            if existing is None:
                existing = {
                    "episode_id": event.get("episode_id"),
                    "task_id": task_id,
                    "scope": scope,
                    "container_id": event.get("container_id"),
                    "container_name": name,
                    "operations": [],
                    "final_state": "not_created",
                    "residual": False,
                }
                containers.append(existing)
            elif existing["container_id"] is None and event.get("container_id") is not None:
                existing["container_id"] = event.get("container_id")
            operations = event.get("operations") or []
            if not isinstance(operations, list):
                raise ValueError("cleanup event operations must be a list")
            existing["operations"].extend(operations)
            residual = bool(event.get("residual"))
            existing["residual"] = residual
            if residual:
                existing["final_state"] = "residual"
            elif existing["operations"]:
                existing["final_state"] = (
                    "not_found"
                    if existing["operations"][-1].get("result") == "not_found"
                    else "removed"
                )
        self._write_cleanup()

    def set_processes(self, processes: list[dict[str, Any]]) -> None:
        self.cleanup_details["processes"] = processes
        self._write_cleanup()

    def set_runtime_handles(self, handles: list[dict[str, Any]]) -> None:
        self.cleanup_details["runtime_handles"] = handles
        self._write_cleanup()

    def set_gpu_diagnostics(self, diagnostics: list[dict[str, Any]]) -> None:
        self.cleanup_details["gpu_diagnostics"] = diagnostics
        self._write_cleanup()

    def finalize_cleanup(self) -> None:
        cleanup = self.cleanup_details
        resources = (
            cleanup["containers"] + cleanup["processes"] + cleanup["runtime_handles"]
        )
        residuals: list[str] = []
        for container in cleanup["containers"]:
            if container["residual"]:
                residuals.append(
                    container["container_id"] or container["container_name"]
                )
        for process in cleanup["processes"]:
            if process.get("residual") or process.get("final_state") == "residual":
                residuals.append(str(process.get("pid")))
        for handle in cleanup["runtime_handles"]:
            if handle.get("residual") or handle.get("final_state") == "residual":
                residuals.append(str(handle.get("identifier")))
        cleanup["residuals"] = residuals
        cleanup["status"] = "failed" if residuals else "completed"
        cleanup["clean_release"] = not residuals
        if not resources:
            cleanup["clean_release"] = True
        self.run["cleanup"].update(
            {
                "status": cleanup["status"],
                "clean_release": cleanup["clean_release"],
                "residual_count": len(residuals),
            }
        )
        self._write_cleanup()
        self.flush_run()

    def fail(
        self,
        *,
        category: str,
        primary_type: str,
        message: str,
        stage: str,
        interrupted: bool = False,
    ) -> None:
        state: RunState = "interrupted" if interrupted else "failed"
        self._finish_indexes(state)
        self.run["failure"] = {
            "category": "interrupted" if interrupted else category,
            "type": primary_type,
            "message": message,
            "stage": stage,
            "log": "train.log",
        }
        self.run["status"] = state
        self._finish_time()
        self.flush_run()

    def complete(self) -> None:
        if (
            self.run["cleanup"]["status"] != "completed"
            or self.run["cleanup"]["residual_count"]
        ):
            raise RuntimeError(
                "run cannot complete before all known resources are released"
            )
        self.run["status"] = "completed"
        self._finish_time()
        self.flush_run()

    def flush_run(self) -> None:
        _atomic_write_json(self.output_dir / "run.json", self.run)

    def _write_cleanup(self) -> None:
        _atomic_write_json(self.cleanup_path, self.cleanup_details)

    def _finish_time(self) -> None:
        finished_at = datetime.now(UTC)
        self.run["time"]["finished_at"] = _format_utc(finished_at)
        self.run["time"]["duration_seconds"] = round(
            (finished_at - self._started_at).total_seconds(), 3
        )

    def _finish_indexes(self, state: RunState) -> None:
        finished_at = _utc_now()
        if self.group is not None and self.group["state"] == "running":
            self.group["state"] = state
            self._write_group()
        if self.batch is not None and self.batch["state"] == "running":
            self.batch["state"] = state
            self.batch["finished_at"] = finished_at
            self._write_batch()

    def _rollout_dir(self, index: int) -> Path:
        expected = 0 if self.group is None else len(self.group["rollout_dirs"])
        if self._group_dir is None or index not in range(expected):
            raise ValueError(
                f"rollout index must be between zero and {expected - 1}"
            )
        return self._group_dir / f"{index:04d}"

    def _write_batch(self) -> None:
        if self._batch_dir is None or self.batch is None:
            raise RuntimeError("batch has not been allocated")
        _atomic_write_json(self._batch_dir / "batch.json", self.batch)

    def _write_group(self) -> None:
        if self._group_dir is None or self.group is None:
            raise RuntimeError("group has not been allocated")
        _atomic_write_json(self._group_dir / "group.json", self.group)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _append_json_line(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("failed to append metrics row")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now() -> str:
    return _format_utc(datetime.now(UTC))


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _atomic_write_yaml(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
