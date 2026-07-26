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

from swe_agent.config import ProjectConfig
from swe_agent.models import Trajectory, Verification


RunState = Literal["running", "completed", "failed", "interrupted"]
IndexState = Literal["running", "completed", "failed", "interrupted"]


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
)


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
    ) -> None:
        self.config = config
        self.run_id = run_id or config.output.run_id or generate_run_id()
        self.output_dir = (Path(config.output.output_root) / self.run_id).resolve()
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(exist_ok=False)
        self.rollouts_root = self.output_dir / "rollouts"
        self._batch_dir: Path | None = None
        self._group_dir: Path | None = None
        self.batch: dict[str, Any] | None = None
        self.group: dict[str, Any] | None = None
        self._all_rewards: list[int] = []
        self._degenerate_groups = 0

        started_at = _utc_now()
        self.run: dict[str, Any] = {
            "schema_version": "1",
            "identity": {
                "run_id": self.run_id,
                "output_dir": self.output_dir.as_posix(),
                "config_file": "config.yaml",
            },
            "provenance": {
                "started_at": started_at,
                "finished_at": None,
                "code_commit": code_commit,
                "code_dirty": code_dirty,
                "dependency_versions": dependency_versions or installed_dependency_versions(),
                "model_path": config.model.model_path,
                "resolved_model_path": Path(config.model.model_path).resolve().as_posix(),
                "model_revision": model_revision,
                "generation_backend": "vllm",
                "official_dataset_revision": config.dataset.official_revision,
                "subset_dataset_revision": config.dataset.subset_revision,
                "image_platform": config.docker.platform,
                "seed": seed,
            },
            "lifecycle": {"state": "running"},
            "failure": None,
            "training": {
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
            },
            "cleanup": {
                "state": "pending",
                "clean_release": None,
                "residuals": [],
                "containers": [],
                "processes": [],
                "runtime_handles": [],
                "gpu_diagnostics": [],
            },
        }
        _atomic_write_yaml(
            self.output_dir / "config.yaml", config.model_dump(mode="json")
        )
        _atomic_write_json(self.output_dir / "run.json", self.run)
        self.log_path = self.output_dir / config.output.train_log
        self.log_path.touch(exist_ok=False)

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message.rstrip()}\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

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
            "schema_version": "1",
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
            "schema_version": "1",
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
        rewards: list[float],
        verifications: list[Verification | None],
    ) -> None:
        if self.group is None:
            raise ValueError("no active group to complete")
        expected = len(self.group["rollout_dirs"])
        if len(episode_ids) != expected or len(rewards) != expected or len(verifications) != expected:
            raise ValueError("completed group requires aligned rollouts")
        integer_rewards = []
        for reward in rewards:
            if reward not in (0, 0.0, 1, 1.0):
                raise ValueError("binary group rewards must be zero or one")
            integer_rewards.append(int(reward))
        resolved = sum(v is not None and v.result == "resolved" for v in verifications)
        unresolved = sum(v is not None and v.result == "unresolved" for v in verifications)
        reward_mean = fmean(integer_rewards)
        reward_std = pstdev(integer_rewards) if len(integer_rewards) > 1 else 0.0
        self.group.update(
            {
                "state": "completed",
                "episode_ids": list(episode_ids),
                "rewards": integer_rewards,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "degenerate": math.isclose(reward_std, 0.0),
                "verification_counts": {
                    "resolved": resolved,
                    "unresolved": unresolved,
                    "not_run": expected - resolved - unresolved,
                },
            }
        )
        training = self.run["training"]
        training["groups_generated"] += 1
        training["rollouts_generated"] += expected
        self._all_rewards.extend(integer_rewards)
        training["reward_mean"] = fmean(self._all_rewards)
        training["reward_std"] = (
            pstdev(self._all_rewards) if len(self._all_rewards) > 1 else 0.0
        )
        self._degenerate_groups += int(math.isclose(reward_std, 0.0))
        training["frac_reward_zero_std"] = self._degenerate_groups / training["groups_generated"]
        self.run["training"]["observations"]["reward_degenerate"] = (
            self._degenerate_groups == training["groups_generated"]
        )
        self._write_group()
        self.flush_run()

    def complete_batch(self, global_step: int) -> None:
        if self.batch is None or self.group is None or self.group["state"] != "completed":
            raise RuntimeError("group must complete before its batch can be consumed")
        if global_step < 1:
            raise ValueError("public Trainer global_step must be positive")
        self._finalize_batch(consumed_by=[global_step])
        self.run["training"]["global_step"] = global_step
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

    def update_training(self, **values: Any) -> None:
        unknown = set(values) - set(self.run["training"])
        if unknown:
            raise ValueError("unknown run training fields: " + ", ".join(sorted(unknown)))
        self.run["training"].update(values)
        self.flush_run()

    def observe_native_policy_path(self, reached: bool) -> None:
        """累积每个已完成 group 的原生策略闭环观察结果。"""

        if not isinstance(reached, bool):
            raise ValueError("native policy path observation must be a bool")
        training = self.run["training"]
        if training["native_policy_path_reached"] is True:
            return
        training["native_policy_path_reached"] = reached
        self.flush_run()

    def update_observations(self, **values: bool | None) -> None:
        observations = self.run["training"]["observations"]
        unknown = set(values) - set(observations)
        if unknown:
            raise ValueError("unknown run observation fields: " + ", ".join(sorted(unknown)))
        observations.update(values)
        self.flush_run()

    def merge_cleanup_events(self, events: list[dict[str, object]]) -> None:
        containers: list[dict[str, Any]] = self.run["cleanup"]["containers"]
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
            existing = next((item for item in containers if item["container_name"] == name), None)
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
        self.flush_run()

    def set_processes(self, processes: list[dict[str, Any]]) -> None:
        self.run["cleanup"]["processes"] = processes
        self.flush_run()

    def set_runtime_handles(self, handles: list[dict[str, Any]]) -> None:
        self.run["cleanup"]["runtime_handles"] = handles
        self.flush_run()

    def set_gpu_diagnostics(self, diagnostics: list[dict[str, Any]]) -> None:
        self.run["cleanup"]["gpu_diagnostics"] = diagnostics
        self.flush_run()

    def finalize_cleanup(self) -> None:
        cleanup = self.run["cleanup"]
        resources = (
            cleanup["containers"] + cleanup["processes"] + cleanup["runtime_handles"]
        )
        residuals: list[str] = []
        for container in cleanup["containers"]:
            if container["residual"]:
                residuals.append(container["container_id"] or container["container_name"])
        for process in cleanup["processes"]:
            if process.get("residual") or process.get("final_state") == "residual":
                residuals.append(str(process.get("pid")))
        for handle in cleanup["runtime_handles"]:
            if handle.get("residual") or handle.get("final_state") == "residual":
                residuals.append(str(handle.get("identifier")))
        cleanup["residuals"] = residuals
        cleanup["state"] = "failed" if residuals else "completed"
        cleanup["clean_release"] = not residuals
        if not resources:
            cleanup["clean_release"] = True
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
            "primary_type": primary_type,
            "message": message,
            "stage": stage,
            "traceback_log_ref": "train.log",
        }
        self.run["lifecycle"]["state"] = state
        self.run["provenance"]["finished_at"] = _utc_now()
        self.flush_run()

    def complete(self) -> None:
        if self.run["cleanup"]["state"] != "completed" or self.run["cleanup"]["residuals"]:
            raise RuntimeError("run cannot complete before all known resources are released")
        self.run["lifecycle"]["state"] = "completed"
        self.run["provenance"]["finished_at"] = _utc_now()
        self.flush_run()

    def flush_run(self) -> None:
        _atomic_write_json(self.output_dir / "run.json", self.run)

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
            raise ValueError(f"rollout index must be between zero and {expected - 1}")
        return self._group_dir / f"{index:04d}"

    def _write_batch(self) -> None:
        if self._batch_dir is None or self.batch is None:
            raise RuntimeError("batch has not been allocated")
        _atomic_write_json(self._batch_dir / "batch.json", self.batch)

    def _write_group(self) -> None:
        if self._group_dir is None or self.group is None:
            raise RuntimeError("group has not been allocated")
        _atomic_write_json(self._group_dir / "group.json", self.group)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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
