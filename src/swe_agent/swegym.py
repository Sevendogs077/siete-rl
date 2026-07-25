"""SWE-Gym 任务加载器：从锁定数据集与资产构造公开 Sample 和私有 Evaluation。

设计边界：本模块只负责"加载"——深度一致性校验（跨表字段一致、资产哈希、
镜像指纹、离线化正确性）集中在 `swe_agent.qualify`，由 scripts/qualify.sh
单次运行，不在训练启动路径上把守。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from datasets import Dataset

from swe_agent.config import ProjectConfig
from swe_agent.models import Environment, Evaluation, Sample, Task
from swe_agent.prompts import build_prompt


OFFICIAL_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SUBSET_REVISION = "3f22e68f673027edbaebe3424e4c20ae580563fd"
COMPARE_FIELDS = (
    "repo",
    "base_commit",
    "version",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)

OFFLINE_REPLACEMENT = """# Offline replacement for `make init`.
# 镜像中已经预装运行和测试依赖；这里只重新绑定当前 /testbed 源码。
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \\
  python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
"""


class SWEGymContractError(RuntimeError):
    """数据、资产或任务身份不满足加载要求。"""


TaskContext = Mapping[str, tuple[Sample, Evaluation]]


def load_qualified_instance(config: ProjectConfig, project_root: Path) -> tuple[Sample, Evaluation]:
    """加载一个任务；只保留加载所必需的完整性检查。"""

    task_id = config.dataset.task_id
    official_path = _resolve(project_root, config.dataset.official_path)
    subset_path = _resolve(project_root, config.dataset.subset_path)
    assets_dir = _resolve(project_root, config.dataset.assets_dir)

    official = _read_exact_row(official_path, task_id)
    subset = _read_exact_row(subset_path, task_id)
    eval_script = subset.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script.strip():
        raise SWEGymContractError("derived dataset is missing eval_script")

    offline_path = assets_dir / "eval_script.offline.sh"
    try:
        offline = offline_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SWEGymContractError(f"missing offline evaluator asset: {offline_path}") from exc

    task = Task(
        task_id=task_id,
        repo_name=str(official["repo"]),
        base_commit=str(official["base_commit"]),
        problem_statement=str(official["problem_statement"]),
    )
    environment = Environment(
        environment_id=f"swegym:{task_id}",
        task_id=task_id,
        image_name=config.docker.image,
        expected_image_id=config.docker.expected_image_id,
        expected_registry_digest=config.docker.expected_registry_digest,
        workdir="/testbed",
        cpus=config.docker.cpus,
        memory=config.docker.memory,
        pids_limit=config.docker.pids_limit,
        exec_timeout_sec=config.docker.exec_timeout_sec,
        verifier_timeout_sec=config.docker.verifier_timeout_sec,
    )
    return Sample(task=task, environment=environment), Evaluation(offline_eval_script=offline)


def load_task_context(config: ProjectConfig, project_root: Path) -> TaskContext:
    sample, evaluation = load_qualified_instance(config, project_root)
    return MappingProxyType({sample.task.task_id: (sample, evaluation)})


def build_training_dataset(sample: Sample) -> Dataset:
    """构造只含公开 task_id 与 prompt 的单任务 TRL Dataset。"""

    return Dataset.from_list(
        [{"task_id": sample.task.task_id, "prompt": build_prompt(sample.task)}]
    )


def transform_eval_script_offline(script: str) -> str:
    lines = script.splitlines(keepends=True)
    indexes = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == "make init"]
    if len(indexes) != 1:
        raise SWEGymContractError(
            f"eval_script must contain exactly one standalone make init; found {len(indexes)}"
        )
    replacement = OFFLINE_REPLACEMENT
    if lines[indexes[0]].endswith("\r\n"):
        replacement = replacement.replace("\n", "\r\n")
    lines[indexes[0] : indexes[0] + 1] = [replacement]
    return "".join(lines)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_exact_row(path: Path, instance_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise SWEGymContractError(f"Parquet file does not exist: {path}")
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path, filters=[("instance_id", "=", instance_id)]).to_pylist()
    except Exception as exc:
        raise SWEGymContractError(f"failed to read Parquet file {path}: {exc}") from exc
    if len(rows) != 1:
        raise SWEGymContractError(
            f"{path} must contain exactly one instance_id={instance_id} row; found {len(rows)}"
        )
    if not isinstance(rows[0], dict):
        raise SWEGymContractError(f"Parquet row is not an object: {path}")
    return rows[0]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SWEGymContractError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SWEGymContractError(f"JSON must contain a top-level object: {path}")
    return value


def _require_sha256(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise SWEGymContractError(f"manifest SHA-256 is invalid for {path.name}")
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SWEGymContractError(f"failed to hash {path}: {exc}") from exc
    if actual != expected:
        raise SWEGymContractError(f"SHA-256 mismatch for {path}")


def _normalize(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value
