"""SWE-Gym 任务加载器：从锁定数据集与资产构造公开 Sample 和私有 Evaluation。

设计边界：本模块只负责"加载"——深度一致性校验（跨表字段一致、资产哈希、
镜像指纹、离线化正确性）集中在 `siete_rl.qualify`，由 scripts/qualify.sh
单次运行，不在训练启动路径上把守。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from datasets import Dataset

from siete_rl.config import ProjectConfig
from siete_rl.models import Environment, Evaluation, Sample, Task
from siete_rl.prompts import build_prompt


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


def select_task_ids(config: ProjectConfig, project_root: Path) -> list[str]:
    """按配置从资产目录选择确定且非空的任务列表。"""

    tasks_dir = _resolve(project_root, config.dataset.tasks_dir)
    if config.dataset.task_ids is not None:
        selected = sorted(config.dataset.task_ids)
    else:
        try:
            selected = sorted(path.name for path in tasks_dir.iterdir() if path.is_dir())
        except OSError as exc:
            raise SWEGymContractError(f"failed to list task assets under {tasks_dir}: {exc}") from exc
    if config.dataset.max_tasks is not None:
        selected = selected[: config.dataset.max_tasks]
    if not selected:
        raise SWEGymContractError(f"no task assets found under {tasks_dir}")
    return selected


def load_task_instance(
    config: ProjectConfig, project_root: Path, task_id: str
) -> tuple[Sample, Evaluation]:
    """从一个任务的锁定数据和自包含资产构造运行时上下文。"""

    official_path = _resolve(project_root, config.dataset.official_path)
    subset_path = _resolve(project_root, config.dataset.subset_path)

    official = _read_exact_row(official_path, task_id)
    subset = _read_exact_row(subset_path, task_id)
    eval_script = subset.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script.strip():
        raise SWEGymContractError(f"{task_id}: derived dataset is missing eval_script")

    assets_dir = _resolve(project_root, config.dataset.tasks_dir) / task_id
    offline_path = assets_dir / "eval_script.offline.sh"
    try:
        offline = offline_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SWEGymContractError(f"missing offline evaluator asset: {offline_path}") from exc
    manifest = _read_object(assets_dir / "manifest.json")

    task = Task(
        task_id=task_id,
        repo_name=str(official["repo"]),
        base_commit=str(official["base_commit"]),
        problem_statement=str(official["problem_statement"]),
    )
    environment = Environment(
        environment_id=f"swegym:{task_id}",
        task_id=task_id,
        image_name=manifest["image_name"],
        expected_image_id=manifest["expected_image_id"],
        expected_registry_digest=manifest["expected_registry_digest"],
        workdir="/testbed",
        cpus=config.docker.cpus,
        memory=config.docker.memory,
        pids_limit=config.docker.pids_limit,
        exec_timeout_sec=config.docker.exec_timeout_sec,
        verifier_timeout_sec=config.docker.verifier_timeout_sec,
    )
    return Sample(task=task, environment=environment), Evaluation(
        offline_eval_script=offline,
        fail_to_pass=_load_test_list(official, "FAIL_TO_PASS", task_id),
        pass_to_pass=_load_test_list(official, "PASS_TO_PASS", task_id),
    )


def load_task_context(config: ProjectConfig, project_root: Path) -> TaskContext:
    return MappingProxyType(
        {
            task_id: load_task_instance(config, project_root, task_id)
            for task_id in select_task_ids(config, project_root)
        }
    )


def build_training_dataset(context: TaskContext) -> Dataset:
    """构造每个选中任务一行的公开 TRL Dataset。"""

    return Dataset.from_list(
        [
            {"task_id": task_id, "prompt": build_prompt(sample.task)}
            for task_id, (sample, _) in context.items()
        ]
    )


def transform_eval_script_offline(script: str) -> str:
    lines = script.splitlines(keepends=True)
    make_init = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == "make init"]
    pip_install = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n").startswith("python -m pip install") and "pip install -e ." in line
    ]
    hits = len(make_init) + len(pip_install)
    if hits != 1:
        raise SWEGymContractError(
            "eval_script must contain exactly one install line "
            "(standalone make init or python -m pip install); "
            f"found {hits} (make init: {len(make_init)}, pip install: {len(pip_install)})"
        )
    index = make_init[0] if make_init else pip_install[0]
    replacement = OFFLINE_REPLACEMENT
    if lines[index].endswith("\r\n"):
        replacement = replacement.replace("\n", "\r\n")
    lines[index : index + 1] = [replacement]
    return "".join(lines)


def _load_test_list(row: dict[str, Any], field: str, task_id: str) -> list[str]:
    """解析官方行中的测试清单字段：JSON 字符串或已解析的字符串列表。"""

    raw = row[field]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SWEGymContractError(f"{task_id}: {field} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
        raise SWEGymContractError(f"{task_id}: {field} must be a JSON list of test ids")
    return list(raw)


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
