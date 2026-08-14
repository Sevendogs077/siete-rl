"""SWE-Gym 任务加载器：从锁定数据集与资产构造公开 Sample 和私有 Evaluation。

设计边界：本模块只负责"加载"——深度一致性校验（跨表字段一致、资产哈希、
镜像指纹、离线化正确性）集中在 `siete_rl.qualify`，由 scripts/qualify.sh
单次运行，不在训练启动路径上把守。
"""

from __future__ import annotations

import hashlib
import json
import re
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

OFFLINE_REPLACEMENT = """PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \\
  python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
"""

_INSTALL_BLOCKS = (
    "python -m pip install 'numpy<2'; python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y;",
    "make init",
    "python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y;",
    'python -m pip install --upgrade pip wheel GitPython; python -m pip install "cython<3.0.0" && python -m pip install --no-build-isolation pyyaml==5.4.1; python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; python -m pip install -r tests/requirements.txt || true; python -m pip install -r test-requirements.txt || true; python -m pip install -e ".[tests,dev,all_remotes,all,testing]"; python -m pip install "numpy<=1.20"; python -m pip install "pytest<8";',
    "sed -i '/^git+https:\\/\\/github.com\\/Project-MONAI\\//d' requirements-dev.txt; python -m pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt;python setup.py develop;",
    "python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
    "python -m pip install --no-deps -e .",
    "sed -i 's|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' requirements/dev.txt; { tail -n1 requirements/requirements.txt | grep -q \".\" && echo \"\"; } >> requirements/requirements.txt; echo \"pip==24.0\" >> requirements/requirements.txt;pip install \"pip==24.0\"; pip install -r requirements/dev.txt; pip install -e .;",
    "python -m pip install -e .;",
    "pip install -r requirements/dev.txt; pip install -e .;",
    "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r",
    'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;',
    "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r;",
    "echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
    "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
    "python -m pip install -e .; python -m pip install bokeh_sampledata;",
)
_INSTALL_PATTERNS = tuple(
    re.compile(rf"^{re.escape(block)}(?=\r?$)", re.MULTILINE)
    for block in _INSTALL_BLOCKS
)


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
    matches = [match for pattern in _INSTALL_PATTERNS for match in pattern.finditer(script)]
    if len(matches) != 1:
        raise SWEGymContractError(
            "eval_script must contain exactly one recognized install block; "
            f"found {len(matches)}"
        )
    match = matches[0]
    replacement = OFFLINE_REPLACEMENT
    if "\r\n" in script:
        replacement = replacement.replace("\n", "\r\n")
    return script[: match.start()] + replacement.rstrip("\r\n") + script[match.end() :]


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
