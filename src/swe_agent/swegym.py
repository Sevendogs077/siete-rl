"""固定 SWE-Gym 实例、锁定数据和私有 evaluator 的 fail-closed loader。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from datasets import Dataset

from swe_agent.config import FIXED_IMAGE, FIXED_IMAGE_ID, FIXED_TASK_ID, ProjectConfig
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
    """数据、资产或固定任务身份不满足实施合同。"""


TaskContext = Mapping[str, tuple[Sample, Evaluation]]


def load_qualified_instance(config: ProjectConfig, project_root: Path) -> tuple[Sample, Evaluation]:
    """资格化两份锁定数据与资产，只返回公开 Sample 和精简私有 Evaluation。"""

    if config.dataset.task_id != FIXED_TASK_ID:
        raise SWEGymContractError(f"only {FIXED_TASK_ID} is allowed")
    if config.dataset.official_revision != OFFICIAL_REVISION:
        raise SWEGymContractError("official dataset revision does not match the qualified revision")
    if config.dataset.subset_revision != SUBSET_REVISION:
        raise SWEGymContractError("subset dataset revision does not match the qualified revision")

    official_path = Path(config.dataset.official_path)
    subset_path = Path(config.dataset.subset_path)
    assets_dir = Path(config.dataset.assets_dir)
    expected_official = (
        project_root
        / "data/swegym/SWE-Gym__SWE-Gym"
        / OFFICIAL_REVISION
        / "data/train-00000-of-00001.parquet"
    )
    expected_subset = (
        project_root
        / "data/swegym/SumanthRH__SWE-Gym-Subset"
        / SUBSET_REVISION
        / "data/train-00000-of-00001.parquet"
    )
    expected_assets = project_root / "assets/swegym" / FIXED_TASK_ID
    if official_path != expected_official or subset_path != expected_subset or assets_dir != expected_assets:
        raise SWEGymContractError("dataset and asset paths must use the project-owned qualified locations")

    manifest = _read_object(assets_dir / "manifest.json")
    _validate_manifest(manifest, config, official_path, subset_path, assets_dir)
    official = _read_exact_row(official_path, FIXED_TASK_ID)
    subset = _read_exact_row(subset_path, FIXED_TASK_ID)
    for field in COMPARE_FIELDS:
        if _normalize(official.get(field)) != _normalize(subset.get(field)):
            raise SWEGymContractError(f"dataset fields do not match: {field}")

    selected = _read_object(assets_dir / "selected_instance.json")
    for field in ("instance_id", *COMPARE_FIELDS):
        expected = FIXED_TASK_ID if field == "instance_id" else official.get(field)
        if _normalize(selected.get(field)) != _normalize(expected):
            raise SWEGymContractError(f"selected_instance field mismatch: {field}")
    eval_script = subset.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script.strip():
        raise SWEGymContractError("derived dataset is missing eval_script")
    if selected.get("eval_script") != eval_script:
        raise SWEGymContractError("selected eval_script does not match the subset row")
    if selected.get("image_name") != config.docker.image:
        raise SWEGymContractError("selected image does not match configuration")

    original = (assets_dir / "eval_script.sh").read_text(encoding="utf-8")
    offline = (assets_dir / "eval_script.offline.sh").read_text(encoding="utf-8")
    gold = (assets_dir / "gold.patch").read_text(encoding="utf-8")
    test_patch = (assets_dir / "test.patch").read_text(encoding="utf-8")
    if original != eval_script:
        raise SWEGymContractError("eval_script.sh does not match the dataset")
    if transform_eval_script_offline(original) != offline:
        raise SWEGymContractError("offline evaluator is not the single allowed make init replacement")
    if gold != official.get("patch") or test_patch != official.get("test_patch"):
        raise SWEGymContractError("qualified patch assets do not match the official row")

    task = Task(
        task_id=FIXED_TASK_ID,
        repo_name=str(official["repo"]),
        base_commit=str(official["base_commit"]),
        problem_statement=str(official["problem_statement"]),
    )
    environment = Environment(
        environment_id=f"swegym:{FIXED_TASK_ID}",
        task_id=FIXED_TASK_ID,
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


def _validate_manifest(
    manifest: dict[str, Any],
    config: ProjectConfig,
    official_path: Path,
    subset_path: Path,
    assets_dir: Path,
) -> None:
    expected_identity = {
        "schema_version": "1",
        "task_id": FIXED_TASK_ID,
        "repo_name": "getmoto/moto",
        "base_commit": "447710c6a68e7d5ea7ad6d7df93c663de32ac7f1",
        "image_name": FIXED_IMAGE,
        "expected_image_id": FIXED_IMAGE_ID,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise SWEGymContractError(f"manifest identity mismatch: {field}")
    if config.docker.image != manifest["image_name"] or config.docker.expected_image_id != manifest["expected_image_id"]:
        raise SWEGymContractError("manifest image identity does not match configuration")

    files = manifest.get("files")
    datasets = manifest.get("datasets")
    if not isinstance(files, dict) or set(files) != {
        "selected_instance.json",
        "eval_script.sh",
        "eval_script.offline.sh",
        "gold.patch",
        "test.patch",
    }:
        raise SWEGymContractError("manifest asset file set is invalid")
    for filename, expected_hash in files.items():
        _require_sha256(assets_dir / filename, expected_hash)
    if not isinstance(datasets, dict) or set(datasets) != {"official", "subset"}:
        raise SWEGymContractError("manifest dataset set is invalid")
    for name, path, revision in (
        ("official", official_path, OFFICIAL_REVISION),
        ("subset", subset_path, SUBSET_REVISION),
    ):
        entry = datasets.get(name)
        if not isinstance(entry, dict) or entry.get("revision") != revision:
            raise SWEGymContractError(f"manifest dataset revision mismatch: {name}")
        _require_sha256(path, entry.get("sha256"))


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
