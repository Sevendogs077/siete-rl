from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from siete_rl import swegym
from siete_rl.config import load_config
from siete_rl.models import Evaluation
from siete_rl.swegym import (
    SWEGymContractError,
    build_training_dataset,
    load_task_context,
    load_task_instance,
    transform_eval_script_offline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_openhands_7b_lora.yaml"
TASK_ID = "getmoto__moto-7023"


@pytest.fixture(scope="module")
def loaded():
    config, project_root, _ = load_config(CONFIG_PATH)
    sample, evaluation = load_task_instance(config, project_root, TASK_ID)
    return config, project_root, sample, evaluation


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_real_two_table_assets_and_runtime_boundary(loaded) -> None:
    _, _, sample, evaluation = loaded
    assert sample.task.task_id == TASK_ID
    assert sample.task.repo_name == "getmoto/moto"
    assert sample.task.base_commit == "447710c6a68e7d5ea7ad6d7df93c663de32ac7f1"
    assert sample.environment.image_name.endswith("getmoto_s_moto-7023:latest")
    assert sample.environment.expected_image_id.startswith("sha256:")
    assert set(Evaluation.model_fields) == {"offline_eval_script", "fail_to_pass", "pass_to_pass"}
    assert "PIP_NO_INDEX=1" in evaluation.offline_eval_script


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_task_context_covers_all_selected_tasks_and_is_read_only(loaded) -> None:
    config, project_root, sample, evaluation = loaded
    context = load_task_context(config, project_root)
    assert len(context) >= 2
    assert context[TASK_ID] == (sample, evaluation)
    assert context[TASK_ID][0].environment.image_name.endswith("getmoto_s_moto-7023:latest")
    assert "PIP_NO_INDEX=1" in context[TASK_ID][1].offline_eval_script
    assert len(build_training_dataset(context)) == len(context)
    with pytest.raises(TypeError):
        context["other"] = (sample, evaluation)  # type: ignore[index]


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_dataset_and_prompt_contain_only_public_fields(loaded) -> None:
    config, project_root, sample, evaluation = loaded
    dataset = build_training_dataset(load_task_context(config, project_root))
    assert dataset.column_names == ["task_id", "prompt"]
    assert len(dataset) >= 2
    public_payload = json.dumps(
        {"sample": sample.model_dump(mode="json"), "rows": list(dataset)},
        ensure_ascii=False,
    )
    assets = Path(config.dataset.tasks_dir) / TASK_ID
    for secret in (
        (assets / "gold.patch").read_text(encoding="utf-8"),
        (assets / "test.patch").read_text(encoding="utf-8"),
        (assets / "eval_script.sh").read_text(encoding="utf-8"),
        evaluation.offline_eval_script,
    ):
        assert secret not in public_payload
    assert "Qwen XML" not in public_payload


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_load_task_instance_carries_test_lists(loaded) -> None:
    _, _, _, evaluation = loaded
    assert evaluation.fail_to_pass == [
        "tests/test_lakeformation/test_lakeformation.py::test_deregister_resource"
    ]
    assert evaluation.pass_to_pass == [
        "tests/test_lakeformation/test_lakeformation.py::test_list_data_cells_filter",
        "tests/test_lakeformation/test_lakeformation.py::test_revoke_permissions",
        "tests/test_lakeformation/test_lakeformation.py::test_list_resources",
        "tests/test_lakeformation/test_lakeformation.py::test_list_permissions",
        "tests/test_lakeformation/test_lakeformation.py::test_describe_resource",
        "tests/test_lakeformation/test_lakeformation.py::test_batch_revoke_permissions",
        "tests/test_lakeformation/test_lakeformation.py::test_register_resource",
        "tests/test_lakeformation/test_lakeformation.py::test_data_lake_settings",
    ]


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_load_task_instance_rejects_non_list_test_field(loaded, monkeypatch: pytest.MonkeyPatch) -> None:
    # official 行的 FAIL_TO_PASS 不是 JSON 字符串列表时必须拒绝。
    config, project_root, _, _ = loaded
    real_read = swegym._read_exact_row

    def fake_read(path: Path, instance_id: str):
        row = real_read(path, instance_id)
        if "FAIL_TO_PASS" in row:
            row = dict(row, FAIL_TO_PASS={"not": "a list"})
        return row

    monkeypatch.setattr(swegym, "_read_exact_row", fake_read)
    with pytest.raises(SWEGymContractError, match="FAIL_TO_PASS"):
        load_task_instance(config, project_root, TASK_ID)


# 依赖私有 data/assets（loaded fixture 读取真实 parquet 资产）。
@pytest.mark.external_assets
def test_load_task_instance_rejects_invalid_json_test_field(loaded, monkeypatch: pytest.MonkeyPatch) -> None:
    # FAIL_TO_PASS 是字符串但不是合法 JSON 时也必须拒绝。
    config, project_root, _, _ = loaded
    real_read = swegym._read_exact_row

    def fake_read(path: Path, instance_id: str):
        row = real_read(path, instance_id)
        if "FAIL_TO_PASS" in row:
            row = dict(row, FAIL_TO_PASS="not-json{")
        return row

    monkeypatch.setattr(swegym, "_read_exact_row", fake_read)
    with pytest.raises(SWEGymContractError, match="FAIL_TO_PASS"):
        load_task_instance(config, project_root, TASK_ID)


def test_exactly_one_parquet_row_is_required(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"instance_id": "x", "value": 1},
                {"instance_id": "x", "value": 2},
            ]
        ),
        path,
    )
    with pytest.raises(SWEGymContractError, match="exactly one"):
        swegym._read_exact_row(path, "x")


# 依赖私有 data/assets（读取 assets 下真实 eval_script）。
@pytest.mark.external_assets
def test_offline_transform_is_exact_and_fail_closed(loaded) -> None:
    config, _, _, _ = loaded
    assets = Path(config.dataset.tasks_dir) / TASK_ID
    original = (assets / "eval_script.sh").read_text(encoding="utf-8")
    offline = (assets / "eval_script.offline.sh").read_text(encoding="utf-8")
    assert transform_eval_script_offline(original) == offline
    with pytest.raises(SWEGymContractError, match="exactly one"):
        transform_eval_script_offline("echo no-init\n")
    with pytest.raises(SWEGymContractError, match="exactly one"):
        transform_eval_script_offline("make init\nmake init\n")


@pytest.mark.parametrize(
    "script",
    [
        "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r\npytest -q\n",
        "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r;\npytest -q\n",
        "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r\npytest -q\n",
    ],
)
def test_offline_transform_replaces_mypy_pip_install_line(script: str) -> None:
    offline = transform_eval_script_offline(script)
    assert "PIP_NO_INDEX=1" in offline
    assert "pip install -r test-requirements.txt" not in offline
    # OFFLINE_REPLACEMENT 注释里提及 `make init`；这里断言不存在独立的 make init 行。
    assert all(line != "make init" for line in offline.splitlines())
