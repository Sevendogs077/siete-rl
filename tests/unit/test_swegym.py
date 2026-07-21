from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swe_agent.config import load_config
from swe_agent.models import Evaluation
from swe_agent import swegym
from swe_agent.swegym import (
    SWEGymContractError,
    build_training_dataset,
    load_qualified_instance,
    load_task_context,
    transform_eval_script_offline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


@pytest.fixture(scope="module")
def loaded():
    config, project_root, _ = load_config(CONFIG_PATH)
    sample, evaluation = load_qualified_instance(config, project_root)
    return config, project_root, sample, evaluation


def test_real_two_table_assets_and_runtime_boundary(loaded) -> None:
    config, _, sample, evaluation = loaded
    assert sample.task.task_id == "getmoto__moto-7023"
    assert sample.task.repo_name == "getmoto/moto"
    assert sample.task.base_commit == "447710c6a68e7d5ea7ad6d7df93c663de32ac7f1"
    assert sample.environment.image_name == config.docker.image
    assert sample.environment.expected_image_id == config.docker.expected_image_id
    assert set(Evaluation.model_fields) == {"offline_eval_script"}
    assert "PIP_NO_INDEX=1" in evaluation.offline_eval_script


def test_task_context_is_read_only_and_keyed_by_task_id(loaded) -> None:
    config, project_root, sample, evaluation = loaded
    context = load_task_context(config, project_root)
    assert context[sample.task.task_id] == (sample, evaluation)
    with pytest.raises(TypeError):
        context["other"] = (sample, evaluation)  # type: ignore[index]


def test_dataset_and_prompt_contain_only_public_fields(loaded) -> None:
    _, _, sample, evaluation = loaded
    dataset = build_training_dataset(sample)
    assert dataset.column_names == ["task_id", "prompt"]
    assert len(dataset) == 1
    public_payload = json.dumps(
        {"sample": sample.model_dump(mode="json"), "row": dataset[0]},
        ensure_ascii=False,
    )
    assets = PROJECT_ROOT / "assets/swegym/getmoto__moto-7023"
    for secret in (
        (assets / "gold.patch").read_text(encoding="utf-8"),
        (assets / "test.patch").read_text(encoding="utf-8"),
        (assets / "eval_script.sh").read_text(encoding="utf-8"),
        evaluation.offline_eval_script,
    ):
        assert secret not in public_payload
    assert "Qwen XML" not in public_payload


def test_revision_and_cross_table_mismatch_fail_closed(loaded, monkeypatch: pytest.MonkeyPatch) -> None:
    config, project_root, _, _ = loaded
    wrong_revision = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"official_revision": "0" * 40})
        }
    )
    with pytest.raises(SWEGymContractError, match="official dataset revision"):
        load_qualified_instance(wrong_revision, project_root)

    original = swegym._read_exact_row
    calls = 0

    def mismatched(path: Path, instance_id: str):
        nonlocal calls
        calls += 1
        row = dict(original(path, instance_id))
        if calls == 2:
            row["base_commit"] = "0" * 40
        return row

    monkeypatch.setattr(swegym, "_read_exact_row", mismatched)
    with pytest.raises(SWEGymContractError, match="base_commit"):
        load_qualified_instance(config, project_root)


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


def test_offline_transform_is_exact_and_fail_closed(loaded) -> None:
    config, _, _, _ = loaded
    assets = Path(config.dataset.assets_dir)
    original = (assets / "eval_script.sh").read_text(encoding="utf-8")
    offline = (assets / "eval_script.offline.sh").read_text(encoding="utf-8")
    assert transform_eval_script_offline(original) == offline
    with pytest.raises(SWEGymContractError, match="exactly one"):
        transform_eval_script_offline("echo no-init\n")
    with pytest.raises(SWEGymContractError, match="exactly one"):
        transform_eval_script_offline("make init\nmake init\n")
