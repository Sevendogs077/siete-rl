from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from siete_rl.asset_generation import generate_task_assets
from siete_rl.config import DatasetConfig
from siete_rl import swegym
from siete_rl.swegym import (
    SWEGymContractError,
    build_training_dataset,
    load_task_context,
    transform_eval_script_offline,
)


TASK_ID = "getmoto__moto-7023"


@pytest.mark.parametrize(
    "raw",
    [
        {"not": "a list"},
        "not-json{",
        [""],
        [1],
    ],
    ids=["mapping", "invalid-json", "empty-test-id", "non-string-test-id"],
)
def test_test_list_parser_rejects_non_string_lists(raw: object) -> None:
    with pytest.raises(SWEGymContractError, match="FAIL_TO_PASS"):
        swegym._load_test_list({"FAIL_TO_PASS": raw}, "FAIL_TO_PASS", TASK_ID)


@pytest.mark.parametrize("raw", ['["tests/test_a.py::test_fix"]', ["tests/test_a.py::test_fix"]])
def test_test_list_parser_accepts_json_and_materialized_lists(raw: object) -> None:
    assert swegym._load_test_list({"FAIL_TO_PASS": raw}, "FAIL_TO_PASS", TASK_ID) == [
        "tests/test_a.py::test_fix"
    ]


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


@pytest.mark.parametrize(
    "script",
    ["echo no-init\n", "make init\nmake init\n"],
    ids=["missing-install", "multiple-installs"],
)
def test_offline_transform_requires_exactly_one_install_line(script: str) -> None:
    with pytest.raises(SWEGymContractError, match="exactly one"):
        transform_eval_script_offline(script)


REAL_INSTALL_BLOCKS = [
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
]


@pytest.mark.parametrize("install_block", REAL_INSTALL_BLOCKS)
def test_offline_transform_replaces_every_real_install_block(
    install_block: str,
) -> None:
    test_command = "pytest -rA tests/test_real.py\n"
    script = f"cd /testbed\n{install_block}\n{test_command}"
    offline = transform_eval_script_offline(script)
    assert "PIP_NO_INDEX=1" in offline
    assert install_block not in offline
    assert offline.endswith(test_command)


def test_runtime_loader_selects_stage_and_preserves_row_order(tmp_path: Path) -> None:
    rows = []
    for position, (task_id, stage) in enumerate(
        (("stage1-a", 1), ("stage1-b", 1), ("stage2-a", 2))
    ):
        row = {
            "instance_id": task_id,
            "repo": "owner/repo",
            "base_commit": f"{position + 1:040x}",
            "version": "fixture",
            "problem_statement": f"problem {position}",
            "patch": "",
            "test_patch": "",
            "FAIL_TO_PASS": ["test_a"],
            "PASS_TO_PASS": [],
            "eval_script": "make init\npytest -q\n",
            "stage": stage,
            "stage_position": position if stage == 1 else 0,
        }
        rows.append(row)
        generate_task_assets(
            row, tmp_path / "assets", image_id="sha256:" + "1" * 64
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")

    from siete_rl.config import load_config

    config, _, _ = load_config(Path(__file__).resolve().parents[2] / "configs/stage1.yaml")
    config = config.model_copy(
        update={
            "dataset": DatasetConfig(
                train_path=str(tmp_path / "train.parquet"),
                tasks_dir=str(tmp_path / "assets"),
                stage=1,
            )
        }
    )
    context = load_task_context(config, tmp_path)
    dataset = build_training_dataset(context)

    assert list(context) == ["stage1-a", "stage1-b"]
    assert dataset["task_id"] == ["stage1-a", "stage1-b"]
