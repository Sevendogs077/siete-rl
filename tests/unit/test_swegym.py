from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from siete_rl import swegym
from siete_rl.swegym import (
    SWEGymContractError,
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
