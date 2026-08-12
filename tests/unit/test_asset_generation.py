from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from siete_rl.asset_generation import fetch_registry_digest, generate_task_assets
from siete_rl.swegym import SWEGymContractError


def _rows(instance_id: str) -> list[dict]:
    return [
        {
            "instance_id": instance_id,
            "repo": "getmoto/moto",
            "base_commit": "a" * 40,
            "version": "4.2",
            "problem_statement": "problem",
            "patch": "gold-diff",
            "test_patch": "test-diff",
            "FAIL_TO_PASS": ["t1"],
            "PASS_TO_PASS": ["t2"],
            "eval_script": "make init\npytest -q\n",
        }
    ]


def _kwargs(official: Path, subset: Path, assets_dir: Path) -> dict:
    return dict(
        task_id="owner__repo-1",
        official_path=official,
        subset_path=subset,
        assets_dir=assets_dir,
        image="docker.io/x/owner_s_repo-1:latest",
        image_id="sha256:" + "1" * 64,
        registry_digest="sha256:" + "2" * 64,
        official_revision="rev-official",
        subset_revision="rev-subset",
    )


def test_generate_task_assets_is_idempotent(tmp_path: Path) -> None:
    official = tmp_path / "official.parquet"
    subset = tmp_path / "subset.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), official)
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), subset)
    kwargs = _kwargs(official, subset, tmp_path / "assets")
    first = generate_task_assets(**kwargs)
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}
    second = generate_task_assets(**kwargs)
    assert second == first
    assert {p.name: p.stat().st_mtime_ns for p in second} == mtimes


def test_generate_task_assets_rewrites_on_upstream_change(tmp_path: Path) -> None:
    official = tmp_path / "official.parquet"
    subset = tmp_path / "subset.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), official)
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), subset)
    kwargs = _kwargs(official, subset, tmp_path / "assets")
    generate_task_assets(**kwargs)
    root = tmp_path / "assets" / "owner__repo-1"
    assert (root / "gold.patch").read_text() == "gold-diff"

    changed = _rows("owner__repo-1")
    changed[0]["patch"] = "gold-diff-v2"
    pq.write_table(pa.Table.from_pylist(changed), official)
    generate_task_assets(**kwargs)
    assert (root / "gold.patch").read_text() == "gold-diff-v2"


def test_generate_task_assets_rejects_invalid_task_id(tmp_path: Path) -> None:
    official = tmp_path / "official.parquet"
    subset = tmp_path / "subset.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), official)
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), subset)
    kwargs = _kwargs(official, subset, tmp_path / "assets")
    kwargs["task_id"] = "../escape"
    with pytest.raises(SWEGymContractError, match="task_id"):
        generate_task_assets(**kwargs)
    assert not (tmp_path / "escape").exists()


def test_fetch_registry_digest_rejects_invalid_task_id() -> None:
    with pytest.raises(SWEGymContractError, match="task_id"):
        fetch_registry_digest("docker.1panel.live", "../escape")
