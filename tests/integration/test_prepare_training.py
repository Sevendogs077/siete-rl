from pathlib import Path
from collections import deque
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from siete_rl.docker import CommandResult
from siete_rl.prepare import PreparedSource, build_training_table, prepare_training_assets


def write_four_source_fixtures(tmp_path: Path) -> dict[str, PreparedSource]:
    stage1_ids = ["Project-MONAI__MONAI-1", "getmoto__moto-2"]
    stage2_ids = ["pandas-dev__pandas-3", "pytest-dev__pytest-4"]
    all_ids = stage1_ids + stage2_ids

    rows_by_source = {
        "skyrl-stage1": [{"instance": {"instance_id": task_id}} for task_id in stage1_ids],
        "skyrl-stage2": [{"instance": {"instance_id": task_id}} for task_id in stage2_ids],
        "swegym-official": [
            {
                "instance_id": task_id,
                "repo": task_id.split("-")[0],
                "base_commit": f"commit-{index}",
                "version": "fixture",
                "problem_statement": f"problem {index}",
                "patch": f"patch {index}",
                "test_patch": f"test patch {index}",
                "FAIL_TO_PASS": [f"test-{index}"],
                "PASS_TO_PASS": [],
            }
            for index, task_id in enumerate(all_ids)
        ],
        "swegym-scripts": [
            {
                "instance_id": task_id,
                "eval_script": f"make init\necho verify-{index}\n",
            }
            for index, task_id in enumerate(all_ids)
        ],
    }

    sources: dict[str, PreparedSource] = {}
    for source_name, rows in rows_by_source.items():
        path = tmp_path / f"{source_name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        sources[source_name] = PreparedSource(
            path=path,
            platform="local",
            dataset_id=source_name,
            revision="fixture",
        )
    return sources


def test_build_training_table_preserves_stage_order_and_original_ids(
    tmp_path: Path,
) -> None:
    sources = write_four_source_fixtures(tmp_path)
    first = build_training_table(sources, tmp_path / "train.parquet", seed=42)
    second = build_training_table(sources, tmp_path / "train-again.parquet", seed=42)

    first_rows = pq.read_table(first).to_pylist()
    second_rows = pq.read_table(second).to_pylist()

    assert [row["instance_id"] for row in first_rows] == [
        row["instance_id"] for row in second_rows
    ]
    assert [row["stage"] for row in first_rows] == [1, 1, 2, 2]
    assert {row["instance_id"] for row in first_rows} == {
        "Project-MONAI__MONAI-1",
        "getmoto__moto-2",
        "pandas-dev__pandas-3",
        "pytest-dev__pytest-4",
    }
    assert all(row["eval_script"].strip() for row in first_rows)


class FakeDockerClient:
    def __init__(self, responses: list[CommandResult]) -> None:
        self.responses = deque(responses)
        self.calls: list[list[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int,
    ) -> CommandResult:
        del input_text, timeout_sec
        self.calls.append(list(argv))
        response = self.responses.popleft()
        return CommandResult(
            argv=list(argv),
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            duration_sec=0.01,
            timed_out=response.timed_out,
        )


def docker_result(exit_code: int = 0, stdout: str = "") -> CommandResult:
    return CommandResult([], exit_code, stdout, "", 0.01)


def test_prepare_training_assets_uses_canonical_and_mirror_once(tmp_path: Path) -> None:
    sources = write_four_source_fixtures(tmp_path)
    table = build_training_table(sources, tmp_path / "train.parquet", seed=42)
    rows = pq.read_table(table).to_pylist()
    client = FakeDockerClient(
        [
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(1),
            docker_result(1),
            docker_result(),
            docker_result(),
            docker_result(),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
            docker_result(stdout='[{"Os":"linux","Architecture":"amd64"}]'),
        ]
    )

    image_count, asset_count = prepare_training_assets(
        rows, tmp_path / "assets", client
    )

    assert (image_count, asset_count) == (4, 4)
    assert (tmp_path / "assets/Project-MONAI__MONAI-1/manifest.json").is_file()
    pull_calls = [call for call in client.calls if call[:2] == ["docker", "pull"]]
    assert len(pull_calls) == 2
    assert pull_calls[1][2].startswith("dockerproxy.net/")
