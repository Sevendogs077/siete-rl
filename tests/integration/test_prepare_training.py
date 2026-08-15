from collections import deque
import json
from pathlib import Path
from threading import Event, Lock
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from siete_rl.docker import CommandResult
from siete_rl.prepare import PreparedSource, build_training_table, prepare_training_assets


IMAGE_ID = "sha256:" + "1" * 64
IMAGE_INSPECT = json.dumps(
    [{"Id": IMAGE_ID, "Os": "linux", "Architecture": "amd64"}]
)


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
        self.pull_timeouts: list[int | None] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | None,
    ) -> CommandResult:
        del input_text
        command = list(argv)
        self.calls.append(command)
        if command[:2] == ["docker", "pull"]:
            self.pull_timeouts.append(timeout_sec)
        if command[-2:] == ["--format", "{{.Id}}"]:
            return docker_result(stdout=IMAGE_ID)
        response = self.responses.popleft()
        return CommandResult(
            argv=list(argv),
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            duration_sec=0.01,
            timed_out=response.timed_out,
        )


class ParallelPullDockerClient:
    def __init__(self) -> None:
        self._lock = Lock()
        self._overlap = Event()
        self._active_pulls = 0
        self.max_active_pulls = 0
        self._images: set[str] = set()

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: int,
    ) -> CommandResult:
        del input_text, timeout_sec
        command = list(argv)
        if command[:3] == ["docker", "image", "inspect"]:
            with self._lock:
                exists = command[3] in self._images
            if exists:
                return docker_result(
                    stdout=IMAGE_ID if "--format" in command else IMAGE_INSPECT
                )
            return docker_result(1)
        if command[:2] == ["docker", "pull"]:
            if command[2].startswith("docker.io/"):
                return docker_result(1)
            with self._lock:
                self._active_pulls += 1
                self.max_active_pulls = max(
                    self.max_active_pulls, self._active_pulls
                )
                if self._active_pulls >= 2:
                    self._overlap.set()
            self._overlap.wait(timeout=1)
            with self._lock:
                self._active_pulls -= 1
            return docker_result()
        if command[:2] == ["docker", "tag"]:
            with self._lock:
                self._images.add(command[3])
            return docker_result()
        if command[:2] == ["docker", "rmi"]:
            return docker_result()
        raise AssertionError(f"unexpected Docker command: {command}")


def docker_result(exit_code: int = 0, stdout: str = "") -> CommandResult:
    return CommandResult([], exit_code, stdout, "", 0.01)


def test_prepare_training_assets_exits_after_failed_mirror_pull(
    tmp_path: Path,
) -> None:
    sources = write_four_source_fixtures(tmp_path)
    table = build_training_table(sources, tmp_path / "train.parquet", seed=42)
    rows = pq.read_table(table).to_pylist()[:1]
    client = FakeDockerClient([docker_result(1) for _ in range(11)])

    with pytest.raises(RuntimeError, match="failed to pull training image"):
        prepare_training_assets(rows, tmp_path / "assets", client, pull_workers=1)

    pull_calls = [call for call in client.calls if call[:2] == ["docker", "pull"]]
    assert len(pull_calls) == 1
    assert pull_calls[0][2].startswith("dockerproxy.net/")


def test_prepare_training_assets_pulls_missing_images_concurrently(
    tmp_path: Path,
) -> None:
    sources = write_four_source_fixtures(tmp_path)
    table = build_training_table(sources, tmp_path / "train.parquet", seed=42)
    rows = pq.read_table(table).to_pylist()
    client = ParallelPullDockerClient()

    image_count, asset_count = prepare_training_assets(
        rows, tmp_path / "assets", client
    )

    assert (image_count, asset_count) == (4, 4)
    assert client.max_active_pulls >= 2
