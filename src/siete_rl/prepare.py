"""生成锁定来源的 Stage 1/2 训练课程。"""

from __future__ import annotations

import json
import random
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from siete_rl.asset_generation import generate_task_assets, image_tag_for
from siete_rl.docker import DockerClient, SubprocessDockerClient
from siete_rl.models import Environment


SKYRL_STAGE1 = {
    "modelscope_id": "NovaSky-AI/SkyRL-v0-80-data",
    "modelscope_revision": "65c39e9e7a4402058ba1ab6622cd77d4f741c207",
    "huggingface_id": "NovaSky-AI/SkyRL-v0-80-data",
    "huggingface_revision": "ee804395b10d041a5b8de1ed9fe1001557a14848",
    "filename": "train.parquet",
}
SKYRL_STAGE2 = {
    "modelscope_id": "NovaSky-AI/SkyRL-v0-220-data",
    "modelscope_revision": "245c20fd4ff99ddbb1ccbe0279fca86271094051",
    "huggingface_id": "NovaSky-AI/SkyRL-v0-220-data",
    "huggingface_revision": "a3eac67a20507675c2aec5e258d29b7605fbbdb0",
    "filename": "train.parquet",
}
SWEGYM_OFFICIAL = {
    "huggingface_id": "SWE-Gym/SWE-Gym",
    "huggingface_revision": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
    "filename": "data/train-00000-of-00001.parquet",
}
SWEGYM_SCRIPTS = {
    "huggingface_id": "SumanthRH/SWE-Gym",
    "huggingface_revision": "34bc75c1fc6be55d7667d075223b0a15dcaf9690",
    "filename": "data/train-00000-of-00001.parquet",
}

HF_MIRROR = "https://hf-mirror.com"


@dataclass(frozen=True)
class PreparedSource:
    path: Path
    platform: Literal["local", "modelscope", "huggingface"]
    dataset_id: str
    revision: str


def _install_parquet(downloaded: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(downloaded, temporary)
    pq.read_metadata(temporary)
    temporary.replace(target)
    return target


def _huggingface_source(name: str, definition: Mapping[str, str], root: Path) -> PreparedSource:
    import httpx
    from huggingface_hub import hf_hub_download

    dataset_id = definition["huggingface_id"]
    revision = definition["huggingface_revision"]
    target = root / name / revision / definition["filename"]
    if not target.is_file():
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=definition["filename"],
                )
            )
        except (httpx.HTTPError, RuntimeError):
            downloaded = Path(
                hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=definition["filename"],
                    endpoint=HF_MIRROR,
                )
            )
        _install_parquet(downloaded, target)
    else:
        pq.read_metadata(target)
    return PreparedSource(target, "huggingface", dataset_id, revision)


def _modelscope_source(name: str, definition: Mapping[str, str], root: Path) -> PreparedSource:
    dataset_id = definition["modelscope_id"]
    revision = definition["modelscope_revision"]
    target = root / name / revision / definition["filename"]
    if target.is_file():
        pq.read_metadata(target)
        return PreparedSource(target, "modelscope", dataset_id, revision)

    query = urllib.parse.urlencode(
        {"Revision": revision, "FilePath": definition["filename"]}
    )
    url = f"https://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo?{query}"
    temporary = target.with_suffix(target.suffix + ".download")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        normalized = body.lower()
        if exc.code != 404 and "not exist" not in normalized and "不存在" not in body:
            raise
        definition = dict(definition)
        return _huggingface_source(name, definition, root)
    try:
        _install_parquet(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return PreparedSource(target, "modelscope", dataset_id, revision)


def ensure_training_sources(project_root: Path) -> dict[str, PreparedSource]:
    import httpx
    from huggingface_hub import set_client_factory

    set_client_factory(lambda: httpx.Client(trust_env=False))
    root = project_root / "data" / "swegym" / "sources"
    return {
        "skyrl-stage1": _modelscope_source("skyrl-stage1", SKYRL_STAGE1, root),
        "skyrl-stage2": _modelscope_source("skyrl-stage2", SKYRL_STAGE2, root),
        "swegym-official": _huggingface_source(
            "swegym-official", SWEGYM_OFFICIAL, root
        ),
        "swegym-scripts": _huggingface_source("swegym-scripts", SWEGYM_SCRIPTS, root),
    }


def _skyrl_ids(source: PreparedSource) -> list[str]:
    rows = pq.read_table(source.path, columns=["instance"]).to_pylist()
    return [row["instance"]["instance_id"] for row in rows]


def _index_rows(source: PreparedSource) -> dict[str, dict[str, object]]:
    rows = pq.read_table(source.path).to_pylist()
    indexed = {row["instance_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{source.dataset_id} contains duplicate instance_id values")
    return indexed


def build_training_table(
    sources: Mapping[str, PreparedSource], output_path: Path, seed: int
) -> Path:
    stage1_ids = _skyrl_ids(sources["skyrl-stage1"])
    stage2_ids = _skyrl_ids(sources["skyrl-stage2"])
    if len(set(stage1_ids)) != len(stage1_ids):
        raise ValueError("Stage 1 contains duplicate instance IDs")
    if len(set(stage2_ids)) != len(stage2_ids):
        raise ValueError("Stage 2 contains duplicate instance IDs")
    overlap = set(stage1_ids) & set(stage2_ids)
    if overlap:
        raise ValueError(f"Stage 1 and Stage 2 overlap: {sorted(overlap)}")

    official = _index_rows(sources["swegym-official"])
    scripts = _index_rows(sources["swegym-scripts"])
    random.Random(seed).shuffle(stage1_ids)
    random.Random(seed).shuffle(stage2_ids)

    rows: list[dict[str, object]] = []
    for stage, task_ids in ((1, stage1_ids), (2, stage2_ids)):
        for stage_position, task_id in enumerate(task_ids):
            if task_id not in official or task_id not in scripts:
                raise ValueError(f"training task is missing a source row: {task_id}")
            eval_script = scripts[task_id].get("eval_script")
            if not isinstance(eval_script, str) or not eval_script.strip():
                raise ValueError(f"training task has an empty eval_script: {task_id}")
            row = dict(official[task_id])
            row["eval_script"] = eval_script
            row["stage"] = stage
            row["stage_position"] = stage_position
            rows.append(row)

    metadata = {
        name: {
            "platform": source.platform,
            "dataset_id": source.dataset_id,
            "revision": source.revision,
        }
        for name, source in sources.items()
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"siete_rl_sources": json.dumps(metadata, sort_keys=True).encode("utf-8")}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(output_path)
    return output_path


def prepare_training_assets(
    rows: list[dict[str, object]], assets_dir: Path, client: DockerClient
) -> tuple[int, int]:
    image_count = 0
    asset_count = 0
    for row in rows:
        task_id = str(row["instance_id"])
        image = image_tag_for(task_id)
        inspected = client.run(["docker", "image", "inspect", image], timeout_sec=30)
        if inspected.exit_code != 0 or inspected.timed_out:
            pulled = client.run(["docker", "pull", image], timeout_sec=3600)
            if pulled.exit_code != 0 or pulled.timed_out:
                mirror = f"dockerproxy.net/{image.removeprefix('docker.io/')}"
                mirror_pull = client.run(["docker", "pull", mirror], timeout_sec=3600)
                if mirror_pull.exit_code != 0 or mirror_pull.timed_out:
                    raise RuntimeError(f"failed to pull training image: {image}")
                tagged = client.run(["docker", "tag", mirror, image], timeout_sec=30)
                if tagged.exit_code != 0 or tagged.timed_out:
                    raise RuntimeError(f"failed to tag training image: {image}")
                removed = client.run(["docker", "rmi", mirror], timeout_sec=30)
                if removed.exit_code != 0 or removed.timed_out:
                    raise RuntimeError(f"failed to remove temporary mirror tag: {mirror}")

        environment = Environment(
            environment_id=f"prepare:{task_id}",
            task_id=task_id,
            image_name=image,
            workdir="/testbed",
            cpus=1,
            memory="1g",
            pids_limit=1,
            exec_timeout_sec=1,
            verifier_timeout_sec=1,
        )
        from siete_rl.docker import inspect_image

        inspect_image(client, environment)
        image_count += 1
        generate_task_assets(row, assets_dir)
        asset_count += 1
    return image_count, asset_count


def prepare_training(project_root: Path, seed: int) -> Path:
    sources = ensure_training_sources(project_root)
    output = build_training_table(
        sources, project_root / "data" / "swegym" / "train.parquet", seed
    )
    rows = pq.read_table(output, columns=["instance_id", "stage"]).to_pylist()
    stage1 = [row for row in rows if row["stage"] == 1]
    stage2 = [row for row in rows if row["stage"] == 2]
    if len(stage1) != 82 or len(stage2) != 220 or len({row["instance_id"] for row in rows}) != 302:
        raise ValueError("locked training course must contain 82 + 220 unique tasks")
    full_rows = pq.read_table(output).to_pylist()
    images, assets = prepare_training_assets(
        full_rows, project_root / "assets" / "swegym", SubprocessDockerClient()
    )
    print(f"prepared stage1={len(stage1)} stage2={len(stage2)} images={images} assets={assets}")
    return output
