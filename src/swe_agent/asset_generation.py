"""单任务资产生成：从锁定 parquet 与 daemon 事实产出六个文件；批量生成由 prepare.sh 驱动，深度校验由 qualify 负责。"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from swe_agent.swegym import (
    COMPARE_FIELDS,
    SWEGymContractError,
    _read_exact_row,
    transform_eval_script_offline,
)


ASSET_FILES = (
    "selected_instance.json",
    "eval_script.sh",
    "eval_script.offline.sh",
    "gold.patch",
    "test.patch",
)

TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _require_valid_task_id(task_id: str) -> None:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise SWEGymContractError(f"task_id 含非法字符（仅允许 [A-Za-z0-9_.-]）: {task_id!r}")


def generate_task_assets(
    *,
    task_id: str,
    official_path: Path,
    subset_path: Path,
    assets_dir: Path,
    image: str,
    image_id: str,
    registry_digest: str,
    official_revision: str,
    subset_revision: str,
) -> list[Path]:
    _require_valid_task_id(task_id)
    official = _read_exact_row(official_path, task_id)
    subset = _read_exact_row(subset_path, task_id)
    root = assets_dir / task_id
    root.mkdir(parents=True, exist_ok=True)

    texts = {
        "eval_script.sh": subset["eval_script"],
        "eval_script.offline.sh": transform_eval_script_offline(subset["eval_script"]),
        "gold.patch": official["patch"],
        "test.patch": official["test_patch"],
    }
    selected = {"instance_id": task_id, "eval_script": subset["eval_script"], "image_name": image}
    for field in COMPARE_FIELDS:
        value = official[field]
        selected[field] = value.tolist() if hasattr(value, "tolist") else value
    texts["selected_instance.json"] = json.dumps(selected, ensure_ascii=False, indent=2)

    written: list[Path] = []
    for name, text in texts.items():
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        written.append(path)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "1",
        "task_id": task_id,
        "repo_name": official["repo"],
        "base_commit": official["base_commit"],
        "image_name": image,
        "expected_image_id": image_id,
        "expected_registry_digest": registry_digest,
        "files": {name: sha(root / name) for name in ASSET_FILES},
        "datasets": {
            "official": {"revision": official_revision, "sha256": sha(official_path)},
            "subset": {"revision": subset_revision, "sha256": sha(subset_path)},
        },
    }
    manifest_path = root / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != manifest_text:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    written.append(manifest_path)
    return written


def image_tag_for(task_id: str) -> str:
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{task_id.replace('__', '_s_')}:latest"


def fetch_registry_digest(mirror: str, task_id: str) -> str:
    _require_valid_task_id(task_id)
    path = f"xingyaoww/sweb.eval.x86_64.{task_id.replace('__', '_s_')}"
    request = urllib.request.Request(
        f"https://{mirror}/v2/{path}/manifests/latest",
        headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json,"
                      "application/vnd.docker.distribution.manifest.list.v2+json",
            # 镜像站按 UA 反爬：Python-urllib 默认 UA 会被 403，伪装成 docker 客户端。
            "User-Agent": "docker/24.0.7 go/go1.21 kernel/6.8 os/linux arch/amd64",
        },
        method="HEAD",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                digest = response.headers["Docker-Content-Digest"]
            if not digest.startswith("sha256:"):
                raise SWEGymContractError(f"镜像站返回非法 digest（{task_id}）: {digest}")
            return digest
        except Exception as exc:  # 网络错误、响应头缺失、非法 digest 一律退避重试
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    raise SWEGymContractError(
        f"获取镜像 digest 失败（{task_id}），3 次尝试均失败: {last_error}"
    ) from last_error
