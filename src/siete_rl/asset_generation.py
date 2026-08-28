from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from siete_rl.swegym import (
    COMPARE_FIELDS,
    SWEGymContractError,
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
    row: Mapping[str, Any], assets_dir: Path, *, image_id: str
) -> list[Path]:
    task_id = str(row["instance_id"])
    _require_valid_task_id(task_id)
    image = image_tag_for(task_id)
    root = assets_dir / task_id
    root.mkdir(parents=True, exist_ok=True)

    texts = {
        "eval_script.sh": row["eval_script"],
        "eval_script.offline.sh": transform_eval_script_offline(row["eval_script"]),
        "gold.patch": row["patch"],
        "test.patch": row["test_patch"],
    }
    selected = {"instance_id": task_id, "eval_script": row["eval_script"], "image_name": image}
    for field in COMPARE_FIELDS:
        value = row[field]
        selected[field] = value.tolist() if hasattr(value, "tolist") else value
    texts["selected_instance.json"] = json.dumps(selected, ensure_ascii=False, indent=2)

    written: list[Path] = []
    for name, text in texts.items():
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        written.append(path)

    manifest = {
        "schema_version": "1",
        "task_id": task_id,
        "repo_name": row["repo"],
        "base_commit": row["base_commit"],
        "image_name": image,
        "expected_image_id": image_id,
    }
    manifest_path = root / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != manifest_text:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    written.append(manifest_path)
    return written


def image_tag_for(task_id: str) -> str:
    _require_valid_task_id(task_id)
    repository = task_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{repository}:latest"
