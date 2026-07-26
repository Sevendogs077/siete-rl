"""唯一全套资格检查：config / dataset / assets / docker / tokenizer / GPU。

由 scripts/qualify.sh 单次运行；训练启动路径不重复这些检查。
设计原则：不变量在此验证，运行路径只做加载（见 swe_agent.swegym）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from swe_agent.config import ProjectConfig, load_config
from swe_agent.swegym import (
    COMPARE_FIELDS,
    OFFICIAL_REVISION,
    SUBSET_REVISION,
    SWEGymContractError,
    _normalize,
    _read_exact_row,
    _read_object,
    _require_sha256,
    _resolve,
    select_task_ids,
    transform_eval_script_offline,
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_dataset(config: ProjectConfig, project_root: Path) -> list[Check]:
    checks: list[Check] = []
    for task_id in select_task_ids(config, project_root):
        checks.extend(_check_dataset_one(config, project_root, task_id))
    return checks


def _check_dataset_one(
    config: ProjectConfig, project_root: Path, task_id: str
) -> list[Check]:
    checks: list[Check] = []
    official_path = _resolve(project_root, config.dataset.official_path)
    subset_path = _resolve(project_root, config.dataset.subset_path)
    official = subset = None
    try:
        official = _read_exact_row(official_path, task_id)
        checks.append(Check(f"dataset.{task_id}.official_row", True, task_id))
    except SWEGymContractError as exc:
        checks.append(Check(f"dataset.{task_id}.official_row", False, str(exc)))
    try:
        subset = _read_exact_row(subset_path, task_id)
        checks.append(Check(f"dataset.{task_id}.subset_row", True, task_id))
    except SWEGymContractError as exc:
        checks.append(Check(f"dataset.{task_id}.subset_row", False, str(exc)))
    if official is not None and subset is not None:
        mismatched = [
            field
            for field in COMPARE_FIELDS
            if _normalize(official.get(field)) != _normalize(subset.get(field))
        ]
        checks.append(
            Check(
                f"dataset.{task_id}.cross_table_fields",
                not mismatched,
                "一致" if not mismatched else f"字段不一致: {', '.join(mismatched)}",
            )
        )
        eval_script = subset.get("eval_script")
        ok = isinstance(eval_script, str) and bool(eval_script.strip())
        checks.append(Check(f"dataset.{task_id}.eval_script", ok, "存在" if ok else "缺失"))
    revision_ok = (
        config.dataset.official_revision == OFFICIAL_REVISION
        and config.dataset.subset_revision == SUBSET_REVISION
    )
    checks.append(
        Check(
            f"dataset.{task_id}.revisions",
            revision_ok,
            "与锁定版本一致" if revision_ok else "与锁定版本不一致",
        )
    )
    return checks


def check_assets(config: ProjectConfig, project_root: Path) -> list[Check]:
    checks: list[Check] = []
    for task_id in select_task_ids(config, project_root):
        checks.extend(_check_assets_one(config, project_root, task_id))
    return checks


def _check_assets_one(
    config: ProjectConfig, project_root: Path, task_id: str
) -> list[Check]:
    checks: list[Check] = []
    assets_dir = _resolve(project_root, config.dataset.tasks_dir) / task_id
    official = subset = None
    try:
        official = _read_exact_row(_resolve(project_root, config.dataset.official_path), task_id)
        subset = _read_exact_row(_resolve(project_root, config.dataset.subset_path), task_id)
    except SWEGymContractError:
        pass

    def _text(name: str) -> str | None:
        path = assets_dir / name
        return path.read_text(encoding="utf-8") if path.is_file() else None

    for name in (
        "selected_instance.json",
        "eval_script.sh",
        "eval_script.offline.sh",
        "gold.patch",
        "test.patch",
        "manifest.json",
    ):
        checks.append(
            Check(
                f"assets.{task_id}.exists.{name}",
                _text(name) is not None,
                str(assets_dir / name),
            )
        )
    original, offline = _text("eval_script.sh"), _text("eval_script.offline.sh")
    if subset is not None and original is not None:
        checks.append(
            Check(
                f"assets.{task_id}.eval_script_matches_dataset",
                original == subset.get("eval_script"),
                "eval_script.sh 与数据集行一致" if original == subset.get("eval_script") else "不一致",
            )
        )
    if original is not None and offline is not None:
        try:
            ok = transform_eval_script_offline(original) == offline
            detail = "离线化正确" if ok else "离线脚本不等于受控替换"
        except SWEGymContractError as exc:
            ok, detail = False, str(exc)
        checks.append(Check(f"assets.{task_id}.offline_transform", ok, detail))
    if official is not None:
        gold, test_patch = _text("gold.patch"), _text("test.patch")
        if gold is not None:
            checks.append(
                Check(
                    f"assets.{task_id}.gold_matches_patch",
                    gold == official.get("patch"),
                    "gold.patch",
                )
            )
        if test_patch is not None:
            checks.append(
                Check(
                    f"assets.{task_id}.test_matches_test_patch",
                    test_patch == official.get("test_patch"),
                    "test.patch",
                )
            )
    manifest_path = assets_dir / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = _read_object(manifest_path)
            for name, expected_hash in (manifest.get("files") or {}).items():
                _require_sha256(assets_dir / name, expected_hash)
            checks.append(
                Check(
                    f"assets.{task_id}.manifest_hashes",
                    True,
                    "manifest 文件哈希全部一致",
                )
            )
        except SWEGymContractError as exc:
            checks.append(Check(f"assets.{task_id}.manifest_hashes", False, str(exc)))
    selected_path = assets_dir / "selected_instance.json"
    if selected_path.is_file() and official is not None:
        try:
            selected = _read_object(selected_path)
            mismatched = [
                field
                for field in ("instance_id", *COMPARE_FIELDS, "eval_script")
                if _normalize(selected.get(field))
                != _normalize(task_id if field == "instance_id" else
                              official.get(field) if field != "eval_script" else subset.get(field))
            ]
            if manifest is None or _normalize(selected.get("image_name")) != _normalize(
                manifest.get("image_name")
            ):
                mismatched.append("image_name")
            checks.append(
                Check(
                    f"assets.{task_id}.selected_instance",
                    not mismatched,
                    "一致" if not mismatched else f"字段不一致: {', '.join(mismatched)}",
                )
            )
        except SWEGymContractError as exc:
            checks.append(Check(f"assets.{task_id}.selected_instance", False, str(exc)))
    return checks


def check_docker(config: ProjectConfig, project_root: Path) -> list[Check]:
    from swe_agent.docker import DockerRuntimeError, SubprocessDockerClient, inspect_image
    from swe_agent.models import Environment

    try:
        client = SubprocessDockerClient()
    except DockerRuntimeError as exc:
        return [Check("docker.daemon", False, str(exc))]
    checks = [Check("docker.daemon", True, client.docker_host)]
    tasks_dir = _resolve(project_root, config.dataset.tasks_dir)
    for task_id in select_task_ids(config, project_root):
        try:
            manifest = _read_object(tasks_dir / task_id / "manifest.json")
            probe = Environment(
                environment_id=f"qualify:{task_id}",
                task_id=task_id,
                image_name=manifest["image_name"],
                expected_image_id=manifest["expected_image_id"],
                expected_registry_digest=manifest["expected_registry_digest"],
                workdir="/testbed",
                cpus=config.docker.cpus,
                memory=config.docker.memory,
                pids_limit=config.docker.pids_limit,
                exec_timeout_sec=config.docker.exec_timeout_sec,
                verifier_timeout_sec=config.docker.verifier_timeout_sec,
            )
            inspect_image(client, probe)
            checks.append(
                Check(
                    f"docker.{task_id}.image",
                    True,
                    f"{probe.image_name} id 匹配",
                )
            )
        except Exception as exc:
            checks.append(Check(f"docker.{task_id}.image", False, str(exc)))
    return checks


def check_tokenizer(config: ProjectConfig) -> list[Check]:
    try:
        import inspect as _inspect

        from swe_agent.environment import SWEEnvironment
        from swe_agent.models import Task
        from swe_agent.prompts import build_prompt
        from swe_agent.train import build_processing_class

        tokenizer = build_processing_class(config)
        environment = SWEEnvironment(
            task_context={},
            sandbox_factory=lambda *args, **kwargs: None,
            verifier_factory=lambda *args, **kwargs: None,
            output_limit_chars=config.chat.max_observation_chars,
            max_timeout_sec=config.docker.exec_timeout_sec,
        )
        tools = [
            member
            for name, member in _inspect.getmembers(environment, predicate=_inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        ]
        rendered = tokenizer.apply_chat_template(
            build_prompt(
                Task(
                    task_id="qualify",
                    repo_name="owner/repo",
                    base_commit="0" * 40,
                    problem_statement="probe",
                )
            ),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        ok = rendered.count('"type": "function"') == 6 and "<tool_call>" in rendered
        return [Check("tokenizer.tool_render", ok, "六工具 + <tool_call> 指令" if ok else "渲染异常")]
    except Exception as exc:
        return [Check("tokenizer.tool_render", False, str(exc))]


def check_gpu() -> list[Check]:
    try:
        import torch

        if not torch.cuda.is_available():
            return [Check("gpu.available", False, "无可用 CUDA 设备", required=False)]
        lines = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            lines.append(f"GPU{index} {torch.cuda.get_device_name(index)} 空闲 {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB")
        return [Check("gpu.available", True, "；".join(lines), required=False)]
    except Exception as exc:
        return [Check("gpu.available", False, str(exc), required=False)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swe_agent.qualify", description="单次全套资格检查")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-docker", action="store_true", help="跳过 docker daemon/镜像检查")
    args = parser.parse_args(argv)

    checks: list[Check] = []
    try:
        config, project_root, resolved = load_config(args.config)
        checks.append(Check("config.load", True, resolved.as_posix()))
    except Exception as exc:
        print(f"FAIL config.load: {exc}")
        return 1
    checks.extend(check_dataset(config, project_root))
    checks.extend(check_assets(config, project_root))
    if not args.no_docker:
        checks.extend(check_docker(config, project_root))
    checks.extend(check_tokenizer(config))
    checks.extend(check_gpu())

    failed = 0
    for check in checks:
        marker = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
        if check.required and not check.ok:
            failed += 1
        print(f"{marker} {check.name}: {check.detail}")
    print(f"== qualify {'通过' if failed == 0 else f'失败（{failed} 项）'} ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
