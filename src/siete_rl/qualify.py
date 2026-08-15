"""唯一全套资格检查：config / dataset / assets / docker / tokenizer / GPU。

由 scripts/qualify.sh 单次运行；训练启动路径不重复这些检查。
设计原则：不变量在此验证，运行路径只做加载（见 siete_rl.swegym）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from siete_rl.config import ProjectConfig, load_config
from siete_rl.swegym import SWEGymContractError, TaskContext, load_task_context


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_docker(task_context: TaskContext) -> list[Check]:
    from siete_rl.docker import DockerRuntimeError, SubprocessDockerClient, inspect_image

    try:
        client = SubprocessDockerClient()
    except DockerRuntimeError as exc:
        return [Check("docker.daemon", False, str(exc))]
    checks = [Check("docker.daemon", True, client.docker_host)]
    for task_id, (sample, _) in task_context.items():
        try:
            inspect_image(client, sample.environment)
            checks.append(
                Check(
                    f"docker.{task_id}.image",
                    True,
                    f"{sample.environment.image_name} id 匹配",
                )
            )
        except Exception as exc:
            checks.append(Check(f"docker.{task_id}.image", False, str(exc)))
    return checks


def check_tokenizer(config: ProjectConfig) -> list[Check]:
    try:
        import inspect as _inspect

        from siete_rl.environment import SWEEnvironment
        from siete_rl.models import Task
        from siete_rl.prompts import build_prompt
        from siete_rl.train import build_processing_class

        tokenizer = build_processing_class(config)
        environment = SWEEnvironment(
            task_context={},
            sandbox_factory=lambda *args, **kwargs: None,
            verifier_factory=lambda *args, **kwargs: None,
            output_limit_chars=config.chat.max_observation_chars,
            max_timeout_sec=config.docker.exec_timeout_sec,
        )
        tools = [
            name
            for name, member in _inspect.getmembers(environment, predicate=_inspect.ismethod)
            if name != "reset" and not name.startswith("_")
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
        expected = ["execute_bash", "finish", "str_replace_editor"]
        ok = tools == expected and rendered.count("---- BEGIN FUNCTION") == 3 and "<tool_call>" not in rendered
        return [Check("tokenizer.tool_render", ok, "OpenHands 三工具 scaffold" if ok else "渲染异常")]
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
    parser = argparse.ArgumentParser(prog="siete_rl.qualify", description="单次全套资格检查")
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
    task_context: TaskContext | None = None
    try:
        task_context = load_task_context(config, project_root)
        checks.append(
            Check(
                "dataset.training_context",
                True,
                f"已加载 {len(task_context)} 个训练任务及运行资产",
            )
        )
    except (SWEGymContractError, KeyError, ValueError) as exc:
        checks.append(Check("dataset.training_context", False, str(exc)))
    if not args.no_docker and task_context is not None:
        checks.extend(check_docker(task_context))
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
