from __future__ import annotations

import atexit
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer
from trl.chat_template_utils import parse_response

from siete_rl.docker import (
    ContainerCreateError,
    DockerSandbox,
    SubprocessDockerClient,
    sweep_run_containers,
)
from siete_rl.environment import SWEEnvironment
from siete_rl.launcher import VLLMServer
from siete_rl.models import Environment, Evaluation, Sample, Task
from siete_rl.prompts import build_prompt
from siete_rl.tool_protocol import install_openhands_tool_protocol
from siete_rl.trainer import SWEGRPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_REVISION = "91aa3ed51b709be6457e12d00300a6a596d4c6a3"
HARNESS_REVISION = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
PYTORCH_SAMPLER_REASON = (
    "FlashInfer 0.6.11.post2 sampling JIT fails against the installed CUDA/CUB "
    "with BlockAdjacentDifference::FlagHeads compile errors"
)
VERIFIED_PARQUET = (
    PROJECT_ROOT
    / "data/swegym/SWE-Bench__SWE-bench_Verified"
    / VERIFIED_REVISION
    / "data/test-00000-of-00001.parquet"
)
DEFAULT_HARNESS_ROOT = PROJECT_ROOT / ".external" / "swe-bench"
EXTERNAL_REFERENCES = (
    {
        "source": "SWE-Gym paper",
        "model": "Qwen2.5-Coder-7B-Instruct zero-shot",
        "verified_resolve_rate": 0.018,
        "comparison": "external_system_reference",
    },
    {
        "source": "SWE-Gym paper",
        "model": "SWE-Gym OpenHands-7B-Agent SFT",
        "verified_resolve_rate": 0.106,
        "comparison": "external_system_reference",
    },
    {
        "source": "SkyRL README",
        "model": "OpenHands-7B-Agent base",
        "verified_resolve_rate": 0.110,
        "comparison": "published_release_reference",
    },
    {
        "source": "SkyRL README",
        "model": "SkyRL-Agent-7B-v0",
        "verified_resolve_rate": 0.146,
        "comparison": "published_release_reference",
    },
)


class EvalError(RuntimeError):
    """评测输入、资源或运行结果不满足固定协议。"""


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    max_prompt_length: int
    max_completion_length: int
    max_model_length: int
    max_tool_calling_iterations: int
    max_consecutive_protocol_errors: int
    max_observation_chars: int
    exec_timeout_sec: int
    grader_timeout_sec: int
    cpus: float
    memory: str
    pids_limit: int
    seed: int
    repetition_penalty: float
    gpu_memory_utilization: float


@dataclass(frozen=True, slots=True)
class EvalRun:
    root: Path
    adapter_path: Path
    base_model_path: Path
    rank: int
    protocol: EvalProtocol
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    task_id: str
    patch: str
    termination: str
    infrastructure_error: str | None
    messages: list[dict[str, Any]]
    duration_sec: float


def parse_strict_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if value == "True":
        return True
    if value == "False":
        return False
    raise EvalError("EVAL_BASE must be exactly 'True' or 'False'")


def _positive_env_int(name: str, *, default: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EvalError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise EvalError(f"{name} must be a positive integer")
    return parsed


def load_eval_run(path: str | Path) -> EvalRun:
    root = Path(path).expanduser().resolve()
    allowed_roots = ((PROJECT_ROOT / "outputs").resolve(), (PROJECT_ROOT / "_archive").resolve())
    if not any(root.is_relative_to(parent) for parent in allowed_roots):
        raise EvalError("run root must be under this project's outputs/ or _archive/")
    if not root.is_dir():
        raise EvalError(f"run root does not exist: {root}")
    adapter_config_path = root / "adapter_config.json"
    adapter_model_path = root / "adapter_model.safetensors"
    config_path = root / "config.yaml"
    for required in (adapter_config_path, adapter_model_path, config_path):
        if not required.is_file():
            raise EvalError(f"required final-run artifact is missing: {required}")
    adapter = _read_object(adapter_config_path)
    config = _read_yaml_object(config_path)
    base_value = adapter.get("base_model_name_or_path")
    rank = adapter.get("r")
    if not isinstance(base_value, str) or not base_value:
        raise EvalError("adapter_config.json has no base_model_name_or_path")
    if not isinstance(rank, int) or rank < 1:
        raise EvalError("adapter_config.json has no positive LoRA rank")
    base_model_path = Path(base_value).expanduser().resolve()
    if not base_model_path.is_dir():
        raise EvalError(f"adapter base model does not exist locally: {base_model_path}")
    model_path = _field(config, "model", "model_path", expected=str)
    if Path(model_path).expanduser().resolve() != base_model_path:
        raise EvalError("run config model_path disagrees with adapter base_model_name_or_path")
    protocol = EvalProtocol(
        max_prompt_length=_positive_int(config, "chat", "max_prompt_length"),
        max_completion_length=_positive_int(config, "generation", "max_completion_length"),
        max_model_length=_positive_int(config, "vllm", "max_model_length"),
        max_tool_calling_iterations=_positive_int(
            config, "generation", "max_tool_calling_iterations"
        ),
        max_consecutive_protocol_errors=_positive_int(
            config, "generation", "max_consecutive_protocol_errors"
        ),
        max_observation_chars=_positive_int(config, "chat", "max_observation_chars"),
        exec_timeout_sec=_positive_int(config, "docker", "exec_timeout_sec"),
        grader_timeout_sec=_positive_int(config, "docker", "verifier_timeout_sec"),
        cpus=_positive_number(config, "docker", "cpus"),
        memory=_field(config, "docker", "memory", expected=str),
        pids_limit=_positive_int(config, "docker", "pids_limit"),
        seed=_nonnegative_int(config, "runtime", "base_seed"),
        repetition_penalty=_positive_number(config, "generation", "repetition_penalty"),
        gpu_memory_utilization=_positive_number(config, "vllm", "gpu_memory_utilization"),
    )
    if protocol.max_prompt_length + protocol.max_completion_length > protocol.max_model_length:
        raise EvalError("run config prompt/completion budget exceeds vLLM context")
    return EvalRun(
        root=root,
        adapter_path=root,
        base_model_path=base_model_path,
        rank=rank,
        protocol=protocol,
        config=MappingProxyType(config),
    )


def load_verified_rows(task_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(VERIFIED_PARQUET).to_pylist()
    except Exception as exc:
        raise EvalError(f"failed to read pinned Verified parquet: {exc}") from exc
    if len(rows) != 500:
        raise EvalError(f"pinned Verified split must contain 500 rows; found {len(rows)}")
    by_id = {str(row.get("instance_id")): row for row in rows}
    if len(by_id) != 500:
        raise EvalError("pinned Verified split contains duplicate instance IDs")
    if task_ids is None:
        return rows
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise EvalError("unknown EVAL_TASK_IDS: " + ", ".join(missing))
    if len(set(task_ids)) != len(task_ids):
        raise EvalError("EVAL_TASK_IDS must not contain duplicates")
    return [by_id[task_id] for task_id in task_ids]


def public_sample_from_row(
    row: Mapping[str, Any], protocol: EvalProtocol, client: SubprocessDockerClient
) -> Sample:
    """只把公开问题字段送入 Agent；grader 字段不会进入 Task。"""

    task_id = _required_row_string(row, "instance_id")
    image_name = official_image_name(task_id)
    inspected = _inspect_official_image(client, image_name)
    image_id = inspected.get("Id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise EvalError(f"{task_id}: official image has invalid local ID")
    task = Task(
        task_id=task_id,
        repo_name=_required_row_string(row, "repo"),
        base_commit=_required_row_string(row, "base_commit"),
        problem_statement=_required_row_string(row, "problem_statement"),
    )
    environment = Environment(
        environment_id=f"swebench-verified:{task_id}",
        task_id=task_id,
        image_name=image_name,
        expected_image_id=image_id,
        workdir="/testbed",
        cpus=protocol.cpus,
        memory=protocol.memory,
        pids_limit=protocol.pids_limit,
        exec_timeout_sec=protocol.exec_timeout_sec,
        verifier_timeout_sec=protocol.grader_timeout_sec,
    )
    return Sample(task=task, environment=environment)


def official_image_name(task_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{task_id.lower()}:latest".replace("__", "_1776_")


class EvalDockerSandbox(DockerSandbox):
    """只为 official prep commit 收紧适配；训练 Docker 合同保持原样。"""

    def _verify_base_contract(self) -> None:
        head = self.exec(["git", "-C", self.environment.workdir, "rev-parse", "HEAD"])
        if head.exit_code != 0 or head.timed_out:
            raise ContainerCreateError("failed to inspect official image HEAD")
        actual = head.stdout.strip()
        if actual != self.task.base_commit:
            parent = self.exec(
                ["git", "-C", self.environment.workdir, "rev-parse", "HEAD^"]
            )
            if (
                parent.exit_code != 0
                or parent.timed_out
                or parent.stdout.strip() != self.task.base_commit
            ):
                raise ContainerCreateError(
                    "official image base commit mismatch: "
                    f"expected HEAD or HEAD^={self.task.base_commit}, actual HEAD={actual}"
                )
            checkout = self.exec(
                ["git", "-C", self.environment.workdir, "checkout", "--detach", self.task.base_commit]
            )
            if checkout.exit_code != 0 or checkout.timed_out:
                raise ContainerCreateError("failed to checkout official image base commit")
        clean = self.exec(["git", "-C", self.environment.workdir, "clean", "-fd"])
        if clean.exit_code != 0 or clean.timed_out:
            raise ContainerCreateError("failed to clean official image worktree")
        verified = self.exec(["git", "-C", self.environment.workdir, "rev-parse", "HEAD"])
        status = self.exec(["git", "-C", self.environment.workdir, "status", "--porcelain"])
        if (
            verified.exit_code != 0
            or verified.timed_out
            or verified.stdout.strip() != self.task.base_commit
        ):
            raise ContainerCreateError("official image did not settle on the exact base commit")
        if status.exit_code != 0 or status.timed_out:
            raise ContainerCreateError("failed to inspect official image worktree")
        if status.stdout.strip():
            raise ContainerCreateError("official image worktree is not clean after base checkout")


class EvalEnvironment(SWEEnvironment):
    """不调用训练 verifier 的 eval-only environment。"""

    def finalize_eval(self, messages: list[dict[str, Any]], started: float) -> EvalOutcome:
        infrastructure_error = self._infrastructure_error
        patch = self._frozen_patch
        try:
            if patch is None and self._sandbox is not None and infrastructure_error is None:
                patch = self._sandbox.get_diff()
        except BaseException as exc:
            infrastructure_error = exc
            patch = ""
        if self._terminal_event is not None:
            termination = self._terminal_event.kind
        elif self._loop_exit is not None:
            termination = self._loop_exit
        elif infrastructure_error is not None:
            termination = "infra_error"
        else:
            termination = "iteration_cap"
        task_id = self._sample.task.task_id if self._sample is not None else "unknown"
        try:
            self._close()
        except BaseException as exc:
            infrastructure_error = infrastructure_error or exc
            termination = "infra_error"
        return EvalOutcome(
            task_id=task_id,
            patch=patch or "",
            termination=termination,
            infrastructure_error=(
                None
                if infrastructure_error is None
                else f"{type(infrastructure_error).__name__}: {infrastructure_error}"
            ),
            messages=messages,
            duration_sec=time.monotonic() - started,
        )


class HTTPTokenGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        max_completion_length: int,
        max_model_length: int,
        seed: int,
        repetition_penalty: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_completion_length = max_completion_length
        self.max_model_length = max_model_length
        self.seed = seed
        self.repetition_penalty = repetition_penalty

    def generate(self, prompt_ids: list[list[int]], *_: Any) -> tuple[list[list[int]], None]:
        outputs: list[list[int]] = []
        for tokens in prompt_ids:
            remaining = min(
                self.max_completion_length,
                self.max_model_length - len(tokens) - 1,
            )
            if remaining < 1:
                outputs.append([])
                continue
            response = _post_json(
                f"{self.base_url}/v1/completions",
                {
                    "model": self.model_name,
                    "prompt": tokens,
                    "add_special_tokens": False,
                    "temperature": 0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": self.repetition_penalty,
                    "seed": self.seed,
                    "max_tokens": remaining,
                    "return_token_ids": True,
                },
                timeout_sec=1800,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise EvalError("vLLM completion response must contain exactly one choice")
            token_ids = choices[0].get("token_ids")
            if not isinstance(token_ids, list) or not all(isinstance(value, int) for value in token_ids):
                raise EvalError("vLLM completion response is missing token_ids")
            outputs.append(token_ids)
        return outputs, None


def run_agent_loop(
    *,
    sample: Sample,
    tokenizer: Any,
    generator: HTTPTokenGenerator,
    protocol: EvalProtocol,
    docker_client: SubprocessDockerClient,
    eval_run_id: str,
) -> EvalOutcome:
    evaluation = Evaluation(offline_eval_script="official harness only")
    context = MappingProxyType({sample.task.task_id: (sample, evaluation)})

    def sandbox_factory(current: Sample, episode_id: str, scope: str) -> EvalDockerSandbox:
        if scope != "rollout":
            raise EvalError("eval environment must never create a verifier sandbox")
        return EvalDockerSandbox(
            client=docker_client,
            task=current.task,
            environment=current.environment,
            run_id=eval_run_id,
            episode_id=episode_id,
            scope="rollout",
        )

    def forbidden_verifier(*_: Any) -> Any:
        raise EvalError("eval environment must never instantiate the training verifier")

    environment = EvalEnvironment(
        task_context=context,
        sandbox_factory=sandbox_factory,
        verifier_factory=forbidden_verifier,
        output_limit_chars=protocol.max_observation_chars,
        max_timeout_sec=protocol.exec_timeout_sec,
    )
    started = time.monotonic()
    try:
        environment.reset(sample.task.task_id)
        prompt = build_prompt(sample.task)
        prompt_ids = list(
            tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        )
        if len(prompt_ids) + 1 >= protocol.max_model_length:
            raise EvalError(
                f"{sample.task.task_id}: public prompt has {len(prompt_ids)} tokens, "
                f"leaving no completion capacity in {protocol.max_model_length} tokens"
            )
        first_ids, _ = generator.generate([prompt_ids])
        first = parse_response(tokenizer, first_ids[0], prefix=prompt_ids) if first_ids[0] else {}
        if not first:
            environment._record_loop_exit("context_overlong")
            return environment.finalize_eval([], started)
        trainer = object.__new__(SWEGRPOTrainer)
        trainer.max_tool_calling_iterations = protocol.max_tool_calling_iterations
        trainer.max_consecutive_protocol_errors = protocol.max_consecutive_protocol_errors
        trainer.max_completion_length = protocol.max_completion_length
        trainer._tool_parallel_workers = 1
        trainer.use_vllm = False
        trainer.vllm_mode = "server"
        trainer._is_vlm = False
        trainer.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=protocol.max_model_length)
        )
        trainer._tokenizer = tokenizer
        trainer.environments = [environment]
        trainer._sync_tool_dicts = [
            {
                "execute_bash": environment.execute_bash,
                "str_replace_editor": environment.str_replace_editor,
                "finish": environment.finish,
            }
        ]
        trainer._async_tool_dicts = [{}]
        trainer._generate_single_turn = generator.generate
        _, completions, _, _, _, _, _ = trainer._tool_call_loop(
            prompts=[list(prompt)],
            prompt_ids=[prompt_ids],
            completion_ids=[first_ids[0]],
            completions=[[first]],
            logprobs=None,
            images=None,
            multimodal_fields={},
        )
        return environment.finalize_eval(completions[0], started)
    except BaseException as exc:
        environment._infrastructure_error = environment._infrastructure_error or exc
        return environment.finalize_eval([], started)


def build_predictions(
    outcomes: Sequence[EvalOutcome], *, model_name: str, task_ids: Sequence[str]
) -> list[dict[str, str]]:
    by_id = {outcome.task_id: outcome for outcome in outcomes}
    return [
        {
            "instance_id": task_id,
            "model_name_or_path": model_name,
            "model_patch": by_id[task_id].patch if task_id in by_id else "",
        }
        for task_id in task_ids
    ]


def build_comparison(
    *, candidate_report: Mapping[str, Any], base_report: Mapping[str, Any] | None
) -> dict[str, Any]:
    candidate = _report_summary(candidate_report)
    value: dict[str, Any] = {
        "candidate": candidate,
        "local_base": "not_run",
        "external_references": list(EXTERNAL_REFERENCES),
        "external_reference_caveat": (
            "External scaffold, budget, decoding, or exact revision may differ; "
            "these are not apples-to-apples LoRA deltas."
        ),
    }
    if base_report is not None:
        base = _report_summary(base_report)
        value["local_base"] = base
        if candidate["evaluated"] == candidate["total"] == base["evaluated"] == base["total"]:
            value["candidate_minus_local_base_percentage_points"] = round(
                100 * (candidate["resolve_rate"] - base["resolve_rate"]), 6
            )
        else:
            value["candidate_minus_local_base_percentage_points"] = "not_available_incomplete_grading"
    return value


def resolve_harness() -> tuple[Path, Path]:
    root = Path(os.environ.get("EVAL_HARNESS_ROOT", DEFAULT_HARNESS_ROOT)).expanduser().resolve()
    # harness Python 无默认值：必须由 EVAL_HARNESS_PYTHON 显式指定，缺失即失败关闭。
    harness_python_env = os.environ.get("EVAL_HARNESS_PYTHON", "").strip()
    if not harness_python_env:
        raise EvalError(
            "EVAL_HARNESS_PYTHON is not set; point it at a Python with swebench installed"
        )
    # 不能 resolve venv 的 Python symlink；解引用会丢失该 venv 的 site-packages。
    python = Path(harness_python_env).expanduser()
    if not python.is_absolute():
        python = (PROJECT_ROOT / python).absolute()
    if not (root / "swebench/harness/run_evaluation.py").is_file():
        raise EvalError(f"pinned SWE-bench harness checkout is missing: {root}")
    revision = _run_checked(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    dirty = _run_checked(
        ["git", "-C", str(root), "status", "--porcelain"], allow_empty=True
    ).strip()
    if revision != HARNESS_REVISION or dirty:
        raise EvalError(
            f"SWE-bench harness must be clean at {HARNESS_REVISION}; "
            f"got revision={revision}, dirty={bool(dirty)}"
        )
    if not python.is_file():
        raise EvalError(f"existing harness Python is missing: {python}")
    fixture = root / "swebench/harness/constants/fixtures/tokio-rs__tokio-6724.Cargo.lock"
    if not fixture.is_file():
        raise EvalError("pinned harness source checkout is missing packaged fixture")
    return root, python


def run_official_harness(
    *,
    output_dir: Path,
    predictions_path: Path | str,
    task_ids: Sequence[str],
    run_id: str,
    timeout_sec: int,
    protocol: EvalProtocol,
    max_workers: int = 1,
    harness_root: Path,
    harness_python: Path,
) -> dict[str, Any]:
    command = [
        str(harness_python),
        "-m",
        "siete_rl.swebench_harness",
        "--dataset_name",
        str(VERIFIED_PARQUET),
        "--split",
        "test",
        "--instance_ids",
        *task_ids,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout_sec),
        "--cache_level",
        "instance",
        "--clean",
        "false",
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--instance_image_tag",
        "latest",
        "--report_dir",
        str(output_dir),
    ]
    env = dict(os.environ)
    env["DOCKER_HOST"] = "unix:///run/docker-swegym/docker.sock"
    env["SIETE_HARNESS_CPUS"] = str(protocol.cpus)
    env["SIETE_HARNESS_MEMORY"] = protocol.memory
    env["SIETE_HARNESS_PIDS_LIMIT"] = str(protocol.pids_limit)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT / "src"), str(harness_root))
    )
    completed = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(600, timeout_sec * max(1, len(task_ids))),
        check=False,
    )
    (output_dir / "official-harness.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise EvalError(
            f"official harness failed with exit {completed.returncode}; "
            f"see {output_dir / 'official-harness.log'}"
        )
    reports = [
        path
        for path in output_dir.glob(f"*.{run_id}.json")
        if path.name != "official-report.json"
    ]
    if len(reports) != 1:
        raise EvalError(f"official harness produced {len(reports)} run reports")
    destination = output_dir / "official-report.json"
    shutil.move(str(reports[0]), destination)
    return _read_object(destination)


def build_vllm_command(
    *, run: EvalRun, port: int, lora: bool, model_path: Path | None = None
) -> list[str]:
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise EvalError(f"vllm executable is missing next to project Python: {executable}")
    command = [
        str(executable),
        "serve",
        str(model_path or run.base_model_path),
        "--served-model-name",
        "base" if lora else "candidate",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(run.protocol.max_model_length),
        "--gpu-memory-utilization",
        str(_eval_gpu_memory_utilization(run.protocol.gpu_memory_utilization)),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
    ]
    if lora:
        command.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"candidate={run.adapter_path}",
                "--max-lora-rank",
                str(run.rank),
            ]
        )
    return command


def run_adapter_reference(
    run: EvalRun, tokenizer: Any, prompt_ids: list[int]
) -> list[int]:
    """在 vLLM 启动前生成 HF+PEFT greedy 参考，并立即释放显存。"""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        run.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, run.adapter_path, is_trainable=False)
    values = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=values,
            do_sample=False,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = generated[0, values.shape[1] :].tolist()
    del generated, values, model
    gc.collect()
    torch.cuda.empty_cache()
    return answer


def merge_adapter(run: EvalRun, destination: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if destination.exists():
        raise EvalError(f"refusing to overwrite merged fallback: {destination}")
    model = AutoModelForCausalLM.from_pretrained(
        run.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    merged = PeftModel.from_pretrained(model, run.adapter_path, is_trainable=False).merge_and_unload()
    merged.save_pretrained(destination, safe_serialization=True)
    del merged, model
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def evaluate_variant(
    *,
    name: str,
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    run: EvalRun,
    tokenizer: Any,
    docker_client: SubprocessDockerClient,
    server_url: str,
    eval_run_id: str,
    output_dir: Path,
    rollout_workers: int = 1,
) -> list[EvalOutcome]:
    generator = HTTPTokenGenerator(
        base_url=server_url,
        model_name=model_name,
        max_completion_length=run.protocol.max_completion_length,
        max_model_length=run.protocol.max_model_length,
        seed=run.protocol.seed,
        repetition_penalty=run.protocol.repetition_penalty,
    )
    outcomes_by_index: dict[int, EvalOutcome] = {}

    def evaluate_sample(sample: Sample) -> EvalOutcome:
        return run_agent_loop(
            sample=sample,
            tokenizer=tokenizer,
            generator=generator,
            protocol=run.protocol,
            docker_client=docker_client,
            eval_run_id=eval_run_id,
        )

    def record(index: int, outcome: EvalOutcome) -> None:
        outcomes_by_index[index] = outcome
        ordered = [outcomes_by_index[key] for key in sorted(outcomes_by_index)]
        _write_json(
            output_dir / "rollout-state.json",
            {
                "variant": name,
                "completed": len(ordered),
                "total": len(rows),
                "outcomes": [_outcome_record(item) for item in ordered],
            },
        )

    if rollout_workers == 1:
        for index, row in enumerate(rows):
            sample = public_sample_from_row(row, run.protocol, docker_client)
            record(index, evaluate_sample(sample))
    elif rows:
        samples = [public_sample_from_row(row, run.protocol, docker_client) for row in rows]
        executor = ThreadPoolExecutor(max_workers=rollout_workers)
        future_indices = {}
        try:
            for index, sample in enumerate(samples):
                future_indices[executor.submit(evaluate_sample, sample)] = index
            for future in as_completed(future_indices):
                record(future_indices[future], future.result())
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    outcomes = [outcomes_by_index[index] for index in range(len(rows))]
    predictions = build_predictions(
        outcomes,
        model_name=model_name,
        task_ids=[_required_row_string(row, "instance_id") for row in rows],
    )
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    return outcomes


def preflight(*, gpu: str, harness_root: Path, harness_python: Path) -> dict[str, Any]:
    if not gpu or "," in gpu:
        raise EvalError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")
    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = executable_dir + os.pathsep + os.environ.get("PATH", "")
    sampler_backend = configure_sampler_backend()
    vllm_version = _run_checked([str(Path(sys.executable).with_name("vllm")), "--version"])
    ninja_version = _run_checked([str(Path(sys.executable).with_name("ninja")), "--version"])
    ninja_available = _run_checked(
        [
            sys.executable,
            "-c",
            (
                "import torch.utils.cpp_extension as cpp_extension; "
                "print(cpp_extension.is_ninja_available())"
            ),
        ]
    ).strip()
    if ninja_available != "True":
        raise EvalError("torch.utils.cpp_extension.is_ninja_available() is not True")
    gpu_state = _run_checked(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    return {
        "gpu": gpu,
        "gpu_state_before": gpu_state,
        "vllm_version": vllm_version.strip(),
        "ninja_version": ninja_version.strip(),
        "torch_ninja_available": True,
        "sampler_backend": sampler_backend,
        "sampler_backend_reason": (
            PYTORCH_SAMPLER_REASON if sampler_backend == "pytorch" else "explicit override"
        ),
        "harness_root": str(harness_root),
        "harness_python": str(harness_python),
        "harness_revision": HARNESS_REVISION,
    }


def execute(run_root: str | Path) -> Path:
    run = load_eval_run(run_root)
    eval_base = parse_strict_bool(os.environ.get("EVAL_BASE"), default=False)
    rollout_workers = _positive_env_int("EVAL_ROLLOUT_WORKERS")
    harness_workers = _positive_env_int("EVAL_HARNESS_WORKERS")
    task_ids_value = os.environ.get("EVAL_TASK_IDS")
    task_ids = None
    if task_ids_value is not None:
        task_ids = [value.strip() for value in task_ids_value.split(",") if value.strip()]
        if not task_ids:
            raise EvalError("EVAL_TASK_IDS must select at least one task")
    rows = load_verified_rows(task_ids)
    selected_ids = [_required_row_string(row, "instance_id") for row in rows]
    harness_root, harness_python = resolve_harness()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    preflight_record = preflight(gpu=gpu, harness_root=harness_root, harness_python=harness_python)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = run.root / "evals" / timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    eval_run_id = f"eval-{timestamp}-{uuid4().hex[:8]}"
    docker_client = SubprocessDockerClient()
    atexit.register(_atexit_sweep, docker_client, eval_run_id)
    metadata = {
        "status": "running",
        "run_root": str(run.root),
        "eval_run_id": eval_run_id,
        "started_at": _now(),
        "workspace": _git_metadata(PROJECT_ROOT),
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "revision": VERIFIED_REVISION,
            "parquet": str(VERIFIED_PARQUET),
            "selected_count": len(rows),
            "task_order": selected_ids,
        },
        "protocol": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": run.protocol.seed,
            "seed_guarantee": "recorded input; vLLM does not promise cross-process token identity",
            "rollouts_per_task": 1,
            "request_concurrency": rollout_workers,
            "official_harness_workers": harness_workers,
            "max_prompt_length": run.protocol.max_prompt_length,
            "max_completion_length": run.protocol.max_completion_length,
            "max_model_length": run.protocol.max_model_length,
            "max_tool_calling_iterations": run.protocol.max_tool_calling_iterations,
            "repetition_penalty": run.protocol.repetition_penalty,
        },
        "eval_base": eval_base,
        "preflight": preflight_record,
    }
    _write_json(output_root / "metadata.json", metadata)
    cleanup_error: BaseException | None = None
    active_servers: list[VLLMServer] = []
    try:
        gold_dir = output_root / "gold-smoke"
        gold_dir.mkdir()
        gold_report = run_official_harness(
            output_dir=gold_dir,
            predictions_path="gold",
            task_ids=[selected_ids[0]],
            run_id=f"{eval_run_id}-gold",
            timeout_sec=run.protocol.grader_timeout_sec,
            protocol=run.protocol,
            max_workers=harness_workers,
            harness_root=harness_root,
            harness_python=harness_python,
        )
        if int(gold_report.get("resolved_instances", 0)) != 1:
            raise EvalError("official gold-patch smoke was not resolved")
        metadata["gold_smoke"] = {"task_id": selected_ids[0], "resolved": True}

        tokenizer = AutoTokenizer.from_pretrained(
            run.base_model_path, trust_remote_code=True, local_files_only=True
        )
        tokenizer = install_openhands_tool_protocol(tokenizer)
        parity_prompt = build_prompt(public_sample_from_row(rows[0], run.protocol, docker_client).task)
        parity_prompt_ids = list(
            tokenizer.apply_chat_template(
                parity_prompt, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        )
        reference_ids = run_adapter_reference(run, tokenizer, parity_prompt_ids)
        lora_port = _free_port()
        lora_url = f"http://127.0.0.1:{lora_port}"
        lora_server = VLLMServer(
            build_vllm_command(run=run, port=lora_port, lora=True),
            server_gpu=gpu,
            base_url=lora_url,
            log_path=output_root / "vllm.log",
        )
        active_servers.append(lora_server)
        lora_server.start()
        parity_generator = HTTPTokenGenerator(
            base_url=lora_url,
            model_name="candidate",
            max_completion_length=32,
            max_model_length=run.protocol.max_model_length,
            seed=run.protocol.seed,
            repetition_penalty=run.protocol.repetition_penalty,
        )
        runtime_ids = parity_generator.generate([parity_prompt_ids])[0][0]
        parity_passed = runtime_ids == reference_ids
        metadata["adapter_parity"] = {
            "passed": parity_passed,
            "reference_token_ids": reference_ids,
            "runtime_lora_token_ids": runtime_ids,
        }
        base_report: dict[str, Any] | None = None
        if parity_passed:
            metadata["candidate_serve_mode"] = "runtime_lora"
            metadata["base_serve_mode"] = "shared-LoRA-engine base" if eval_base else "not_run"
            if eval_base:
                base_dir = output_root / "base"
                base_dir.mkdir()
                base_outcomes = evaluate_variant(
                    name="base",
                    model_name="base",
                    rows=rows,
                    run=run,
                    tokenizer=tokenizer,
                    docker_client=docker_client,
                    server_url=lora_url,
                    eval_run_id=eval_run_id,
                    output_dir=base_dir,
                    rollout_workers=rollout_workers,
                )
                _write_json(base_dir / "metadata.json", _variant_metadata("base", base_outcomes))
                base_report = run_official_harness(
                    output_dir=base_dir,
                    predictions_path=base_dir / "predictions.jsonl",
                    task_ids=selected_ids,
                    run_id=f"{eval_run_id}-base",
                    timeout_sec=run.protocol.grader_timeout_sec,
                    protocol=run.protocol,
                    max_workers=harness_workers,
                    harness_root=harness_root,
                    harness_python=harness_python,
                )
            candidate_dir = output_root / "candidate"
            candidate_dir.mkdir()
            candidate_outcomes = evaluate_variant(
                name="candidate",
                model_name="candidate",
                rows=rows,
                run=run,
                tokenizer=tokenizer,
                docker_client=docker_client,
                server_url=lora_url,
                eval_run_id=eval_run_id,
                output_dir=candidate_dir,
                rollout_workers=rollout_workers,
            )
            _write_json(
                candidate_dir / "metadata.json",
                _variant_metadata("candidate", candidate_outcomes),
            )
            candidate_report = run_official_harness(
                output_dir=candidate_dir,
                predictions_path=candidate_dir / "predictions.jsonl",
                task_ids=selected_ids,
                run_id=f"{eval_run_id}-candidate",
                timeout_sec=run.protocol.grader_timeout_sec,
                protocol=run.protocol,
                max_workers=harness_workers,
                harness_root=harness_root,
                harness_python=harness_python,
            )
            metadata["vllm_cleanup"] = lora_server.close()
        else:
            metadata["vllm_cleanup"] = lora_server.close()
            merged_path = merge_adapter(run, output_root / "candidate" / "merged-model")
            metadata["candidate_serve_mode"] = "merged_model_fallback"
            metadata["base_serve_mode"] = "serial base-only server" if eval_base else "not_run"
            metadata["fallback_protocol_consequence"] = (
                "base and final are served serially with the same vLLM version, greedy requests, "
                "context, task order and engine limits; base server omits LoRA routing."
            )
            base_report = None
            if eval_base:
                base_report = _run_standalone_variant(
                    name="base",
                    model_path=run.base_model_path,
                    rows=rows,
                    run=run,
                    tokenizer=tokenizer,
                    docker_client=docker_client,
                    eval_run_id=eval_run_id,
                    output_root=output_root,
                    gpu=gpu,
                    harness_root=harness_root,
                    harness_python=harness_python,
                    rollout_workers=rollout_workers,
                    harness_workers=harness_workers,
                )
            candidate_report = _run_standalone_variant(
                name="candidate",
                model_path=merged_path,
                rows=rows,
                run=run,
                tokenizer=tokenizer,
                docker_client=docker_client,
                eval_run_id=eval_run_id,
                output_root=output_root,
                gpu=gpu,
                harness_root=harness_root,
                harness_python=harness_python,
                rollout_workers=rollout_workers,
                harness_workers=harness_workers,
            )
        comparison = build_comparison(
            candidate_report=candidate_report, base_report=base_report
        )
        _write_json(output_root / "comparison.json", comparison)
        metadata["status"] = "completed"
        metadata["finished_at"] = _now()
        _write_json(output_root / "metadata.json", metadata)
        return output_root
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        metadata["finished_at"] = _now()
        _write_json(output_root / "metadata.json", metadata)
        raise
    finally:
        server_handles = []
        for server in reversed(active_servers):
            if server.pid is not None:
                server_handles.append(server.close())
        if server_handles:
            metadata["vllm_final_sweep"] = server_handles
        try:
            removed = sweep_run_containers(docker_client, eval_run_id)
            metadata["container_sweep"] = {"removed": removed, "residual": False}
        except BaseException as exc:
            cleanup_error = exc
            metadata["container_sweep"] = {
                "removed": [],
                "residual": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_json(output_root / "metadata.json", metadata)
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


def _run_standalone_variant(
    *,
    name: str,
    model_path: Path,
    rows: Sequence[Mapping[str, Any]],
    run: EvalRun,
    tokenizer: Any,
    docker_client: SubprocessDockerClient,
    eval_run_id: str,
    output_root: Path,
    gpu: str,
    harness_root: Path,
    harness_python: Path,
    rollout_workers: int,
    harness_workers: int,
) -> dict[str, Any]:
    output_dir = output_root / name
    output_dir.mkdir(exist_ok=True)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = VLLMServer(
        build_vllm_command(run=run, port=port, lora=False, model_path=model_path),
        server_gpu=gpu,
        base_url=base_url,
        log_path=output_dir / "vllm.log",
    )
    try:
        server.start()
        outcomes = evaluate_variant(
            name=name,
            model_name="candidate",
            rows=rows,
            run=run,
            tokenizer=tokenizer,
            docker_client=docker_client,
            server_url=base_url,
            eval_run_id=eval_run_id,
            output_dir=output_dir,
            rollout_workers=rollout_workers,
        )
        metadata = _variant_metadata(name, outcomes)
        metadata["vllm_cleanup"] = server.close()
        _write_json(output_dir / "metadata.json", metadata)
    finally:
        if server.pid is not None:
            server.close()
    return run_official_harness(
        output_dir=output_dir,
        predictions_path=output_dir / "predictions.jsonl",
        task_ids=[_required_row_string(row, "instance_id") for row in rows],
        run_id=f"{eval_run_id}-{name}",
        timeout_sec=run.protocol.grader_timeout_sec,
        protocol=run.protocol,
        max_workers=harness_workers,
        harness_root=harness_root,
        harness_python=harness_python,
    )


def _variant_metadata(name: str, outcomes: Sequence[EvalOutcome]) -> dict[str, Any]:
    return {
        "variant": name,
        "outcomes": [_outcome_record(outcome) for outcome in outcomes],
        "empty_patch_count": sum(not outcome.patch.strip() for outcome in outcomes),
        "infrastructure_failure_count": sum(
            outcome.infrastructure_error is not None for outcome in outcomes
        ),
    }


def _outcome_record(outcome: EvalOutcome) -> dict[str, Any]:
    return {
        "instance_id": outcome.task_id,
        "termination": outcome.termination,
        "patch_empty": not outcome.patch.strip(),
        "infrastructure_error": outcome.infrastructure_error,
        "duration_sec": round(outcome.duration_sec, 6),
        "messages": outcome.messages,
    }


def _report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    total = int(report.get("total_instances", 0))
    evaluated = int(report.get("completed_instances", 0)) + int(
        report.get("empty_patch_instances", 0)
    )
    resolved = int(report.get("resolved_instances", 0))
    return {
        "total": total,
        "evaluated": evaluated,
        "resolved": resolved,
        "resolve_rate": resolved / total if total else 0.0,
        "empty_patch": int(report.get("empty_patch_instances", 0)),
        "infrastructure_errors": int(report.get("error_instances", 0)),
    }


def _post_json(url: str, payload: Mapping[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout_sec) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise EvalError(f"vLLM request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError("vLLM response must be a JSON object")
    return value


def _inspect_official_image(client: SubprocessDockerClient, image_name: str) -> dict[str, Any]:
    result = client.run(["docker", "image", "inspect", image_name], timeout_sec=30)
    if result.exit_code != 0 or result.timed_out:
        raise EvalError(f"official image is unavailable locally: {image_name}")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EvalError("docker image inspect returned invalid JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise EvalError("docker image inspect returned an invalid structure")
    payload = values[0]
    if payload.get("Os") != "linux" or payload.get("Architecture") != "amd64":
        raise EvalError("official image platform must be linux/amd64")
    return payload


def _git_metadata(path: Path) -> dict[str, Any]:
    return {
        "commit": _run_checked(["git", "-C", str(path), "rev-parse", "HEAD"]).strip(),
        "dirty": bool(
            _run_checked(
                ["git", "-C", str(path), "status", "--porcelain"], allow_empty=True
            ).strip()
        ),
    }


def _run_checked(command: Sequence[str], *, allow_empty: bool = False) -> str:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvalError(f"preflight command failed: {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise EvalError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    if not allow_empty and not completed.stdout.strip():
        raise EvalError(f"command returned empty output: {' '.join(command)}")
    return completed.stdout


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"failed to read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError(f"JSON must contain an object: {path}")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvalError(f"failed to read run config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError(f"run config must contain an object: {path}")
    return value


def _field(config: Mapping[str, Any], section: str, name: str, *, expected: type) -> Any:
    group = config.get(section)
    if not isinstance(group, dict) or name not in group:
        raise EvalError(f"run config is missing {section}.{name}; refusing to use a default")
    value = group[name]
    if not isinstance(value, expected) or isinstance(value, bool):
        raise EvalError(f"run config field {section}.{name} has the wrong type")
    return value


def _positive_int(config: Mapping[str, Any], section: str, name: str) -> int:
    value = _field(config, section, name, expected=int)
    if value < 1:
        raise EvalError(f"run config field {section}.{name} must be positive")
    return value


def _nonnegative_int(config: Mapping[str, Any], section: str, name: str) -> int:
    value = _field(config, section, name, expected=int)
    if value < 0:
        raise EvalError(f"run config field {section}.{name} must be nonnegative")
    return value


def _positive_number(config: Mapping[str, Any], section: str, name: str) -> float:
    group = config.get(section)
    if not isinstance(group, dict) or name not in group:
        raise EvalError(f"run config is missing {section}.{name}; refusing to use a default")
    value = group[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise EvalError(f"run config field {section}.{name} must be positive")
    return float(value)


def _required_row_string(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise EvalError(f"Verified row is missing public field {name}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _eval_gpu_memory_utilization(configured: float) -> float:
    value = os.environ.get("EVAL_GPU_MEMORY_UTILIZATION")
    if value is None:
        return configured
    try:
        parsed = float(value)
    except ValueError as exc:
        raise EvalError("EVAL_GPU_MEMORY_UTILIZATION must be a number") from exc
    if not 0 < parsed <= 1:
        raise EvalError("EVAL_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    return parsed


def configure_sampler_backend() -> str:
    """显式选择全会话 sampler；base/final 与 parity 子进程继承同一值。"""

    value = os.environ.get("EVAL_SAMPLER_BACKEND", "pytorch")
    if value == "pytorch":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        return value
    if value == "flashinfer":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        return value
    raise EvalError("EVAL_SAMPLER_BACKEND must be exactly 'pytorch' or 'flashinfer'")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atexit_sweep(client: SubprocessDockerClient, run_id: str) -> None:
    try:
        sweep_run_containers(client, run_id)
    except BaseException:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: scripts/eval.sh <run-dir>   # outputs/ 或 _archive/ 下的 run 目录", file=sys.stderr)
        return 2
    try:
        output = execute(arguments[0])
    except BaseException as exc:
        print(f"eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
