"""TRL 训练入口、真实单步闭环与 run-scoped 资源收束。"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from swe_agent.config import ProjectConfig, load_config


class TrainingNotReadyError(RuntimeError):
    """当前阶段尚未具备启动真实训练所需的项目模块。"""


class RuntimeNotQualifiedError(RuntimeError):
    """配置明确禁止启动尚未资格化的模型路径。"""


class RecordingRuntimeError(RuntimeError):
    """主进程无法按固定合同记录一次 generation group。"""


class TrainingInterrupted(BaseException):
    """完成 run 级清理并携带终端摘要退出薄 signal 边界。"""

    def __init__(self, signum: int, report: dict[str, Any]) -> None:
        super().__init__(f"training interrupted by signal {signum}")
        self.signum = signum
        self.report = report


REQUIRED_DOMAIN_MODULES = (
    "models.py",
    "swegym.py",
    "prompts.py",
    "docker.py",
    "tools.py",
    "verifier.py",
    "environment.py",
    "rewards.py",
    "recording.py",
    "trainer.py",
    "launcher.py",
)


def preflight(config: ProjectConfig, project_root: Path) -> dict[str, Any]:
    """执行便宜、确定且不创建输出目录的启动前检查。"""

    if not config.runtime.runtime_qualified:
        raise RuntimeNotQualifiedError(
            f"模型 {config.model.provenance_id} 的 runtime_qualified=false，禁止真实启动"
        )

    model_path = Path(config.model.model_path)
    tokenizer_path = Path(config.model.tokenizer_path)
    model_config_path = model_path / "config.json"
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    for required in (model_config_path, tokenizer_config_path):
        if not required.is_file():
            raise TrainingNotReadyError(f"preflight 缺少本地模型文件: {required}")

    try:
        model_payload = json.loads(model_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingNotReadyError(f"无法读取模型配置 {model_config_path}: {exc}") from exc
    architectures = model_payload.get("architectures")
    if not isinstance(architectures, list) or config.model.architecture not in architectures:
        raise TrainingNotReadyError(
            "配置 architecture 与本地模型 config.json 不一致: "
            f"expected={config.model.architecture}, actual={architectures}"
        )

    package_root = project_root / "src" / "swe_agent"
    missing_modules = [
        name for name in REQUIRED_DOMAIN_MODULES if not (package_root / name).is_file()
    ]
    return {
        "status": "preflight_passed",
        "task_id": config.dataset.task_id,
        "model": config.model.provenance_id,
        "model_path": model_path.resolve().as_posix(),
        "vllm_mode": config.vllm.mode,
        "vllm_tensor_parallel_size": config.vllm.tensor_parallel_size,
        "missing_domain_modules": missing_modules,
    }


def build_peft_config(config: ProjectConfig) -> Any:
    """把稳定 YAML 合同映射为 PEFT 公共配置。"""

    from peft import LoraConfig

    return LoraConfig(
        task_type=config.peft.task_type,
        r=config.peft.rank,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        bias=config.peft.bias,
        target_modules=list(config.peft.target_modules),
        modules_to_save=config.peft.modules_to_save,
    )


def build_quantization_config(config: ProjectConfig) -> Any | None:
    """7B 返回 None；未来 QLoRA 路径只构造公共 BNB 配置对象。"""

    if not config.quantization.load_in_4bit:
        return None

    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )


def build_grpo_config(
    config: ProjectConfig,
    output_dir: str | Path,
    *,
    seed: int,
    use_cpu: bool = False,
) -> Any:
    """只映射计划已冻结的公共 GRPOConfig 字段。"""

    from trl import GRPOConfig

    grpo = config.grpo
    generation = config.generation
    vllm = config.vllm
    return GRPOConfig(
        output_dir=Path(output_dir).as_posix(),
        model_init_kwargs={"dtype": config.model.dtype},
        trust_remote_code=config.model.trust_remote_code,
        seed=seed,
        data_seed=seed,
        use_cpu=use_cpu,
        bf16=False if use_cpu else grpo.bf16,
        fp16=False,
        per_device_train_batch_size=grpo.per_device_train_batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        generation_batch_size=grpo.generation_batch_size,
        steps_per_generation=grpo.steps_per_generation,
        max_steps=grpo.max_steps,
        learning_rate=grpo.learning_rate,
        weight_decay=grpo.weight_decay,
        max_grad_norm=grpo.max_grad_norm,
        gradient_checkpointing=grpo.gradient_checkpointing,
        logging_steps=grpo.logging_steps,
        save_strategy=grpo.save_strategy,
        save_steps=grpo.save_steps,
        save_total_limit=grpo.save_total_limit,
        report_to=list(grpo.report_to),
        num_generations=grpo.num_generations,
        num_iterations=grpo.num_iterations,
        max_completion_length=generation.max_completion_length,
        max_tool_calling_iterations=generation.max_tool_calling_iterations,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        repetition_penalty=generation.repetition_penalty,
        vllm_structured_outputs_regex=generation.structured_outputs_regex,
        beta=grpo.beta,
        epsilon=grpo.epsilon,
        epsilon_high=grpo.epsilon_high,
        delta=grpo.delta,
        importance_sampling_level=grpo.importance_sampling_level,
        multi_objective_aggregation=grpo.multi_objective_aggregation,
        scale_rewards=grpo.scale_rewards,
        loss_type=grpo.loss_type,
        mask_truncated_completions=grpo.mask_truncated_completions,
        router_aux_loss_coef=grpo.router_aux_loss_coef,
        shuffle_dataset=grpo.shuffle_dataset,
        vllm_importance_sampling_correction=grpo.vllm_importance_sampling_correction,
        vllm_importance_sampling_mode=grpo.vllm_importance_sampling_mode,
        vllm_importance_sampling_clip_max=grpo.vllm_importance_sampling_clip_max,
        vllm_importance_sampling_clip_min=grpo.vllm_importance_sampling_clip_min,
        log_completions=grpo.log_completions,
        use_vllm=vllm.use_vllm,
        vllm_mode=vllm.mode,
        vllm_server_base_url=vllm.server_base_url,
        vllm_model_impl=vllm.model_impl,
        vllm_enable_sleep_mode=vllm.enable_sleep_mode,
        vllm_tensor_parallel_size=vllm.tensor_parallel_size or 1,
        vllm_gpu_memory_utilization=vllm.gpu_memory_utilization,
        vllm_max_model_length=vllm.max_model_length,
    )


def build_trainer(
    config: ProjectConfig,
    *,
    output_dir: str | Path,
    seed: int,
    train_dataset: Any,
    environment_factory: Callable[[], Any],
    reward_func: Callable[..., list[float]],
    processing_class: Any | None = None,
) -> Any:
    """构造 SWEGRPOTrainer（GRPOTrainer 子类，加入环境信号终止）。"""

    from swe_agent.trainer import SWEGRPOTrainer

    return SWEGRPOTrainer(
        model=config.model.model_path,
        args=build_grpo_config(config, output_dir, seed=seed),
        train_dataset=train_dataset,
        processing_class=processing_class,
        environment_factory=environment_factory,
        reward_funcs=reward_func,
        peft_config=build_peft_config(config),
        quantization_config=build_quantization_config(config),
        max_consecutive_format_errors=config.generation.max_consecutive_format_errors,
    )


def build_processing_class(config: ProjectConfig) -> Any:
    """构造正式 Trainer 与协议测试共享的 tokenizer/response-schema 对象。"""

    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema

    from swe_agent.tool_protocol import install_compatible_tool_call_parser

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path,
        local_files_only=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = add_response_schema(tokenizer)
    return install_compatible_tool_call_parser(tokenizer)


def run(config_path: str | Path) -> dict[str, Any]:
    """一次 CLI 调用只创建并执行一个 run。"""

    config, project_root, resolved_config_path = load_config(config_path)
    report = preflight(config, project_root)
    report["config_path"] = resolved_config_path.as_posix()
    if report["missing_domain_modules"]:
        missing = ", ".join(report["missing_domain_modules"])
        raise TrainingNotReadyError(
            "preflight 已通过模型与运行配置检查，但领域实现尚未完成，"
            f"拒绝启动真实训练；缺少: {missing}"
        )

    from swe_agent.launcher import split_visible_gpus

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    server_gpu = split_visible_gpus(config)
    physical_device = _require_single_visible_gpu()
    report = _run_once(
        config=config,
        project_root=project_root,
        seed=config.runtime.base_seed,
        physical_device=physical_device,
        server_gpu=server_gpu,
    )
    if report["lifecycle"] == "interrupted":
        raise TrainingInterrupted(int(report["interrupted_signum"]), report)
    return report


def _run_once(
    *,
    config: ProjectConfig,
    project_root: Path,
    seed: int,
    physical_device: int,
    server_gpu: str | None,
) -> dict[str, Any]:
    """创建并终结唯一 output_dir；所有项目 JSON 只由本函数驱动。"""

    from swe_agent.docker import DockerSandbox, SubprocessDockerClient, sweep_run_containers
    from swe_agent.environment import SWEEnvironment
    from swe_agent.launcher import VLLMServer, build_server_command
    from swe_agent.recording import RunRecorder
    from swe_agent.rewards import binary_reward
    from swe_agent.swegym import build_training_dataset, load_task_context
    from swe_agent.verifier import SWEGymVerifier

    commit, dirty = _code_provenance(project_root)
    recorder = RunRecorder(
        config=config,
        seed=seed,
        run_id=os.environ.get("SWE_AGENT_RUN_ID"),
        code_commit=commit,
        code_dirty=dirty,
        model_revision=_model_revision(Path(config.model.model_path)),
    )
    recorder.log(f"run started seed={seed} device={physical_device}")

    trainer: Any | None = None
    environments: list[SWEEnvironment] = []
    docker_client: SubprocessDockerClient | None = None
    vllm_server: VLLMServer | None = None
    gpu_baseline: dict[str, int] | None = None
    child_process_baseline = _snapshot_child_processes()
    run_processes: dict[int, dict[str, Any]] = {}
    failure: dict[str, str] | None = None
    stage = "gpu_preflight"
    trainer_group_consumed = False
    lora_changed = False
    metrics: dict[str, float | None] = {}
    interrupted_signum: int | None = None

    try:
        if server_gpu is not None:
            stage = "vllm_server"
            if config.vllm.server_base_url is None:
                raise RuntimeError("vllm server mode requires vllm.server_base_url")
            vllm_server = VLLMServer(
                build_server_command(config),
                server_gpu=server_gpu,
                base_url=config.vllm.server_base_url,
                log_path=recorder.output_dir / "vllm.log",
            )
            vllm_server.start()
            recorder.log(f"vLLM server ready pid={vllm_server.pid}")
        gpu_baseline = _gpu_baseline(physical_device)
        stage = "load_task"
        task_context = load_task_context(config, project_root)
        sample, _ = task_context[config.dataset.task_id]
        dataset = build_training_dataset(sample)
        prompt = dataset[0]["prompt"]
        recorder.prepare_first_group(prompt)

        docker_client = SubprocessDockerClient()

        def sandbox_factory(sample_arg, episode_id: str, scope: str):
            return DockerSandbox(
                client=docker_client,
                task=sample_arg.task,
                environment=sample_arg.environment,
                run_id=recorder.run_id,
                episode_id=episode_id,
                scope=scope,
            )

        def verifier_factory(evaluation, episode_id: str):
            return SWEGymVerifier(
                sandbox_factory=lambda: sandbox_factory(sample, episode_id, "verifier"),
                evaluation=evaluation,
            )

        def environment_factory() -> SWEEnvironment:
            environment = SWEEnvironment(
                task_context=task_context,
                sandbox_factory=sandbox_factory,
                verifier_factory=verifier_factory,
                output_limit_chars=config.chat.max_observation_chars,
                max_timeout_sec=config.docker.exec_timeout_sec,
            )
            environments.append(environment)
            return environment

        reward_func = _recording_reward(recorder, binary_reward)
        stage = "tokenizer"
        tokenizer = build_processing_class(config)
        stage = "trainer_construct"
        trainer = build_trainer(
            config,
            output_dir=recorder.output_dir,
            seed=seed,
            train_dataset=dataset,
            environment_factory=environment_factory,
            reward_func=reward_func,
            processing_class=tokenizer,
        )
        run_processes.update(_new_child_processes(child_process_baseline))
        # vLLM 的 in-process engine 构造会使用自己的固定 seed；恢复本 run seed。
        from transformers import set_seed

        set_seed(seed)
        _validate_rendered_prompt_length(trainer, prompt, config.chat.max_prompt_length)
        pre_step_adapter = _adapter_state(trainer.model)
        recorder.log("trainer constructed; rollout policy global_step=0")

        stage = "train"
        train_output = trainer.train()
        global_step = int(trainer.state.global_step)
        if global_step != 1:
            raise RuntimeError(f"expected trainer.state.global_step=1, got {global_step}")
        recorder.complete_batch(global_step)
        trainer_group_consumed = True
        post_step_adapter = _adapter_state(trainer.model)
        lora_changed = _state_digest(pre_step_adapter) != _state_digest(post_step_adapter)

        stage = "save_model"
        trainer.save_model(recorder.output_dir.as_posix())
        _require_final_model_files(recorder.output_dir)
        if not _verify_saved_adapter(recorder.output_dir, post_step_adapter):
            raise RuntimeError("saved adapter could not be reloaded with identical tensors")
        checkpoints = sorted(
            path.name
            for path in recorder.output_dir.glob("checkpoint-*")
            if path.is_dir()
        )
        if checkpoints != ["checkpoint-1"]:
            raise RuntimeError(f"expected exactly checkpoint-1, got {checkpoints}")
        metrics = _training_metrics(trainer, train_output)
        recorder.update_training(
            loss=metrics["loss"],
            grad_norm=metrics["grad_norm"],
            frac_reward_zero_std=metrics["frac_reward_zero_std"],
            checkpoints=checkpoints,
            final_model_ref="adapter_model.safetensors",
        )
        recorder.update_observations(
            nonzero_advantage_observed=(
                False if recorder.group and recorder.group["degenerate"] else True
            ),
            nonzero_gradient_observed=_is_positive_finite(metrics["grad_norm"]),
            nonzero_parameter_update_observed=lora_changed,
            all_sequences_masked_by_is=(
                None
                if metrics["importance_sampling_ratio_max"] is None
                else metrics["importance_sampling_ratio_max"] == 0.0
            ),
        )
        recorder.log(
            "GRPO optimizer step completed; post-step policy is "
            + ("numerically changed" if lora_changed else "numerically unchanged")
        )
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt) or type(exc).__name__ == "WorkflowTermination"
        if interrupted:
            interrupted_signum = int(getattr(exc, "signum", 2))
        failure = {
            "category": "interrupted" if interrupted else _failure_category(exc, stage),
            "primary_type": type(exc).__name__,
            "message": str(exc),
            "stage": stage,
            "interrupted": str(interrupted),
        }
        recorder.log(traceback.format_exc())
    finally:
        run_processes.update(_new_child_processes(child_process_baseline))
        cleanup_errors, environment_handles = _close_environments(environments, recorder)
        server_handle = vllm_server.close() if vllm_server is not None else None
        if docker_client is not None:
            try:
                swept = sweep_run_containers(docker_client, recorder.run_id)
                if swept:
                    recorder.log(f"swept orphan containers: {', '.join(swept)}")
            except BaseException as sweep_error:
                cleanup_errors.append(sweep_error)
        trainer_to_release, trainer = trainer, None
        trainer_refs = _trainer_weakrefs(trainer_to_release)
        trainer_errors, trainer_handles = _release_trainer(trainer_to_release, recorder)
        cleanup_errors.extend(trainer_errors)
        trainer_to_release = None
        gc.collect()
        _log_live_trainer_cuda_references(trainer_refs, recorder)
        processes, process_errors = _finalize_run_processes(run_processes)
        cleanup_errors.extend(process_errors)
        gpu_diagnostic = _finalize_gpu_diagnostic(physical_device, gpu_baseline)
        recorder.set_processes(processes)
        recorder.set_runtime_handles(
            [
                *environment_handles,
                *([server_handle] if server_handle is not None else []),
                *trainer_handles,
            ]
        )
        recorder.set_gpu_diagnostics([gpu_diagnostic])
        recorder.finalize_cleanup()
        if recorder.run["cleanup"]["state"] == "failed" and not cleanup_errors:
            cleanup_errors.append(RuntimeError("one or more run-owned resources remain"))
        if cleanup_errors:
            for cleanup_error in cleanup_errors:
                recorder.log(
                    f"cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            if failure is None:
                first = cleanup_errors[0]
                failure = {
                    "category": "cleanup",
                    "primary_type": type(first).__name__,
                    "message": str(first),
                    "stage": "cleanup",
                    "interrupted": "False",
                }

    native_policy_path_reached = _native_policy_path_reached(environments)
    system_passed = native_policy_path_reached and trainer_group_consumed
    recorder.update_training(
        system_closed_loop="passed" if system_passed else "failed",
        native_policy_path_reached=native_policy_path_reached,
        trainer_group_consumed=trainer_group_consumed,
    )
    if failure is not None:
        recorder.fail(
            category=failure["category"],
            primary_type=failure["primary_type"],
            message=failure["message"],
            stage=failure["stage"],
            interrupted=failure["interrupted"] == "True",
        )
    else:
        recorder.complete()
    recorder.log(
        f"run finished lifecycle={recorder.run['lifecycle']['state']} "
        f"native_policy_path_reached={native_policy_path_reached} "
        f"trainer_group_consumed={trainer_group_consumed} "
        f"system_closed_loop={recorder.run['training']['system_closed_loop']}"
    )
    return {
        "run_id": recorder.run_id,
        "lifecycle": recorder.run["lifecycle"]["state"],
        "native_policy_path_reached": native_policy_path_reached,
        "trainer_group_consumed": trainer_group_consumed,
        "system_closed_loop": recorder.run["training"]["system_closed_loop"],
        "failure": recorder.run["failure"],
        "final_model_ref": recorder.run["training"]["final_model_ref"],
        "cleanup": {
            "state": recorder.run["cleanup"]["state"],
            "clean_release": recorder.run["cleanup"]["clean_release"],
            "residuals": recorder.run["cleanup"]["residuals"],
        },
        "interrupted_signum": interrupted_signum,
    }


def _recording_reward(recorder: Any, reward_adapter: Callable[..., list[float]]):
    recorded = False

    def reward(
        prompts: list[object],
        completions: list[object],
        environments: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        nonlocal recorded
        if recorded:
            raise RecordingRuntimeError("first-stage group reward was invoked more than once")
        if not (len(prompts) == len(completions) == len(environments) == 4):
            raise RecordingRuntimeError("first-stage reward requires four aligned rollouts")
        rewards: list[float] = []
        try:
            rewards = reward_adapter(completions=completions, environments=environments, **kwargs)
            for index, (prompt, completion, environment) in enumerate(
                zip(prompts, completions, environments, strict=True)
            ):
                recorder.write_rollout(
                    index,
                    messages=_join_messages(prompt, completion),
                    trajectory=environment.trajectory,
                    patch=environment.frozen_patch,
                    verification=environment.verification,
                )
            recorder.complete_group(
                episode_ids=[environment.episode_id for environment in environments],
                rewards=rewards,
                verifications=[environment.verification for environment in environments],
            )
            recorded = True
            return rewards
        except RecordingRuntimeError:
            raise
        except BaseException:
            for index, (prompt, completion, environment) in enumerate(
                zip(prompts, completions, environments, strict=True)
            ):
                if environment.trajectory is not None:
                    try:
                        recorder.write_rollout(
                            index,
                            messages=_join_messages(prompt, completion),
                            trajectory=environment.trajectory,
                            patch=environment.frozen_patch,
                            verification=environment.verification,
                        )
                    except BaseException:
                        pass
            raise
        finally:
            for environment in environments:
                recorder.merge_cleanup_events(environment._drain_events())

    reward.__name__ = "binary_reward"
    return reward


def _join_messages(prompt: object, completion: object) -> list[object]:
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise RecordingRuntimeError("TRL structured prompt/completion must both be lists")
    return [*prompt, *completion]


def _require_single_visible_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    value = visible.strip() if visible is not None else ""
    if not value.isdecimal() or "," in value:
        raise RuntimeNotQualifiedError(
            "CUDA_VISIBLE_DEVICES must explicitly select exactly one non-negative integer GPU index"
        )
    physical_device = int(value)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    return physical_device


def _gpu_baseline(physical_device: int) -> dict[str, int]:
    import torch
    import vllm._C  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose exactly one usable CUDA device")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"visible CUDA device is not an A100: {torch.cuda.get_device_name(0)}")
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated(0)),
        "reserved": int(torch.cuda.memory_reserved(0)),
        "owner_pid": os.getpid(),
        "physical_device": physical_device,
    }


def _finalize_gpu_diagnostic(
    physical_device: int, baseline: dict[str, int] | None
) -> dict[str, Any]:
    """记录主进程退出前的 allocator 观察；该结果不是资源释放门禁。"""

    allocated_after: int | None = None
    reserved_after: int | None = None
    note: str | None = None
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_after = int(torch.cuda.memory_allocated(0))
            reserved_after = int(torch.cuda.memory_reserved(0))
    except BaseException as exc:
        note = f"{type(exc).__name__}: {exc}"
    baseline_allocated = baseline["allocated"] if baseline else None
    baseline_reserved = baseline["reserved"] if baseline else None
    if note is None and baseline is None:
        note = "run GPU baseline was unavailable"
    return {
        "device": str(physical_device),
        "owner_pid": baseline["owner_pid"] if baseline else os.getpid(),
        "allocated_bytes_before": baseline_allocated,
        "reserved_bytes_before": baseline_reserved,
        "allocated_bytes_after": allocated_after,
        "reserved_bytes_after": reserved_after,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "observed_at": _utc_now(),
        "diagnostic_only": True,
        "note": note,
    }


def _close_environments(
    environments: list[Any], recorder: Any
) -> tuple[list[BaseException], list[dict[str, Any]]]:
    errors: list[BaseException] = []
    handles: list[dict[str, Any]] = []
    for index, environment in enumerate(environments):
        operations: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for sequence in range(1, 3):
            try:
                environment._close()
                operations.append(_cleanup_operation(sequence, "close", "success"))
            except BaseException as exc:
                last_error = exc
                operations.append(
                    _cleanup_operation(sequence, "close", "failed", str(exc))
                )
            finally:
                recorder.merge_cleanup_events(environment._drain_events())
            if environment._sandbox is None and environment._verifier is None:
                break
        residual = environment._sandbox is not None or environment._verifier is not None
        if residual:
            errors.append(last_error or RuntimeError("environment handle remains open"))
        handles.append(
            {
                "scope": "environment",
                "identifier": environment.episode_id or f"environment-{index}",
                "operations": operations,
                "final_state": "residual" if residual else "closed",
                "residual": residual,
            }
        )
    return errors, handles


def _release_trainer(
    trainer: Any | None, recorder: Any
) -> tuple[list[BaseException], list[dict[str, Any]]]:
    if trainer is None:
        return [], [
            _not_initialized_handle("trainer", "grpo_trainer"),
            _not_initialized_handle("model", "policy_model"),
            _not_initialized_handle("vllm_engine", "colocate_engine"),
        ]
    errors: list[BaseException] = []
    handles: list[dict[str, Any]] = []
    vllm_model_ref: weakref.ReferenceType[Any] | None = None
    backend = getattr(trainer, "vllm_generation", None)
    llm = getattr(backend, "llm", None)
    vllm_error: BaseException | None = None
    vllm_shutdown_returned = False
    try:
        if llm is not None:
            vllm_model_ref = _vllm_model_weakref(llm)
        if llm is not None and getattr(backend, "enable_sleep_mode", False):
            try:
                llm.sleep(level=2)
            except BaseException as exc:
                recorder.log(f"vLLM sleep during cleanup failed: {type(exc).__name__}: {exc}")
        if llm is not None:
            try:
                _clear_vllm_cuda_graphs(llm)
            except BaseException as exc:
                recorder.log(
                    f"vLLM CUDA graph cleanup diagnostic failed: {type(exc).__name__}: {exc}"
                )
            llm.llm_engine.engine_core.shutdown()
            vllm_shutdown_returned = True
            _detach_vllm_engine(llm)
    except BaseException as exc:
        vllm_error = exc
    if llm is None:
        handles.append(_not_initialized_handle("vllm_engine", "colocate_engine"))
    else:
        vllm_residual = not vllm_shutdown_returned or vllm_error is not None
        handles.append(
            {
                "scope": "vllm_engine",
                "identifier": "colocate_engine",
                "operations": [
                    _cleanup_operation(
                        1,
                        "shutdown",
                        "failed" if vllm_residual else "success",
                        str(vllm_error) if vllm_error is not None else None,
                    )
                ],
                "final_state": "residual" if vllm_residual else "closed",
                "residual": vllm_residual,
            }
        )
        if vllm_residual:
            errors.append(vllm_error or RuntimeError("vLLM shutdown did not return"))

    model_error: BaseException | None = None
    try:
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            optimizer.param_groups.clear()
        scheduler = getattr(trainer, "lr_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "optimizer"):
            scheduler.optimizer = None
        trainer.model.to("cpu")
    except BaseException as exc:
        model_error = exc
    handles.append(
        {
            "scope": "model",
            "identifier": "policy_model",
            "operations": [
                _cleanup_operation(
                    1,
                    "release",
                    "failed" if model_error is not None else "success",
                    str(model_error) if model_error is not None else None,
                )
            ],
            "final_state": "residual" if model_error is not None else "released",
            "residual": model_error is not None,
        }
    )
    if model_error is not None:
        errors.append(model_error)

    trainer_error: BaseException | None = None
    try:
        model = getattr(trainer, "model", None)
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)
        released = trainer.accelerator.free_memory(model, optimizer, scheduler)
        trainer.model, trainer.optimizer, trainer.lr_scheduler = released
        trainer.model_wrapped = None
        trainer.ref_model = None
        trainer._buffered_inputs = None
        if backend is not None:
            backend.model = None
            backend.llm = None
        callback_handler = getattr(trainer, "callback_handler", None)
        if callback_handler is not None:
            callback_handler.model = None
            callback_handler.optimizer = None
            callback_handler.lr_scheduler = None
    except BaseException as exc:
        trainer_error = exc
    try:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except BaseException as exc:
        if trainer_error is None:
            trainer_error = exc
        else:
            trainer_error.add_note(
                f"distributed cleanup failed: {type(exc).__name__}: {exc}"
            )
    handles.append(
        {
            "scope": "trainer",
            "identifier": "grpo_trainer",
            "operations": [
                _cleanup_operation(
                    1,
                    "release",
                    "failed" if trainer_error is not None else "success",
                    str(trainer_error) if trainer_error is not None else None,
                )
            ],
            "final_state": "residual" if trainer_error is not None else "released",
            "residual": trainer_error is not None,
        }
    )
    if trainer_error is not None:
        errors.append(trainer_error)
    recorder.log("trainer and colocated vLLM references released")
    del trainer
    gc.collect()
    _log_vllm_model_residual(vllm_model_ref, recorder)
    return errors, handles


def _cleanup_operation(
    sequence: int, operation: str, result: str, error: str | None = None
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "at": _utc_now(),
        "operation": operation,
        "result": result,
        "error": error,
    }


def _not_initialized_handle(scope: str, identifier: str) -> dict[str, Any]:
    operation = "shutdown" if scope == "vllm_engine" else "release"
    return {
        "scope": scope,
        "identifier": identifier,
        "operations": [_cleanup_operation(1, operation, "not_initialized")],
        "final_state": "not_initialized",
        "residual": False,
    }


def _snapshot_child_processes() -> dict[int, dict[str, Any]]:
    """只遍历当前主进程的后代，不扫描或治理其他作业。"""

    discovered: dict[int, dict[str, Any]] = {}
    frontier = [os.getpid()]
    while frontier:
        parent = frontier.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            child_pids = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        for pid in child_pids:
            if pid in discovered:
                continue
            identity = _process_identity(pid)
            if identity is None:
                continue
            discovered[pid] = identity
            frontier.append(pid)
    return discovered


def _new_child_processes(
    baseline: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    current = _snapshot_child_processes()
    return {
        pid: identity
        for pid, identity in current.items()
        if pid not in baseline or baseline[pid].get("start_ticks") != identity.get("start_ticks")
    }


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        state = fields[0]
        start_ticks = int(fields[19])
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None
    command = raw_command.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "state": state,
        "command": command,
        "scope": "vllm_worker" if "vllm" in command.lower() else "trainer_worker",
    }


def _same_process(identity: dict[str, Any]) -> bool:
    current = _process_identity(int(identity["pid"]))
    return (
        current is not None
        and current["start_ticks"] == identity["start_ticks"]
        and current["state"] != "Z"
    )


def _wait_for_process_exit(identity: dict[str, Any], timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not _same_process(identity):
            return True
        time.sleep(0.1)
    return not _same_process(identity)


def _finalize_run_processes(
    processes: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[BaseException]]:
    records: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    for pid, identity in sorted(processes.items()):
        operations: list[dict[str, Any]] = []
        if not _same_process(identity):
            operations.append(_cleanup_operation(1, "verify_exit", "not_found"))
            final_state = "not_found"
            residual = False
        else:
            sequence = 1
            try:
                os.kill(pid, signal.SIGTERM)
                operations.append(_cleanup_operation(sequence, "terminate", "success"))
            except ProcessLookupError:
                operations.append(_cleanup_operation(sequence, "terminate", "not_found"))
            except OSError as exc:
                operations.append(
                    _cleanup_operation(sequence, "terminate", "failed", str(exc))
                )
            sequence += 1
            exited = _wait_for_process_exit(identity)
            operations.append(
                _cleanup_operation(
                    sequence,
                    "join",
                    "success" if exited else "failed",
                    None if exited else "process did not exit after SIGTERM",
                )
            )
            if not exited and _same_process(identity):
                sequence += 1
                try:
                    os.kill(pid, signal.SIGKILL)
                    operations.append(_cleanup_operation(sequence, "terminate", "success"))
                except ProcessLookupError:
                    operations.append(_cleanup_operation(sequence, "terminate", "not_found"))
                except OSError as exc:
                    operations.append(
                        _cleanup_operation(sequence, "terminate", "failed", str(exc))
                    )
                exited = _wait_for_process_exit(identity)
            residual = not exited
            final_state = "residual" if residual else "exited"
        record = {
            "scope": identity["scope"],
            "pid": pid,
            "operations": operations,
            "final_state": final_state,
            "residual": residual,
        }
        records.append(record)
        if residual:
            errors.append(RuntimeError(f"run-owned child process {pid} remains active"))
    return records, errors


def _vllm_model_weakref(llm: Any) -> weakref.ReferenceType[Any] | None:
    core_client = llm.llm_engine.engine_core
    engine_core = getattr(core_client, "engine_core", None)
    executor = getattr(engine_core, "model_executor", None)
    driver_wrapper = getattr(executor, "driver_worker", None)
    worker = getattr(driver_wrapper, "worker", None)
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    return weakref.ref(model) if model is not None else None


def _log_vllm_model_residual(
    model_ref: weakref.ReferenceType[Any] | None, recorder: Any
) -> None:
    model = model_ref() if model_ref is not None else None
    referrer_summary: list[str] = []
    if model is not None:
        for referrer in gc.get_referrers(model):
            if isinstance(referrer, dict):
                keys = [str(key) for key, value in referrer.items() if value is model]
                referrer_summary.append(f"dict_keys={keys[:8]}")
            else:
                referrer_summary.append(type(referrer).__name__)
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.instance
        allocations = list(allocator.pointer_to_data.values()) if allocator else []
        allocation_count = len(allocations)
        allocation_bytes = sum(item.handle[1] for item in allocations)
    except BaseException:
        allocation_count = -1
        allocation_bytes = -1
    recorder.log(
        f"vllm-model-alive={model is not None} referrers={referrer_summary[:12]} "
        f"cumem_allocations={allocation_count} cumem_bytes={allocation_bytes}"
    )
    del model


def _clear_vllm_cuda_graphs(llm: Any) -> None:
    """补足锁定 vLLM 新 GPU runner 的进程内状态清理。"""

    import vllm
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.platforms import current_platform

    if vllm.__version__ != "0.22.1":
        raise RuntimeError(
            "vLLM CUDA Graph teardown is qualified only for vllm==0.22.1, "
            f"got {vllm.__version__}"
        )

    core_client = llm.llm_engine.engine_core
    engine_core = getattr(core_client, "engine_core", None)
    executor = getattr(engine_core, "model_executor", None)
    driver_wrapper = getattr(executor, "driver_worker", None)
    worker = getattr(driver_wrapper, "worker", None)
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("locked vLLM in-process GPU model runner is unavailable")

    manager = getattr(model_runner, "cudagraph_manager", None)
    if manager is not None:
        manager.graphs.clear()
        manager.hidden_states = None
        manager.aux_hidden_states.clear()
        manager.intermediate_tensors = None
        manager.pool = None
        model_runner.cudagraph_manager = None

    # vLLM 0.22.1 的新 GPU runner.shutdown() 只删除 model_runner.model；
    # DefaultModelState 和 LoRA manager 仍会持有同一个模型对象。
    model_state = getattr(model_runner, "model_state", None)
    if model_state is not None:
        model_state.model = None
        encoder_runner = getattr(model_state, "encoder_runner", None)
        if encoder_runner is not None:
            encoder_runner.model = None
        model_runner.model_state = None
    lora_manager = getattr(model_runner, "lora_manager", None)
    if lora_manager is not None:
        adapter_manager = getattr(lora_manager, "_adapter_manager", None)
        if adapter_manager is not None:
            adapter_manager.model = None
        model_runner.lora_manager = None
    model_runner.pooling_runner = None
    model_runner.speculator = None

    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    for wrapper in (
        *list(CUDAGraphWrapper._all_instances),
        *list(BreakableCUDAGraphWrapper._all_instances),
    ):
        wrapper.graph_pool = None
    type(current_platform)._global_graph_pool = None


def _detach_vllm_engine(llm: Any) -> None:
    """在官方 shutdown 后断开锁定 in-process engine 的残留强引用。"""

    llm_engine = llm.llm_engine
    core_client = getattr(llm_engine, "engine_core", None)
    engine_core = getattr(core_client, "engine_core", None)
    if engine_core is not None:
        engine_core.model_executor = None
        engine_core.scheduler = None
        core_client.engine_core = None
    llm_engine.model_executor = None
    llm_engine.engine_core = None


def _trainer_weakrefs(trainer: Any | None) -> dict[str, weakref.ReferenceType[Any]]:
    if trainer is None:
        return {}
    references: dict[str, weakref.ReferenceType[Any]] = {"trainer": weakref.ref(trainer)}
    model = getattr(trainer, "model", None)
    optimizer = getattr(trainer, "optimizer", None)
    if model is not None:
        references["model"] = weakref.ref(model)
    if optimizer is not None:
        references["optimizer"] = weakref.ref(optimizer)
    return references


def _log_live_trainer_cuda_references(
    references: dict[str, weakref.ReferenceType[Any]], recorder: Any
) -> None:
    import torch

    live = [name for name, reference in references.items() if reference() is not None]
    cuda_parameter_bytes = 0
    model_ref = references.get("model")
    model = model_ref() if model_ref is not None else None
    if model is not None:
        cuda_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if parameter.is_cuda
        )
    del model
    recorder.log(
        f"post-teardown live_refs={live} cuda_parameter_bytes={cuda_parameter_bytes} "
        f"allocated={torch.cuda.memory_allocated(0)} reserved={torch.cuda.memory_reserved(0)}"
    )


def _adapter_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    if not state:
        raise RuntimeError("PEFT model produced an empty adapter state")
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_saved_adapter(output_dir: Path, expected: dict[str, Any]) -> bool:
    import torch
    from safetensors.torch import load_file

    config_path = output_dir / "adapter_config.json"
    model_path = output_dir / "adapter_model.safetensors"
    training_args = output_dir / "training_args.bin"
    if not config_path.is_file() or not model_path.is_file() or not training_args.is_file():
        return False
    loaded = load_file(model_path.as_posix(), device="cpu")
    if set(loaded) != set(expected):
        return False
    return all(torch.equal(loaded[name], expected[name]) for name in expected)


def _require_final_model_files(output_dir: Path) -> None:
    missing = [
        name
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_args.bin",
        )
        if not (output_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError("Trainer final model files are missing: " + ", ".join(missing))


def _training_metrics(trainer: Any, train_output: Any) -> dict[str, float | None]:
    history = list(getattr(trainer.state, "log_history", []))

    def latest(key: str) -> float | None:
        for row in reversed(history):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    loss = latest("loss")
    if loss is None:
        value = getattr(train_output, "training_loss", None)
        loss = float(value) if isinstance(value, (int, float)) else None
    return {
        "loss": loss,
        "grad_norm": latest("grad_norm"),
        "frac_reward_zero_std": latest("frac_reward_zero_std"),
        "importance_sampling_ratio_mean": latest(
            "sampling/importance_sampling_ratio/mean"
        ),
        "importance_sampling_ratio_max": latest(
            "sampling/importance_sampling_ratio/max"
        ),
    }


def _is_positive_finite(value: float | None) -> bool | None:
    if value is None:
        return None
    return math.isfinite(value) and value > 0.0


def _native_policy_path_reached(environments: list[Any]) -> bool:
    """从真实 environment 事实判定，不解析 assistant 文本或伪造调用。"""

    for environment in environments:
        trajectory = environment.trajectory
        verification = environment.verification
        patch = environment.frozen_patch
        reward = getattr(environment, "_reward", None)
        if trajectory is None or trajectory.termination != "submitted":
            continue
        tool_names = [step.action.tool_name for step in trajectory.steps]
        if (
            "edit_file" in tool_names
            and "submit" in tool_names
            and isinstance(patch, str)
            and bool(patch.strip())
            and verification is not None
            and reward in (0.0, 1.0)
        ):
            return True
    return False


def _validate_rendered_prompt_length(trainer: Any, prompt: object, limit: int) -> None:
    tools = trainer._env_tools[None]
    encoded = trainer.processing_class.apply_chat_template(
        conversation=[prompt],
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    length = len(input_ids[0])
    if length > limit:
        raise RuntimeError(f"rendered prompt length {length} exceeds configured limit {limit}")


def _failure_category(exc: BaseException, stage: str) -> str:
    from swe_agent.docker import DockerRuntimeError
    from swe_agent.swegym import SWEGymContractError
    from swe_agent.verifier import VerificationInfrastructureError

    if isinstance(exc, RecordingRuntimeError):
        return "recording"
    if isinstance(exc, VerificationInfrastructureError):
        return "verifier"
    if isinstance(exc, DockerRuntimeError):
        return "docker"
    if isinstance(exc, SWEGymContractError):
        return "environment"
    return {
        "gpu_preflight": "dependency",
        "load_task": "environment",
        "tokenizer": "model",
        "trainer_construct": "generation_backend",
        "train": "trainer",
        "save_model": "trainer",
    }.get(stage, "trainer")


def _code_provenance(project_root: Path) -> tuple[str | None, bool]:
    commit = subprocess.run(
        ["git", "-C", project_root.as_posix(), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", project_root.as_posix(), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = commit.stdout.strip() if commit.returncode == 0 else None
    return value, status.returncode != 0 or bool(status.stdout.strip())


def _model_revision(model_path: Path) -> str | None:
    metadata = model_path.resolve() / ".mv"
    try:
        value = metadata.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
