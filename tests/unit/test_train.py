from __future__ import annotations

import os
from pathlib import Path

import pytest

from swe_agent.config import load_config
from swe_agent.train import (
    RuntimeNotQualifiedError,
    _clear_vllm_cuda_graphs,
    _detach_vllm_engine,
    _native_policy_path_reached,
    _recording_reward,
    _release_trainer,
    _require_single_visible_gpu,
    _sweep_orphans_at_exit,
    build_grpo_config,
    build_peft_config,
    build_quantization_config,
    preflight,
    run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_7B = PROJECT_ROOT / "configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml"


def test_preflight_is_read_only_and_reports_complete_stage_modules(tmp_path: Path) -> None:
    config, project_root, _ = load_config(CONFIG_7B)
    output_root = Path(config.output.output_root)
    before = sorted(path.name for path in output_root.iterdir()) if output_root.exists() else []

    report = preflight(config, project_root)

    assert report["status"] == "preflight_passed"
    assert report["vllm_tensor_parallel_size"] is None
    assert "models.py" not in report["missing_domain_modules"]
    assert "swegym.py" not in report["missing_domain_modules"]
    assert "prompts.py" not in report["missing_domain_modules"]
    assert "docker.py" not in report["missing_domain_modules"]
    assert "tools.py" not in report["missing_domain_modules"]
    assert "verifier.py" not in report["missing_domain_modules"]
    assert "environment.py" not in report["missing_domain_modules"]
    assert "rewards.py" not in report["missing_domain_modules"]
    assert report["missing_domain_modules"] == []
    after = sorted(path.name for path in output_root.iterdir()) if output_root.exists() else []
    assert after == before
    assert not list(tmp_path.iterdir())


def test_entry_executes_exactly_one_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    devices: list[int] = []
    outcome = {
        "run_id": "run-0",
        "lifecycle": "completed",
        "native_policy_path_reached": False,
        "trainer_group_consumed": True,
        "system_closed_loop": "failed",
        "failure": None,
        "final_model_ref": "adapter_model.safetensors",
        "cleanup": {"state": "completed", "clean_release": True, "residuals": []},
        "interrupted_signum": None,
    }

    monkeypatch.setattr("swe_agent.train._require_single_visible_gpu", lambda: 3)
    monkeypatch.setattr("swe_agent.launcher.split_visible_gpus", lambda config: "0")

    def fake_run_once(*, config, project_root, seed, physical_device, server_gpu):
        del config, project_root
        seeds.append(seed)
        devices.append(physical_device)
        server_gpus.append(server_gpu)
        return outcome

    server_gpus: list[str] = []
    monkeypatch.setattr("swe_agent.train._run_once", fake_run_once)
    result = run(CONFIG_7B)
    assert seeds == [20260714]
    assert devices == [3]
    assert server_gpus == ["0"]
    assert result == outcome


@pytest.mark.parametrize("value, expected", [("1", 1), ("3", 3), ("03", 3)])
def test_single_visible_gpu_is_selected_only_by_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert _require_single_visible_gpu() == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(expected)


@pytest.mark.parametrize("value", [None, "", "-1", "gpu3", "2,3"])
def test_single_visible_gpu_rejects_missing_or_non_single_selection(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(RuntimeNotQualifiedError, match="exactly one"):
        _require_single_visible_gpu()


def test_public_peft_and_grpo_configs_construct_without_gpu(tmp_path: Path) -> None:
    config, _, _ = load_config(CONFIG_7B)
    peft_config = build_peft_config(config)
    grpo_config = build_grpo_config(
        config,
        tmp_path / "output",
        seed=config.runtime.base_seed,
        use_cpu=True,
    )

    assert peft_config.r == config.peft.rank
    assert peft_config.lora_alpha == config.peft.alpha
    assert set(peft_config.target_modules) == set(config.peft.target_modules)
    assert build_quantization_config(config) is None
    assert grpo_config.num_generations == config.grpo.num_generations
    assert grpo_config.generation_batch_size == config.grpo.generation_batch_size
    assert grpo_config.steps_per_generation == config.grpo.gradient_accumulation_steps
    assert grpo_config.model_init_kwargs == {"dtype": "bfloat16"}
    assert grpo_config.vllm_mode == "server"
    assert grpo_config.vllm_server_base_url == "http://127.0.0.1:8000"
    assert grpo_config.vllm_model_impl == "vllm"
    assert grpo_config.vllm_max_model_length == 32768
    assert grpo_config.vllm_enable_sleep_mode is False
    assert grpo_config.max_tool_calling_iterations == config.generation.max_tool_calling_iterations
    assert grpo_config.loss_type == config.grpo.loss_type
    assert grpo_config.beta == config.grpo.beta
    assert (
        grpo_config.vllm_importance_sampling_correction
        == config.grpo.vllm_importance_sampling_correction
    )
    assert grpo_config.router_aux_loss_coef == 0.0
    assert grpo_config.shuffle_dataset is True


def test_recording_reward_preserves_trl_position_order_and_drains_events() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.rollouts = []
            self.begun = []
            self.group = None
            self.events = []

        def begin_group(self, prompt, rollout_count):
            self.begun.append((prompt, rollout_count))

        def write_rollout(self, index, **values):
            self.rollouts.append((index, values))

        def complete_group(self, **values):
            self.group = values

        def merge_cleanup_events(self, events):
            self.events.extend(events)

    class FakeEnvironment:
        def __init__(self, index: int) -> None:
            self.episode_id = f"episode-{index}"
            self.trajectory = f"trajectory-{index}"
            self.frozen_patch = None
            self.verification = None
            self.index = index

        def _drain_events(self):
            return [{"index": self.index}]

    recorder = FakeRecorder()
    environments = [FakeEnvironment(index) for index in range(4)]
    prompts = [[{"role": "user", "content": str(index)}] for index in range(4)]
    completions = [[{"role": "assistant", "content": str(index)}] for index in range(4)]

    def adapter(*, completions, environments, **kwargs):
        del completions, kwargs
        return [float(environment.index == 0) for environment in environments]

    reward = _recording_reward(recorder, adapter)
    assert reward(
        prompts=prompts,
        completions=completions,
        environments=environments,
    ) == [1.0, 0.0, 0.0, 0.0]
    assert [index for index, _ in recorder.rollouts] == [0, 1, 2, 3]
    assert recorder.rollouts[2][1]["messages"] == prompts[2] + completions[2]
    assert recorder.group["episode_ids"] == [f"episode-{index}" for index in range(4)]
    assert recorder.events == [{"index": index} for index in range(4)]
    # 多步训练：reward 可被多次调用，每次开始一个新 group
    reward(prompts=prompts, completions=completions, environments=environments)
    assert recorder.begun == [(prompts[0], 4), (prompts[0], 4)]


def test_native_policy_path_requires_executed_edit_submit_patch_verifier_and_reward() -> None:
    step = lambda name: type("Step", (), {"action": type("Action", (), {"tool_name": name})()})()
    trajectory = type(
        "Trajectory", (), {"termination": "submitted", "steps": [step("edit_file"), step("submit")]}
    )()
    environment = type(
        "Environment",
        (),
        {
            "trajectory": trajectory,
            "verification": object(),
            "frozen_patch": "diff --git a/x b/x\n",
            "_reward": 0.0,
        },
    )()
    assert _native_policy_path_reached([environment])
    environment.verification = None
    assert not _native_policy_path_reached([environment])


def test_trainer_release_shuts_down_inprocess_vllm_and_moves_model_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr("swe_agent.train._clear_vllm_cuda_graphs", lambda llm: None)

    class EngineCore:
        def shutdown(self):
            events.append("shutdown")

    class LLM:
        llm_engine = type("Engine", (), {"engine_core": EngineCore()})()

        def sleep(self, level):
            events.append(("sleep", level))

        def wake_up(self):
            events.append("wake_up")

    class Model:
        def to(self, device):
            events.append(("model", device))

        def parameters(self):
            return []

    class Optimizer:
        def __init__(self):
            self.state = {"x": 1}
            self.param_groups = [{"params": []}]

        def zero_grad(self, set_to_none):
            events.append(("zero_grad", set_to_none))

    class Accelerator:
        def free_memory(self, *objects):
            events.append("free_memory")
            return tuple(None for _ in objects)

    class Recorder:
        def log(self, message):
            events.append(("log", message))

    trainer = type(
        "Trainer",
        (),
        {
            "vllm_generation": type(
                "Backend", (), {"llm": LLM(), "enable_sleep_mode": True}
            )(),
            "model": Model(),
            "optimizer": Optimizer(),
            "lr_scheduler": object(),
            "model_wrapped": object(),
            "ref_model": None,
            "_buffered_inputs": [],
            "accelerator": Accelerator(),
        },
    )()
    errors, handles = _release_trainer(trainer, Recorder())
    assert errors == []
    assert not any(handle["residual"] for handle in handles)
    assert events[:5] == [
        ("sleep", 2),
        "shutdown",
        ("zero_grad", True),
        ("model", "cpu"),
        "free_memory",
    ]


def test_locked_vllm_cuda_graph_cleanup_releases_manager_state() -> None:
    import vllm
    from vllm.platforms import current_platform

    class Manager:
        graphs = {"graph": object()}
        hidden_states = object()
        aux_hidden_states = [object()]
        intermediate_tensors = object()
        pool = object()

    manager = Manager()
    model_state = type("ModelState", (), {"model": object()})()
    adapter_manager = type("AdapterManager", (), {"model": object()})()
    lora_manager = type("LoraManager", (), {"_adapter_manager": adapter_manager})()
    model_runner = type(
        "ModelRunner",
        (),
        {
            "cudagraph_manager": manager,
            "model_state": model_state,
            "lora_manager": lora_manager,
            "pooling_runner": object(),
            "speculator": object(),
        },
    )()
    worker = type("Worker", (), {"model_runner": model_runner})()
    driver_worker = type("DriverWorker", (), {"worker": worker})()
    executor = type("Executor", (), {"driver_worker": driver_worker})()
    engine_core = type("EngineCore", (), {"model_executor": executor})()
    core_client = type("CoreClient", (), {"engine_core": engine_core})()
    llm = type(
        "LLM",
        (),
        {"llm_engine": type("Engine", (), {"engine_core": core_client})()},
    )()

    assert vllm.__version__ == "0.22.1"
    type(current_platform)._global_graph_pool = object()
    _clear_vllm_cuda_graphs(llm)

    assert manager.graphs == {}
    assert manager.hidden_states is None
    assert manager.aux_hidden_states == []
    assert manager.intermediate_tensors is None
    assert manager.pool is None
    assert model_runner.cudagraph_manager is None
    assert model_state.model is None
    assert adapter_manager.model is None
    assert model_runner.model_state is None
    assert model_runner.lora_manager is None
    assert model_runner.pooling_runner is None
    assert model_runner.speculator is None
    assert type(current_platform)._global_graph_pool is None


def test_locked_vllm_engine_detach_removes_executor_object_chain() -> None:
    executor = object()
    engine_core = type(
        "EngineCore", (), {"model_executor": executor, "scheduler": object()}
    )()
    core_client = type("CoreClient", (), {"engine_core": engine_core})()
    llm_engine = type(
        "LLMEngine",
        (), {"engine_core": core_client, "model_executor": executor},
    )()
    llm = type("LLM", (), {"llm_engine": llm_engine})()

    _detach_vllm_engine(llm)

    assert engine_core.model_executor is None
    assert engine_core.scheduler is None
    assert core_client.engine_core is None
    assert llm_engine.model_executor is None
    assert llm_engine.engine_core is None


def test_atexit_sweep_removes_orphans_and_stays_quiet(capsys: pytest.CaptureFixture) -> None:
    from swe_agent.docker import CommandResult

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, argv, **kwargs):
            argv = list(argv)
            self.calls.append(argv)
            stdout = "aa\n" if argv[1] == "ps" else ""
            return CommandResult(
                argv=argv, exit_code=0, stdout=stdout, stderr="", duration_sec=0.01
            )

    client = FakeClient()
    _sweep_orphans_at_exit(client, "run-x")

    assert client.calls[0][:3] == ["docker", "ps", "-aq"]
    assert "label=swe_agent.run_id=run-x" in client.calls[0]
    assert client.calls[1] == ["docker", "rm", "-f", "aa"]
    assert "swept orphan containers: aa" in capsys.readouterr().err


def test_atexit_sweep_swallows_failures(capsys: pytest.CaptureFixture) -> None:
    class BadClient:
        def run(self, argv, **kwargs):
            raise RuntimeError("daemon down")

    _sweep_orphans_at_exit(BadClient(), "run-x")  # 必须不抛出

    assert "orphan sweep failed" in capsys.readouterr().err
