from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from siete_rl.config import ProjectConfig, load_config
from siete_rl.train import (
    RecordingRuntimeError,
    RuntimeNotQualifiedError,
    _apply_liger_runtime_flags,
    _close_environments,
    _close_vllm_communicator,
    _detach_vllm_client_atexit,
    _recording_reward,
    _require_trainer_visible_gpus,
    build_grpo_config,
    preflight,
    run,
)
from siete_rl.launcher import RunEndpoints
from siete_rl.models import Settlement, Trajectory
from siete_rl.recording import RunRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_7B = PROJECT_ROOT / "configs/stage1.yaml"


def _recording_config(tmp_path: Path) -> ProjectConfig:
    config, _, _ = load_config(CONFIG_7B)
    output = config.output.model_copy(
        update={"output_root": (tmp_path / "outputs").as_posix(), "run_id": None}
    )
    return config.model_copy(update={"output": output})


def _trajectory_for(task_id: str) -> Trajectory:
    return Trajectory.model_validate(
        {
            "task_id": task_id,
            "environment_id": task_id,
            "steps": [],
            "termination": "format_exhausted",
            "settlement": {"status": "empty_patch"},
        }
    )


class _RecordingEnvironment:
    def __init__(self, index: int) -> None:
        self.episode_id = f"episode-{index}"
        self.trajectory = None
        self.frozen_patch = None
        self.verification = None

    def _drain_events(self):
        return []


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _complete_recorded_group(
    recorder: RunRecorder, task_id: str, *, count: int = 2
) -> None:
    environments = [_RecordingEnvironment(index) for index in range(count)]
    prompts = [[{"role": "user", "content": f"prompt-{index}"}] for index in range(count)]
    completions = [
        [{"role": "assistant", "content": f"completion-{index}"}]
        for index in range(count)
    ]

    def adapter(*, environments, **kwargs):
        del kwargs
        for environment in environments:
            environment.trajectory = _trajectory_for(task_id)
        return [0.0] * len(environments)

    reward = _recording_reward(recorder, adapter)
    assert reward(
        prompts=prompts,
        completions=completions,
        environments=environments,
        task_id=[task_id] * count,
    ) == [0.0] * count


def test_preflight_reports_complete_modules_without_creating_run_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = tmp_path / "fake-model"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}', encoding="utf-8")
    (fake_model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(fake_model))
    monkeypatch.setenv("TOKENIZER_PATH", str(fake_model))

    config, project_root, _ = load_config(CONFIG_7B)
    output_root = tmp_path / "outputs"
    config = config.model_copy(
        update={
            "output": config.output.model_copy(
                update={"output_root": str(output_root)}
            )
        }
    )

    report = preflight(config, project_root)

    assert report["status"] == "preflight_passed"
    assert report["vllm_tensor_parallel_size"] == config.vllm.tensor_parallel_size
    assert report["missing_domain_modules"] == []
    assert not output_root.exists()


def test_entry_delegates_to_the_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = {
        "run_id": "run-0",
        "status": "completed",
        "failure": None,
        "artifacts": {"final_model": "adapter_model.safetensors"},
        "cleanup": {
            "status": "completed",
            "clean_release": True,
            "residual_count": 0,
        },
        "interrupted_signum": None,
    }

    monkeypatch.setattr("siete_rl.supervisor.run", lambda path: outcome)
    result = run(CONFIG_7B)
    assert result == outcome


@pytest.mark.parametrize("local_rank, expected", [("0", 2), ("1", 3)])
def test_trainer_visible_gpu_is_selected_by_local_rank(
    monkeypatch: pytest.MonkeyPatch, local_rank: str, expected: int
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    selected = []
    monkeypatch.setattr("torch.cuda.set_device", selected.append)
    assert _require_trainer_visible_gpus() == expected
    assert selected == [int(local_rank)]


@pytest.mark.parametrize("value", [None, "", "2", "2,3,4"])
def test_trainer_visible_gpu_rejects_non_pair_selection(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeNotQualifiedError, match="exactly two"):
        _require_trainer_visible_gpus()


def test_liger_runtime_flags_disable_dynamo_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = load_config(CONFIG_7B)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)

    _apply_liger_runtime_flags(config)

    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert (
        logging.getLogger("transformers.configuration_utils").level == logging.ERROR
    )


def test_grpo_config_uses_run_private_vllm_endpoints(tmp_path: Path) -> None:
    config, _, _ = load_config(CONFIG_7B)
    endpoints = RunEndpoints(host="127.0.0.1", server_port=18421, group_port=18422, ddp_port=18423)
    grpo_config = build_grpo_config(
        config,
        tmp_path / "output",
        seed=config.runtime.base_seed,
        use_cpu=True,
        vllm_endpoints=endpoints,
    )
    assert grpo_config.vllm_server_base_url == endpoints.base_url
    assert grpo_config.vllm_server_port == 18421
    assert grpo_config.vllm_group_port == 18422


def test_grpo_config_maps_epochs_and_prompt_batch_to_trl(tmp_path: Path) -> None:
    config, _, _ = load_config(CONFIG_7B)
    config = config.model_copy(
        update={
            "grpo": config.grpo.model_copy(
                update={
                    "num_train_epochs": 4,
                    "train_batch_size": 2,
                    "num_generations": 8,
                }
            )
        }
    )

    grpo_config = build_grpo_config(
        config,
        tmp_path / "output",
        seed=config.runtime.base_seed,
        use_cpu=True,
    )

    assert grpo_config.num_train_epochs == 4
    assert grpo_config.max_steps == -1
    assert grpo_config.generation_batch_size == 16
    assert grpo_config.gradient_accumulation_steps == 8


def test_vllm_client_cleanup_is_explicit_and_not_atexit(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Client:
        def close_communicator(self) -> None:
            events.append("close")

    client = Client()
    trainer = type(
        "Trainer", (), {"vllm_generation": type("Generation", (), {"vllm_client": client})()}
    )()
    recorder = type("Recorder", (), {"log": lambda self, message: events.append(message)})()
    monkeypatch.setattr("siete_rl.train.atexit.unregister", lambda callback: events.append("unregister"))

    detached = _detach_vllm_client_atexit(trainer, recorder)
    handle = _close_vllm_communicator(detached, recorder)

    assert events[0] == "unregister"
    assert "close" in events
    assert handle["final_state"] == "closed"


def test_recording_reward_rejects_mixed_callback_tasks_before_adapter() -> None:
    calls = 0

    class FakeRecorder:
        def __init__(self) -> None:
            self.begun = []

        def begin_group(self, *args, **kwargs):
            self.begun.append((args, kwargs))

        def merge_cleanup_events(self, events):
            del events

    def adapter(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return [0.0, 0.0]

    recorder = FakeRecorder()
    reward = _recording_reward(recorder, adapter)
    with pytest.raises(RecordingRuntimeError, match="task_id"):
        reward(
            prompts=[[], []],
            completions=[[], []],
            environments=[_RecordingEnvironment(0), _RecordingEnvironment(1)],
            task_id=["task-a", "task-b"],
        )

    assert calls == 0
    assert recorder.begun == []


def test_adapter_failure_isolated_from_completed_group(
    tmp_path: Path,
) -> None:
    task_a = "getmoto__moto-7023"
    task_b = "getmoto__moto-7212"
    recorder = RunRecorder(config=_recording_config(tmp_path), seed=7)
    _complete_recorded_group(recorder, task_a)
    group_a_dir = recorder.output_dir / "rollouts/batch-0000/group-0000"
    group_a_bytes = _tree_bytes(group_a_dir)
    metrics_before = json.loads(json.dumps(recorder.run["results"]["reward"]))

    environments = [_RecordingEnvironment(0), _RecordingEnvironment(1)]

    def failing_adapter(*, environments, **kwargs):
        del kwargs
        environments[0].trajectory = _trajectory_for(task_b)
        raise RuntimeError("adapter B failed")

    reward = _recording_reward(recorder, failing_adapter)
    with pytest.raises(RuntimeError, match="adapter B failed"):
        reward(
            prompts=[[{"role": "user", "content": "B"}]] * 2,
            completions=[[{"role": "assistant", "content": "partial"}]] * 2,
            environments=environments,
            task_id=[task_b, task_b],
        )

    assert _tree_bytes(group_a_dir) == group_a_bytes
    assert not (recorder.output_dir / "rollouts/batch-0001").exists()
    assert recorder.run["train"]["groups_generated"] == 1
    assert recorder.run["train"]["rollouts_generated"] == 2
    assert recorder.run["results"]["reward"] == metrics_before


def test_recording_reward_preserves_trl_position_order_and_drains_events() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.rollouts = []
            self.begun = []
            self.group = None
            self.events = []
            self.native_policy_path_reached = False

        def begin_group(self, prompt, rollout_count, *, task_id):
            self.begun.append((prompt, rollout_count, task_id))

        def write_rollout(self, index, **values):
            self.rollouts.append((index, values))

        def complete_group(self, **values):
            self.group = values

        def observe_native_policy_path(self, reached):
            self.native_policy_path_reached = self.native_policy_path_reached or reached

        def merge_cleanup_events(self, events):
            self.events.extend(events)

    class FakeEnvironment:
        def __init__(self, index: int) -> None:
            self.episode_id = f"episode-{index}"
            self.trajectory = None
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
        for environment in environments:
            environment.trajectory = type(
                "Trajectory",
                (),
                {
                    "task_id": "getmoto__moto-7023",
                    "termination": "format_exhausted",
                    "steps": [],
                    "settlement": Settlement(status="unresolved"),
                },
            )()
        return [float(environment.index == 0) for environment in environments]

    reward = _recording_reward(recorder, adapter)
    assert reward(
        prompts=prompts,
        completions=completions,
        environments=environments,
        task_id=["getmoto__moto-7023"] * 4,
    ) == [1.0, 0.0, 0.0, 0.0]
    assert [index for index, _ in recorder.rollouts] == [0, 1, 2, 3]
    assert recorder.rollouts[2][1]["messages"] == prompts[2] + completions[2]
    assert recorder.group["episode_ids"] == [f"episode-{index}" for index in range(4)]
    assert recorder.events == [{"index": index} for index in range(4)]
    assert recorder.native_policy_path_reached is False
    # 多步训练：reward 可被多次调用，每次开始一个新 group
    reward(
        prompts=prompts,
        completions=completions,
        environments=environments,
        task_id=["getmoto__moto-7023"] * 4,
    )
    assert recorder.begun == [
        (prompts[0], 4, "getmoto__moto-7023"),
        (prompts[0], 4, "getmoto__moto-7023"),
    ]


def test_recording_reward_rejects_a_group_with_multiple_finalized_tasks() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.begun = []

        def begin_group(self, *args, **kwargs):
            self.begun.append((args, kwargs))

        def write_rollout(self, *args, **kwargs):
            del args, kwargs

        def merge_cleanup_events(self, events):
            del events

    class FakeEnvironment:
        def __init__(self, task_id: str) -> None:
            self.trajectory = type(
                "Trajectory",
                (),
                {
                    "task_id": task_id,
                    "termination": "format_exhausted",
                    "settlement": Settlement(status="empty_patch"),
                },
            )()
            self.frozen_patch = None
            self.verification = None

        def _drain_events(self):
            return []

    recorder = FakeRecorder()
    reward = _recording_reward(recorder, lambda **kwargs: [0.0, 0.0])
    with pytest.raises(RecordingRuntimeError, match="group mixes tasks"):
        reward(
            prompts=[[], []],
            completions=[[], []],
            environments=[
                FakeEnvironment("getmoto__moto-7023"),
                FakeEnvironment("getmoto__moto-7212"),
            ],
            task_id=["getmoto__moto-7023"] * 2,
        )
    assert recorder.begun == []


class _InfraRecorder:
    def __init__(self) -> None:
        self.completed = []

    def begin_group(self, *args, **kwargs):
        del args, kwargs

    def write_rollout(self, *args, **kwargs):
        del args, kwargs

    def complete_group(self, *args, **kwargs):
        self.completed.append((args, kwargs))

    def observe_native_policy_path(self, reached):
        del reached

    def merge_cleanup_events(self, events):
        del events


class _InfraEnvironment:
    def __init__(self, termination: str) -> None:
        self.episode_id = "episode"
        self.trajectory = type(
            "Trajectory",
            (),
            {
                "task_id": "getmoto__moto-7023",
                "termination": termination,
                "steps": [],
                "settlement": Settlement(
                    status="infra_error" if termination == "infra_error" else "unresolved"
                ),
            },
        )()
        self.frozen_patch = None
        self.verification = None

    def _drain_events(self):
        return []


def test_recording_reward_never_aborts_fully_censored_group() -> None:
    environments = [_InfraEnvironment("infra_error") for _ in range(4)]
    recorder = _InfraRecorder()
    reward = _recording_reward(recorder, lambda **kwargs: [None] * 4)

    assert reward(
        prompts=[[]] * 4,
        completions=[[]] * 4,
        environments=environments,
        task_id=["getmoto__moto-7023"] * 4,
    ) == [None] * 4
    assert len(recorder.completed) == 1
    assert recorder.completed[0][1]["settlements"] == [
        Settlement(status="infra_error")
    ] * 4


def test_recording_reward_tolerates_scattered_infra_errors() -> None:
    """零星 infra_error 被 censor，不进入健康 rollout 的组均值。"""
    environments = [_InfraEnvironment("infra_error")] + [
        _InfraEnvironment("submitted") for _ in range(3)
    ]
    reward = _recording_reward(_InfraRecorder(), lambda **kwargs: [None, 1.0, 1.0, 1.0])
    assert reward(
        prompts=[[]] * 4,
        completions=[[]] * 4,
        environments=environments,
        task_id=["getmoto__moto-7023"] * 4,
    ) == [
        None,
        1.0,
        1.0,
        1.0,
    ]


def test_reset_executor_shutdown_failure_is_a_cleanup_error() -> None:
    class FakeRecorder:
        def merge_cleanup_events(self, values):
            del values

        def log(self, message):
            self.message = message

    class FakeExecutor:
        def shutdown(self, **kwargs):
            del kwargs
            raise RuntimeError("shutdown failed")

    recorder = FakeRecorder()
    errors, handles = _close_environments(
        [], recorder, reset_executor=FakeExecutor()
    )

    assert handles == []
    assert len(errors) == 1 and str(errors[0]) == "shutdown failed"
    assert "shutdown failed" in recorder.message
