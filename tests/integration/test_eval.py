from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import siete_rl.eval as eval_module
from siete_rl.eval import (
    EvalDockerSandbox,
    EvalError,
    EvalOutcome,
    EvalProtocol,
    build_comparison,
    build_predictions,
    configure_sampler_backend,
    load_eval_run,
    official_image_name,
    parse_strict_bool,
    public_sample_from_row,
    resolve_harness,
    run_agent_loop,
)
from siete_rl.docker import CommandResult
from siete_rl.models import TerminalEvent


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def result(stdout: str = "", *, exit_code: int = 0) -> CommandResult:
    return CommandResult(
        argv=[], exit_code=exit_code, stdout=stdout, stderr="", duration_sec=0.0
    )


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def run(self, argv, *, input_text=None, timeout_sec):
        del input_text, timeout_sec
        self.commands.append(list(argv))
        return self.responses.pop(0)


class ImmediateFuture:
    def __init__(self, function, args, kwargs) -> None:
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._resolved = False
        self._value = None

    def result(self):
        if not self._resolved:
            self._value = self._function(*self._args, **self._kwargs)
            self._resolved = True
        return self._value


class RecordingExecutor:
    instances: list["RecordingExecutor"] = []

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers
        self.futures: list[ImmediateFuture] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.instances.append(self)

    def submit(self, function, *args, **kwargs):
        future = ImmediateFuture(function, args, kwargs)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def use_recording_executor(monkeypatch: pytest.MonkeyPatch) -> list[RecordingExecutor]:
    RecordingExecutor.instances = []
    monkeypatch.setattr(eval_module, "ThreadPoolExecutor", RecordingExecutor)
    return RecordingExecutor.instances


def protocol() -> EvalProtocol:
    return EvalProtocol(
        max_prompt_length=8,
        max_completion_length=24,
        max_model_length=32,
        max_tool_calling_iterations=4,
        max_consecutive_protocol_errors=2,
        max_observation_chars=100,
        exec_timeout_sec=30,
        grader_timeout_sec=60,
        cpus=1.0,
        memory="1g",
        pids_limit=64,
        seed=1,
        repetition_penalty=1.0,
        gpu_memory_utilization=0.5,
    )


def make_eval_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: tuple[str, str] | None = None,
) -> Path:
    monkeypatch.setattr(eval_module, "PROJECT_ROOT", tmp_path)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    run_root = tmp_path / "outputs" / "run-under-test"
    run_root.mkdir(parents=True)
    (run_root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model), "r": 8}),
        encoding="utf-8",
    )
    (run_root / "adapter_model.safetensors").write_bytes(b"")
    config = {
        "model": {"model_path": str(base_model)},
        "chat": {"max_prompt_length": 8, "max_observation_chars": 100},
        "generation": {
            "max_completion_length": 24,
            "max_tool_calling_iterations": 4,
            "max_consecutive_protocol_errors": 2,
            "repetition_penalty": 1.0,
        },
        "vllm": {"max_model_length": 32, "gpu_memory_utilization": 0.5},
        "docker": {
            "exec_timeout_sec": 30,
            "verifier_timeout_sec": 60,
            "cpus": 1.0,
            "memory": "1g",
            "pids_limit": 64,
        },
        "runtime": {"base_seed": 1},
    }
    if missing is not None:
        section, field = missing
        del config[section][field]
    (run_root / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return run_root


@pytest.mark.parametrize("value,expected", [(None, False), ("False", False), ("True", True)])
def test_eval_base_is_strict(value, expected) -> None:
    assert parse_strict_bool(value) is expected


@pytest.mark.parametrize("value", ["true", "false", "1", "", "yes"])
def test_eval_base_rejects_other_spellings(value) -> None:
    with pytest.raises(EvalError, match="exactly"):
        parse_strict_bool(value)


def test_eval_worker_count_defaults_and_accepts_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_ROLLOUT_WORKERS", raising=False)
    assert eval_module._positive_env_int("EVAL_ROLLOUT_WORKERS") == 1
    monkeypatch.setenv("EVAL_ROLLOUT_WORKERS", "4")
    assert eval_module._positive_env_int("EVAL_ROLLOUT_WORKERS") == 4


@pytest.mark.parametrize(
    ("rollout_env", "harness_env", "expected_rollout", "expected_harness"),
    [
        (None, None, "16", "4"),
        ("8", "2", "8", "2"),
    ],
    ids=["defaults", "explicit-overrides"],
)
def test_eval_script_supplies_defaults_and_preserves_overrides(
    tmp_path: Path,
    rollout_env: str | None,
    harness_env: str | None,
    expected_rollout: str,
    expected_harness: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$EVAL_ROLLOUT_WORKERS\" \"$EVAL_HARNESS_WORKERS\" \"$*\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    if rollout_env is None:
        env.pop("EVAL_ROLLOUT_WORKERS", None)
    else:
        env["EVAL_ROLLOUT_WORKERS"] = rollout_env
    if harness_env is None:
        env.pop("EVAL_HARNESS_WORKERS", None)
    else:
        env["EVAL_HARNESS_WORKERS"] = harness_env

    completed = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/eval.sh"), "outputs/example"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        expected_rollout,
        expected_harness,
        "run --no-sync python -m siete_rl.eval outputs/example",
    ]


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "four"])
def test_eval_worker_count_rejects_non_positive_integer(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EVAL_ROLLOUT_WORKERS", value)
    with pytest.raises(EvalError, match="positive integer"):
        eval_module._positive_env_int("EVAL_ROLLOUT_WORKERS")


def test_sampler_fallback_is_explicit_and_session_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_SAMPLER_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert configure_sampler_backend() == "pytorch"
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    monkeypatch.setenv("EVAL_SAMPLER_BACKEND", "flashinfer")
    assert configure_sampler_backend() == "flashinfer"
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    monkeypatch.setenv("EVAL_SAMPLER_BACKEND", "unknown")
    with pytest.raises(EvalError, match="EVAL_SAMPLER_BACKEND"):
        configure_sampler_backend()


def test_load_eval_run_reads_protocol_from_run_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = load_eval_run(make_eval_run(tmp_path, monkeypatch))

    assert run.protocol == protocol()
    assert run.rank == 8
    assert run.base_model_path == tmp_path / "base-model"


def test_harness_resolver_fails_closed_without_eval_harness_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_HARNESS_PYTHON", raising=False)
    with pytest.raises(EvalError, match="EVAL_HARNESS_PYTHON"):
        resolve_harness()


def test_official_harness_receives_configured_worker_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def run(command, **kwargs):
        del kwargs
        captured.extend(command)
        (tmp_path / "result.eval-workers.json").write_text("{}\n", encoding="utf-8")
        return eval_module.subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(eval_module.subprocess, "run", run)
    report = eval_module.run_official_harness(
        output_dir=tmp_path,
        predictions_path=tmp_path / "predictions.jsonl",
        task_ids=["owner__repo-1"],
        run_id="eval-workers",
        timeout_sec=60,
        max_workers=4,
        harness_root=tmp_path,
        harness_python=tmp_path / "python",
    )
    worker_flag = captured.index("--max_workers")
    assert captured[worker_flag : worker_flag + 2] == ["--max_workers", "4"]
    assert report == {}


def test_execute_propagates_worker_settings_to_metadata_and_both_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    current_protocol = protocol()
    run = SimpleNamespace(
        root=run_root,
        protocol=current_protocol,
        base_model_path=tmp_path / "model",
        adapter_path=run_root,
        rank=8,
    )
    variant_workers: list[tuple[str, int]] = []
    harness_workers: list[int] = []

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2]

    class Generator:
        def __init__(self, **_kwargs):
            pass

        def generate(self, *_args):
            return [[7]], None

    class Server:
        def __init__(self, *_args, **_kwargs):
            self.pid = None

        def start(self):
            self.pid = 123

        def close(self):
            self.pid = None
            return {"status": "terminated"}

    def evaluate(**kwargs):
        variant_workers.append((kwargs["name"], kwargs["rollout_workers"]))
        return [EvalOutcome("task-0", "patch", "submitted", None, [], 0.0)]

    def run_harness(**kwargs):
        harness_workers.append(kwargs["max_workers"])
        if kwargs["predictions_path"] == "gold":
            return {"resolved_instances": 1}
        return {
            "total_instances": 1,
            "completed_instances": 1,
            "resolved_instances": 0,
            "empty_patch_instances": 0,
            "error_instances": 0,
        }

    monkeypatch.setenv("EVAL_BASE", "True")
    monkeypatch.setenv("EVAL_ROLLOUT_WORKERS", "5")
    monkeypatch.setenv("EVAL_HARNESS_WORKERS", "3")
    monkeypatch.delenv("EVAL_TASK_IDS", raising=False)
    monkeypatch.setattr(eval_module, "load_eval_run", lambda *_args: run)
    monkeypatch.setattr(
        eval_module,
        "load_verified_rows",
        lambda *_args: [{"instance_id": "task-0"}],
    )
    monkeypatch.setattr(
        eval_module,
        "resolve_harness",
        lambda: (tmp_path, tmp_path / "python"),
    )
    monkeypatch.setattr(eval_module, "preflight", lambda **_kwargs: {})
    monkeypatch.setattr(eval_module, "SubprocessDockerClient", lambda: object())
    monkeypatch.setattr(eval_module.atexit, "register", lambda *_args: None)
    monkeypatch.setattr(eval_module, "sweep_run_containers", lambda *_args: [])
    monkeypatch.setattr(eval_module, "run_official_harness", run_harness)
    monkeypatch.setattr(
        eval_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Tokenizer()),
    )
    monkeypatch.setattr(
        eval_module,
        "install_openhands_tool_protocol",
        lambda tokenizer: tokenizer,
    )
    monkeypatch.setattr(
        eval_module,
        "public_sample_from_row",
        lambda *_args: SimpleNamespace(task=object()),
    )
    monkeypatch.setattr(eval_module, "build_prompt", lambda *_args: [])
    monkeypatch.setattr(eval_module, "run_adapter_reference", lambda *_args: [7])
    monkeypatch.setattr(eval_module, "HTTPTokenGenerator", Generator)
    monkeypatch.setattr(eval_module, "build_vllm_command", lambda **_kwargs: [])
    monkeypatch.setattr(eval_module, "VLLMServer", Server)
    monkeypatch.setattr(eval_module, "_free_port", lambda: 12345)
    monkeypatch.setattr(eval_module, "evaluate_variant", evaluate)
    monkeypatch.setattr(eval_module, "_git_metadata", lambda *_args: {})

    output_root = eval_module.execute(run_root)
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["protocol"]["request_concurrency"] == 5
    assert metadata["protocol"]["official_harness_workers"] == 3
    assert variant_workers == [("base", 5), ("candidate", 5)]
    assert harness_workers == [3, 3, 3]


def test_harness_resolver_accepts_configured_clean_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "swe-bench"
    (root / "swebench/harness/constants/fixtures").mkdir(parents=True)
    (root / "swebench/harness/run_evaluation.py").write_text("", encoding="utf-8")
    (root / "swebench/harness/constants/fixtures/tokio-rs__tokio-6724.Cargo.lock").write_text(
        "", encoding="utf-8"
    )
    python = tmp_path / "harness-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("EVAL_HARNESS_ROOT", str(root))
    monkeypatch.setenv("EVAL_HARNESS_PYTHON", str(python))
    monkeypatch.setattr(
        eval_module,
        "_run_checked",
        lambda command, **_kwargs: eval_module.HARNESS_REVISION
        if "rev-parse" in command
        else "",
    )

    root, python = resolve_harness()

    assert root == tmp_path / "swe-bench"
    assert python == tmp_path / "harness-venv/bin/python"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("chat", "max_prompt_length"),
        ("generation", "max_completion_length"),
        ("generation", "max_tool_calling_iterations"),
        ("vllm", "max_model_length"),
    ],
)
def test_load_eval_run_rejects_missing_generation_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, section: str, field: str
) -> None:
    run_root = make_eval_run(tmp_path, monkeypatch, missing=(section, field))

    with pytest.raises(EvalError, match=field):
        load_eval_run(run_root)


def test_public_sample_excludes_private_grader_fields() -> None:
    image_id = "sha256:" + "1" * 64
    client = Client(
        [
            result(
                json.dumps(
                    [
                        {
                            "Id": image_id,
                            "RepoDigests": [],
                            "Os": "linux",
                            "Architecture": "amd64",
                        }
                    ]
                )
            )
        ]
    )
    row = {
        "instance_id": "astropy__astropy-14539",
        "repo": "astropy/astropy",
        "base_commit": "a" * 40,
        "problem_statement": "public issue",
        "patch": "GOLD SECRET",
        "test_patch": "TEST SECRET",
        "FAIL_TO_PASS": "PRIVATE",
        "PASS_TO_PASS": "PRIVATE",
    }
    sample = public_sample_from_row(row, protocol(), client)
    serialized = sample.model_dump_json()
    assert sample.task.problem_statement == "public issue"
    assert "SECRET" not in serialized and "PRIVATE" not in serialized


def test_official_image_name_matches_harness_namespace_rule() -> None:
    assert official_image_name("astropy__astropy-14539") == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest"
    )


def test_eval_sandbox_accepts_prep_parent_then_checks_out(sample_factory) -> None:
    sample = sample_factory()
    client = Client(
        [
            result("b" * 40 + "\n"),
            result(sample.task.base_commit + "\n"),
            result(),
            result(sample.task.base_commit + "\n"),
            result(),
        ]
    )
    sandbox = EvalDockerSandbox(
        client=client,
        task=sample.task,
        environment=sample.environment,
        run_id="eval",
        episode_id="episode",
        scope="rollout",
    )
    sandbox.container_id = "a" * 64
    sandbox.started = True
    sandbox._verify_base_contract()
    assert any(command[-3:] == ["checkout", "--detach", sample.task.base_commit] for command in client.commands)


@pytest.fixture
def sample_factory():
    from siete_rl.models import Environment, Sample, Task

    def make():
        task = Task(
            task_id="owner__repo-1",
            repo_name="owner/repo",
            base_commit="a" * 40,
            problem_statement="fix it",
        )
        environment = Environment(
            environment_id="eval:1",
            task_id=task.task_id,
            image_name="image",
            expected_image_id="sha256:" + "1" * 64,
            expected_registry_digest="sha256:" + "2" * 64,
            workdir="/testbed",
            cpus=1,
            memory="1g",
            pids_limit=64,
            exec_timeout_sec=30,
            verifier_timeout_sec=60,
        )
        return Sample(task=task, environment=environment)

    return make


def test_predictions_keep_empty_and_infrastructure_failures() -> None:
    outcomes = [
        EvalOutcome("a", "", "submitted", None, [], 1.0),
        EvalOutcome("b", "", "infra_error", "boom", [], 1.0),
    ]
    predictions = build_predictions(outcomes, model_name="candidate", task_ids=["a", "b"])
    assert predictions == [
        {"instance_id": "a", "model_name_or_path": "candidate", "model_patch": ""},
        {"instance_id": "b", "model_name_or_path": "candidate", "model_patch": ""},
    ]


def test_evaluate_variant_single_worker_prepares_each_sample_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{"instance_id": "task-0"}, {"instance_id": "task-1"}]
    events: list[str] = []

    def make_sample(row, *_args):
        task_id = row["instance_id"]
        events.append(f"prepare:{task_id}")
        if task_id == "task-1":
            raise EvalError("missing image")
        return SimpleNamespace(task=SimpleNamespace(task_id=task_id))

    def run_loop(*, sample, **_kwargs):
        events.append(f"run:{sample.task.task_id}")
        return EvalOutcome(sample.task.task_id, "patch", "submitted", None, [], 0.0)

    monkeypatch.setattr(eval_module, "public_sample_from_row", make_sample)
    monkeypatch.setattr(eval_module, "run_agent_loop", run_loop)
    with pytest.raises(EvalError, match="missing image"):
        eval_module.evaluate_variant(
            name="candidate",
            model_name="candidate",
            rows=rows,
            run=SimpleNamespace(protocol=protocol()),
            tokenizer=object(),
            docker_client=Client([]),
            server_url="http://unused",
            eval_run_id="eval",
            output_dir=tmp_path,
            rollout_workers=1,
        )
    assert events == ["prepare:task-0", "run:task-0", "prepare:task-1"]
    state = json.loads((tmp_path / "rollout-state.json").read_text(encoding="utf-8"))
    assert state["completed"] == 1


def test_evaluate_variant_preserves_order_when_workers_finish_out_of_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{"instance_id": f"task-{index}"} for index in range(4)]
    use_recording_executor(monkeypatch)

    def make_sample(row, *_args):
        return SimpleNamespace(task=SimpleNamespace(task_id=row["instance_id"]))

    def run_loop(*, sample, **_kwargs):
        index = int(sample.task.task_id.rsplit("-", 1)[1])
        infrastructure_error = "docker unavailable" if index == 2 else None
        patch = "" if infrastructure_error else f"patch-{index}"
        termination = "infra_error" if infrastructure_error else "submitted"
        return EvalOutcome(
            sample.task.task_id,
            patch,
            termination,
            infrastructure_error,
            [],
            0.0,
        )

    monkeypatch.setattr(eval_module, "public_sample_from_row", make_sample)
    monkeypatch.setattr(eval_module, "run_agent_loop", run_loop)
    monkeypatch.setattr(
        eval_module, "as_completed", lambda futures: reversed(list(futures))
    )
    outcomes = eval_module.evaluate_variant(
        name="candidate",
        model_name="candidate",
        rows=rows,
        run=SimpleNamespace(protocol=protocol()),
        tokenizer=object(),
        docker_client=Client([]),
        server_url="http://unused",
        eval_run_id="eval",
        output_dir=tmp_path,
        rollout_workers=4,
    )
    expected_ids = ["task-0", "task-1", "task-2", "task-3"]
    assert [outcome.task_id for outcome in outcomes] == expected_ids
    state = json.loads((tmp_path / "rollout-state.json").read_text(encoding="utf-8"))
    assert state["completed"] == 4
    assert [item["instance_id"] for item in state["outcomes"]] == expected_ids
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["instance_id"] for item in predictions] == expected_ids
    assert predictions[2]["model_patch"] == ""


def test_evaluate_variant_cancels_pending_tasks_when_progress_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{"instance_id": f"task-{index}"} for index in range(6)]
    executors = use_recording_executor(monkeypatch)
    started: list[str] = []

    def make_sample(row, *_args):
        return SimpleNamespace(task=SimpleNamespace(task_id=row["instance_id"]))

    def run_loop(*, sample, **_kwargs):
        started.append(sample.task.task_id)
        return EvalOutcome(sample.task.task_id, "patch", "submitted", None, [], 0.0)

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(eval_module, "public_sample_from_row", make_sample)
    monkeypatch.setattr(eval_module, "run_agent_loop", run_loop)
    monkeypatch.setattr(eval_module, "_write_json", fail_write)
    monkeypatch.setattr(eval_module, "as_completed", lambda futures: list(futures))
    with pytest.raises(OSError, match="disk full"):
        eval_module.evaluate_variant(
            name="candidate",
            model_name="candidate",
            rows=rows,
            run=SimpleNamespace(protocol=protocol()),
            tokenizer=object(),
            docker_client=Client([]),
            server_url="http://unused",
            eval_run_id="eval",
            output_dir=tmp_path,
            rollout_workers=2,
        )
    assert started == ["task-0"]
    assert executors[0].shutdown_calls == [(True, True)]


def test_comparison_only_claims_delta_for_complete_local_runs() -> None:
    candidate = {
        "total_instances": 2,
        "completed_instances": 2,
        "resolved_instances": 1,
        "empty_patch_instances": 0,
        "error_instances": 0,
    }
    without_base = build_comparison(candidate_report=candidate, base_report=None)
    assert without_base["local_base"] == "not_run"
    assert "candidate_minus_local_base_percentage_points" not in without_base
    base = dict(candidate, resolved_instances=0)
    with_base = build_comparison(candidate_report=candidate, base_report=base)
    assert with_base["candidate_minus_local_base_percentage_points"] == 50.0


def test_eval_agent_loop_calls_the_existing_trainer_state_machine(
    monkeypatch: pytest.MonkeyPatch, sample_factory
) -> None:
    sample = sample_factory()
    seen = {}
    first = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "finish", "arguments": {}},
            }
        ],
    }

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return [1, 2]

    class Generator:
        def generate(self, prompt_ids, *args):
            del prompt_ids, args
            return [[3]], None

    def reset(self, task_id, **kwargs):
        del kwargs
        self._sample = sample
        self.episode_id = "episode"
        seen["reset"] = task_id

    def finish(self):
        self._steps.append(object())
        self._submitted = True
        self._frozen_patch = "patch"
        self._terminal_event = TerminalEvent(kind="submitted", step_index=0)
        return "submitted"

    from siete_rl.trainer import SWEGRPOTrainer

    execute_tool_calls = SWEGRPOTrainer._execute_tool_calls

    def observe_tools(self, tool_call_list, sync_tool_dict, async_tool_dict):
        seen["tools"] = set(sync_tool_dict)
        return execute_tool_calls(self, tool_call_list, sync_tool_dict, async_tool_dict)

    monkeypatch.setattr("siete_rl.eval.parse_response", lambda *args, **kwargs: first)
    monkeypatch.setattr("siete_rl.eval.EvalEnvironment.reset", reset)
    monkeypatch.setattr("siete_rl.eval.EvalEnvironment.finish", finish)
    monkeypatch.setattr(
        "siete_rl.eval.SWEGRPOTrainer._execute_tool_calls", observe_tools
    )
    outcome = run_agent_loop(
        sample=sample,
        tokenizer=Tokenizer(),
        generator=Generator(),
        protocol=protocol(),
        docker_client=Client([]),
        eval_run_id="eval",
    )
    assert outcome.infrastructure_error is None
    assert outcome.patch == "patch"
    assert outcome.termination == "submitted"
    assert outcome.messages == [first]
    assert seen["reset"] == sample.task.task_id
    assert seen["tools"] == {"execute_bash", "str_replace_editor", "finish"}
