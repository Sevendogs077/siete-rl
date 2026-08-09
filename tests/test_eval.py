from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize("value,expected", [(None, False), ("False", False), ("True", True)])
def test_eval_base_is_strict(value, expected) -> None:
    assert parse_strict_bool(value) is expected


@pytest.mark.parametrize("value", ["true", "false", "1", "", "yes"])
def test_eval_base_rejects_other_spellings(value) -> None:
    with pytest.raises(EvalError, match="exactly"):
        parse_strict_bool(value)


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


# 依赖私有 outputs/ 下的真实评测运行目录。
@pytest.mark.external_assets
def test_last_two_real_runs_supply_budget_without_defaults() -> None:
    for name in ("20260801T235407Z-149b", "20260802T065312Z-b873"):
        run = load_eval_run(PROJECT_ROOT / "outputs" / name)
        assert run.protocol.max_tool_calling_iterations == 40
        assert run.protocol.max_prompt_length == 8192
        assert run.protocol.max_completion_length == 24576
        assert run.protocol.max_model_length == 32768


def test_harness_resolver_fails_closed_without_eval_harness_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_HARNESS_PYTHON", raising=False)
    with pytest.raises(EvalError, match="EVAL_HARNESS_PYTHON"):
        resolve_harness()


# 依赖 .external 下的真实评测 harness 检出。
@pytest.mark.external_assets
def test_harness_resolver_accepts_clean_checkout() -> None:
    root, python = resolve_harness()
    assert root.name == "swe-bench"
    assert python.name == "python"
    assert python.parent.name == "bin"
    assert python.parent.parent.name == ".venv"


# 依赖私有 outputs/ 下的真实评测运行目录。
@pytest.mark.external_assets
def test_missing_budget_fails_closed(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "outputs/20260801T235407Z-149b"
    run = PROJECT_ROOT / "outputs" / f"pytest-eval-missing-{tmp_path.name}"
    run.mkdir()
    try:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (run / name).symlink_to(source / name)
        config = (source / "config.yaml").read_text(encoding="utf-8")
        config = config.replace("  max_tool_calling_iterations: 40\n", "")
        (run / "config.yaml").write_text(config, encoding="utf-8")
        with pytest.raises(EvalError, match="max_tool_calling_iterations"):
            load_eval_run(run)
    finally:
        for path in run.iterdir():
            path.unlink()
        run.rmdir()


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
