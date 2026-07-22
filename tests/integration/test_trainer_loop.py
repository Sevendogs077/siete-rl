"""SWEGRPOTrainer._tool_call_loop 的关键路径测试：信号终止、回滚与结束归因。

沿用 test_trl_interfaces.py 的裸实例模式（object.__new__ + 手工装配循环触到的属性），
无需 GPU/vLLM/tokenizer。注意 parse_response 必须 patch `swe_agent.trainer` 的
模块命名空间，而不是 `trl.trainer.grpo_trainer` 的。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from trl import GRPOTrainer

from swe_agent.trainer import SWEGRPOTrainer


class FakeEnvironment:
    """最小环境替身：terminated 信号 + _record_loop_exit 契约。"""

    def __init__(self) -> None:
        self.terminated = False
        self.loop_exit: str | None = None

    def _record_loop_exit(self, reason: str) -> None:
        self.loop_exit = reason


def bare_trainer(*, max_iterations: int = 5, max_completion_length: int = 64) -> SWEGRPOTrainer:
    trainer = object.__new__(SWEGRPOTrainer)
    trainer.max_tool_calling_iterations = max_iterations
    trainer.max_completion_length = max_completion_length
    trainer.use_vllm = False
    trainer.vllm_mode = "colocate"
    trainer._is_vlm = False
    trainer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    trainer._tokenizer = object()
    trainer._get_tool_suffix_ids = lambda messages: [90] * len(messages)
    return trainer


def assistant_tool_call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def final_text(*args, **kwargs) -> dict:
    return {"role": "assistant", "content": "final", "tool_calls": []}


def looping_tool_call(*args, **kwargs) -> dict:
    return assistant_tool_call("inspect", {"path": "x.py"})


def run_loop(trainer, completions):
    return trainer._tool_call_loop(
        prompts=[[{"role": "user", "content": f"fix {i}"}] for i in range(len(completions))],
        prompt_ids=[[1] for _ in completions],
        completion_ids=[[2] for _ in completions],
        completions=completions,
        logprobs=None,
        images=None,
        multimodal_fields={},
    )


def test_mixed_batch_terminates_submitted_and_continues_others(monkeypatch) -> None:
    env_a, env_b = FakeEnvironment(), FakeEnvironment()
    trainer = bare_trainer()
    trainer.environments = [env_a, env_b]
    trainer._sync_tool_dicts = [
        {"submit": lambda: setattr(env_a, "terminated", True) or "submitted"},
        {"inspect": lambda path: f"contents:{path}"},
    ]
    trainer._async_tool_dicts = [{}, {}]
    trainer._generate_single_turn = lambda prompt_ids, *args: ([[91]] * len(prompt_ids), None)
    monkeypatch.setattr("swe_agent.trainer.parse_response", final_text)

    completions = [
        [assistant_tool_call("submit", {})],
        [assistant_tool_call("inspect", {"path": "a.py"})],
    ]
    prompts = [[{"role": "user", "content": "fix 0"}], [{"role": "user", "content": "fix 1"}]]
    tool_mask, returned, completion_ids, _, _, failures, _ = trainer._tool_call_loop(
        prompts=prompts,
        prompt_ids=[[1], [1]],
        completion_ids=[[2], [2]],
        completions=completions,
        logprobs=None,
        images=None,
        multimodal_fields={},
    )

    assert failures == 0
    # 终止样本：tool observation 被回滚，completion 结束于 submit 调用本身
    assert returned[0] == [assistant_tool_call("submit", {})]
    assert prompts[0] == [{"role": "user", "content": "fix 0"}]
    assert completion_ids[0] == [2]
    assert tool_mask[0] == [1]
    assert env_a.terminated is True
    assert env_a.loop_exit is None  # 终态由环境自己的 TerminalEvent 表达
    # 未终止样本：正常收到 observation 并继续生成，结束后归因 model_stopped
    assert [message["role"] for message in returned[1]] == ["assistant", "tool", "assistant"]
    assert returned[1][1]["content"] == "contents:a.py"
    assert returned[1][-1]["content"] == "final"
    assert completion_ids[1] == [2, 90, 91]
    assert tool_mask[1] == [1, 0, 1]
    assert env_b.loop_exit == "model_stopped"


def test_all_samples_terminated_breaks_before_regeneration(monkeypatch) -> None:
    envs = [FakeEnvironment(), FakeEnvironment()]
    trainer = bare_trainer()
    trainer.environments = envs
    trainer._sync_tool_dicts = [
        {"submit": lambda: setattr(envs[0], "terminated", True) or "submitted"},
        {"submit": lambda: setattr(envs[1], "terminated", True) or "submitted"},
    ]
    trainer._async_tool_dicts = [{}, {}]

    def forbidden_generation(*args):
        raise AssertionError("terminated batch must not trigger another generation")

    trainer._generate_single_turn = forbidden_generation
    completions = [[assistant_tool_call("submit", {})], [assistant_tool_call("submit", {})]]
    _, returned, completion_ids, _, call_count, _, _ = run_loop(trainer, completions)

    assert call_count == 2
    assert returned == [[assistant_tool_call("submit", {})], [assistant_tool_call("submit", {})]]
    assert completion_ids == [[2], [2]]
    assert all(env.loop_exit is None for env in envs)


def test_iteration_cap_is_attributed(monkeypatch) -> None:
    env = FakeEnvironment()
    trainer = bare_trainer(max_iterations=2)
    trainer.environments = [env]
    trainer._sync_tool_dicts = [{"inspect": lambda path: "contents"}]
    trainer._async_tool_dicts = [{}]
    trainer._generate_single_turn = lambda prompt_ids, *args: ([[91]] * len(prompt_ids), None)
    monkeypatch.setattr("swe_agent.trainer.parse_response", looping_tool_call)

    run_loop(trainer, [[assistant_tool_call("inspect", {"path": "x.py"})]])

    assert env.loop_exit == "iteration_cap"


def test_sample_without_tool_call_is_model_stopped() -> None:
    env = FakeEnvironment()
    trainer = bare_trainer()
    trainer.environments = [env]
    trainer._sync_tool_dicts = [{}]
    trainer._async_tool_dicts = [{}]

    tool_mask, returned, completion_ids, _, call_count, _, _ = run_loop(
        trainer, [[{"role": "assistant", "content": "no tools today"}]]
    )

    assert call_count == 0
    assert returned == [[{"role": "assistant", "content": "no tools today"}]]
    assert completion_ids == [[2]]
    assert tool_mask == [[1]]
    assert env.loop_exit == "model_stopped"


def test_overlong_sample_is_rolled_back_and_attributed() -> None:
    env = FakeEnvironment()
    trainer = bare_trainer(max_completion_length=1)
    trainer.environments = [env]
    trainer._sync_tool_dicts = [{"inspect": lambda path: "contents"}]
    trainer._async_tool_dicts = [{}]

    _, returned, completion_ids, _, _, _, _ = run_loop(
        trainer, [[assistant_tool_call("inspect", {"path": "x.py"})]]
    )

    assert returned == [[assistant_tool_call("inspect", {"path": "x.py"})]]
    assert completion_ids == [[2]]
    assert env.loop_exit == "context_overlong"


def test_loop_without_environments_matches_stock_behavior(monkeypatch) -> None:
    trainer = bare_trainer()
    trainer.environments = None
    calls: list[str] = []
    trainer._sync_tool_dicts = [{"inspect": lambda path: calls.append(path) or "contents"}]
    trainer._async_tool_dicts = [{}]
    trainer._generate_single_turn = lambda prompt_ids, *args: ([[91]] * len(prompt_ids), None)
    monkeypatch.setattr("swe_agent.trainer.parse_response", final_text)

    tool_mask, returned, completion_ids, _, _, _, _ = run_loop(
        trainer, [[assistant_tool_call("inspect", {"path": "a.py"})]]
    )

    assert calls == ["a.py"]
    assert returned[0][1]["content"] == "contents"
    assert returned[0][-1]["content"] == "final"
    assert completion_ids[0] == [2, 90, 91]
    assert tool_mask[0] == [1, 0, 1]


def test_tool_call_loop_mirrors_trl_source() -> None:
    """剥离 swe_agent 插入块后必须与 trl 原版逐行一致（忽略空行）；TRL 升级时此测试应变红。"""

    original = inspect.getsource(GRPOTrainer._tool_call_loop)
    mirrored = inspect.getsource(SWEGRPOTrainer._tool_call_loop)

    assert mirrored.count("# >>> swe_agent") >= 4
    assert mirrored.count("# >>> swe_agent") == mirrored.count("# <<< swe_agent")

    stripped, skip = [], False
    for line in mirrored.splitlines():
        if "# >>> swe_agent" in line:
            skip = True
            continue
        if "# <<< swe_agent" in line:
            skip = False
            continue
        if not skip:
            stripped.append(line)

    compact = lambda source: [line for line in source.splitlines() if line.strip()]
    assert compact("\n".join(stripped)) == compact(original)
