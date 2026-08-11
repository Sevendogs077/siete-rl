"""OpenHands loop 的无 GPU 状态机回归。"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
from trl import GRPOTrainer

from siete_rl.models import Action, Observation, Step
from siete_rl.process_mask import build_process_token_weights
from siete_rl.trainer import SWEGRPOTrainer


class Environment:
    def __init__(self): self.terminated = False; self.loop_exit = None; self.turn_records = []; self._steps = []
    def _record_loop_exit(self, reason): self.loop_exit = reason


def trainer(*, maximum=5, protocol_errors=2):
    value = object.__new__(SWEGRPOTrainer)
    value.max_tool_calling_iterations = maximum; value.max_consecutive_protocol_errors = protocol_errors; value.max_completion_length = 64; value._tool_parallel_workers = 1
    value.use_vllm = False; value.vllm_mode = "server"; value._is_vlm = False; value.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)); value._tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0); value._get_tool_suffix_ids = lambda messages: [90] * len(messages)
    return value


def call(name, arguments=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments or {}}}]}


def run(value, completion, *, completion_ids=None):
    return value._tool_call_loop(prompts=[[{"role": "user", "content": "fix"}]], prompt_ids=[[1]], completion_ids=[completion_ids or [2, 99]], completions=[[completion]], logprobs=None, images=None, multimodal_fields={})


class ResetEnvironment:
    def __init__(self, name, events, *, timing, result=None, error=None):
        self.name = name
        self.events = events
        self.timing = timing
        self.result = result
        self.error = error

    def _await_reset(self):
        self.events.append(f"await:{self.name}")
        if self.error is not None:
            raise self.error
        return self.result

    def _reset_timing(self):
        return self.timing


def reset_trainer(environments):
    value = object.__new__(SWEGRPOTrainer)
    value.environments = environments
    value.model = SimpleNamespace(training=True)
    value._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return value


def test_generate_drains_resets_before_parent_and_records_batch_span(
    monkeypatch,
) -> None:
    events = []
    value = reset_trainer(
        [
            ResetEnvironment("a", events, timing=(11.0, 14.0)),
            ResetEnvironment("b", events, timing=(10.0, 16.5)),
        ]
    )

    def parent_generate(self, prompts):
        events.append("parent")
        return prompts

    monkeypatch.setattr(GRPOTrainer, "_generate", parent_generate)

    assert value._generate(["prompt"]) == ["prompt"]
    assert events == ["await:a", "await:b", "parent"]
    assert value._metrics["train"]["environment/reset_time"] == [6.5]


def test_generate_drains_every_reset_before_raising_first_error(monkeypatch) -> None:
    events = []
    first = RuntimeError("first reset failed")
    second = ValueError("second reset failed")
    value = reset_trainer(
        [
            ResetEnvironment("a", events, timing=(1.0, 2.0), error=first),
            ResetEnvironment("b", events, timing=(1.0, 3.0), error=second),
            ResetEnvironment("c", events, timing=(1.0, 4.0)),
        ]
    )
    monkeypatch.setattr(
        GRPOTrainer,
        "_generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent generation must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="first reset failed") as captured:
        value._generate(["prompt"])

    assert events == ["await:a", "await:b", "await:c"]
    assert any("second reset failed" in note for note in captured.value.__notes__)


def test_generate_rejects_non_silent_reset_after_draining_batch(monkeypatch) -> None:
    events = []
    value = reset_trainer(
        [
            ResetEnvironment("a", events, timing=(1.0, 2.0), result="observation"),
            ResetEnvironment("b", events, timing=(1.0, 2.5)),
        ]
    )
    monkeypatch.setattr(
        GRPOTrainer,
        "_generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent generation must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="must return None"):
        value._generate(["prompt"])

    assert events == ["await:a", "await:b"]


def test_finish_keeps_model_tokens_and_adds_no_observation() -> None:
    env = Environment(); value = trainer(); value.environments = [env]; value._sync_tool_dicts = [{"finish": lambda: setattr(env, "terminated", True) or ""}]; value._async_tool_dicts = [{}]
    mask, completions, ids, _, count, failures, _ = run(
        value, call("finish"), completion_ids=[2]
    )
    assert count == 1 and failures == 0 and env.terminated
    assert completions == [[call("finish")]]
    assert ids == [[2]] and mask == [[1]]
    assert env.loop_exit == "context_overlong"
    assert env.turn_records[-1].truncated is True


def test_active_row_filter_keeps_rollback_snapshot_aligned() -> None:
    envs = [Environment(), Environment()]
    value = trainer()
    value.max_completion_length = 1
    value.environments = envs

    def finish():
        envs[0]._steps.append(object())
        envs[0].terminated = True
        return ""

    def bash():
        envs[1]._steps.append(object())
        return "obs"

    value._sync_tool_dicts = [{"finish": finish}, {"bash": bash}]
    value._async_tool_dicts = [{}, {}]
    prompts = [
        [{"role": "user", "content": "short"}],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "keep both"},
        ],
    ]

    value._tool_call_loop(
        prompts=prompts,
        prompt_ids=[[1], [2]],
        completion_ids=[[10, 99], [20, 99]],
        completions=[[call("finish")], [call("bash")]],
        logprobs=None,
        images=None,
        multimodal_fields={},
    )

    assert len(prompts[1]) == 2
    assert prompts[1][-1]["content"] == "keep both"
    assert envs[1].loop_exit == "context_overlong"


def test_plain_message_gets_fake_user_and_continues_to_finish(monkeypatch) -> None:
    env = Environment(); value = trainer(); value.environments = [env]; value._sync_tool_dicts = [{"finish": lambda: setattr(env, "terminated", True) or ""}]; value._async_tool_dicts = [{}]
    value._generate_single_turn = lambda ids, *args: ([[91]], None)
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: call("finish"))
    mask, completions, ids, *_ = run(value, {"role": "assistant", "content": "I will inspect this."})
    assert "Please continue working" in completions[0][1]["content"]
    assert completions[0][-1] == call("finish")
    assert ids[0] == [2, 99, 90, 91] and mask[0] == [1, 1, 0, 1]


def test_protocol_error_retries_then_records_format_exhaustion(monkeypatch) -> None:
    env = Environment(); value = trainer(protocol_errors=1); value.environments = [env]; value._sync_tool_dicts = [{}]; value._async_tool_dicts = [{}]
    completion = {"role": "assistant", "content": "<function=finish>", "parse_error": "incomplete function call"}
    mask, completions, ids, *_ = run(value, completion)
    assert env.loop_exit == "format_exhausted"
    assert completions == [[completion]] and ids == [[2, 99]] and mask == [[1, 1]]


def test_turn_records_track_kinds_intervals_and_step_backfill(monkeypatch) -> None:
    # 3 turn：真实 bash 调用 → parse 错误恢复 → finish；fake 工具追加 _steps 模拟 _call_tool 的 Step 语义
    env = Environment(); value = trainer(); value.environments = [env]
    def bash(): env._steps.append(object()); return "obs"
    def finish(): env._steps.append(object()); env.terminated = True; return ""
    value._sync_tool_dicts = [{"bash": bash, "finish": finish}]; value._async_tool_dicts = [{}]
    generations = iter([([91, 92, 99], None), ([93, 99], None)])
    value._generate_single_turn = lambda ids, *args: ([next(generations)[0]], None)
    parses = iter([{"role": "assistant", "content": "oops", "parse_error": "bad call"}, call("finish")])
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: next(parses))
    run(value, call("bash"))
    records = env.turn_records
    assert [r.kind for r in records] == ["step", "invalid_call", "step"]
    assert [r.step_index for r in records] == [0, None, 1]
    assert all(r.token_start < r.token_end for r in records)
    assert all(a.token_end <= b.token_start for a, b in zip(records, records[1:]))


def test_iteration_cap_preserves_unexecuted_pending_action(monkeypatch) -> None:
    env = Environment(); value = trainer(maximum=1); value.environments = [env]

    def bash():
        env._steps.append(
            Step(
                index=0,
                action=Action(tool_name="execute_bash", arguments={"command": "pwd"}),
                observation=Observation(text="/repo", exit_code=0),
            )
        )
        return "/repo"

    value._sync_tool_dicts = [{"bash": bash}]; value._async_tool_dicts = [{}]
    value._generate_single_turn = lambda ids, *args: ([[91, 99]], None)
    monkeypatch.setattr(
        "siete_rl.trainer.parse_response",
        lambda *args, **kwargs: call(
            "str_replace_editor", {"command": "view", "path": "/repo/a.py"}
        ),
    )

    mask, *_ = run(value, call("bash"))

    assert len(env._steps) == 1
    assert env.loop_exit == "iteration_cap"
    assert [r.kind for r in env.turn_records] == ["step", "pending_action"]
    assert [r.step_index for r in env.turn_records] == [0, None]

    weights, stats = build_process_token_weights(
        turns=env.turn_records,
        steps=env._steps,
        termination="iteration_cap",
        advantage=0.5,
        base_mask=mask[0],
    )
    assert weights == mask[0]
    assert stats.candidate_turns == 0


def test_observation_overlong_keeps_complete_turn_credit() -> None:
    env = Environment()
    value = trainer()
    value.max_completion_length = 2
    value.environments = [env]

    def bash():
        env._steps.append(
            Step(
                index=0,
                action=Action(tool_name="execute_bash", arguments={"command": "pwd"}),
                observation=Observation(text="/repo", exit_code=0),
            )
        )
        return "/repo"

    value._sync_tool_dicts = [{"bash": bash}]
    value._async_tool_dicts = [{}]
    value._tool_call_loop(
        prompts=[[{"role": "user", "content": "fix"}]],
        prompt_ids=[[1]],
        completion_ids=[[2, 99]],
        completions=[[call("bash")]],
        logprobs=None,
        images=None,
        multimodal_fields={},
    )

    assert env.loop_exit == "context_overlong"
    assert [turn.truncated for turn in env.turn_records] == [False]


def test_physical_slice_marks_only_final_executed_turn_truncated(monkeypatch) -> None:
    env = Environment()
    value = trainer()
    value.max_completion_length = 4
    value.environments = [env]

    def bash():
        index = len(env._steps)
        env._steps.append(
            Step(
                index=index,
                action=Action(tool_name="execute_bash", arguments={"command": "pwd"}),
                observation=Observation(text="/repo", exit_code=0),
            )
        )
        return "/repo"

    value._sync_tool_dicts = [{"bash": bash}]
    value._async_tool_dicts = [{}]
    value._generate_single_turn = lambda ids, *args: ([[91, 92, 93]], None)
    monkeypatch.setattr(
        "siete_rl.trainer.parse_response", lambda *args, **kwargs: call("bash")
    )
    value._tool_call_loop(
        prompts=[[{"role": "user", "content": "fix"}]],
        prompt_ids=[[1]],
        completion_ids=[[2, 99]],
        completions=[[call("bash")]],
        logprobs=None,
        images=None,
        multimodal_fields={},
    )

    assert env.loop_exit == "context_overlong"
    assert [turn.kind for turn in env.turn_records] == ["step", "step"]
    assert [turn.truncated for turn in env.turn_records] == [False, True]


def test_post_tool_truncated_finish_is_context_overlong(monkeypatch) -> None:
    env = Environment()
    value = trainer()
    value.environments = [env]

    def step(name, *, terminate=False):
        def execute():
            env._steps.append(object())
            if terminate:
                env.terminated = True
            return name

        return execute

    value._sync_tool_dicts = [
        {"bash": step("bash"), "finish": step("finish", terminate=True)}
    ]
    value._async_tool_dicts = [{}]
    value._generate_single_turn = lambda ids, *args: ([[91]], None)
    monkeypatch.setattr(
        "siete_rl.trainer.parse_response", lambda *args, **kwargs: call("finish")
    )

    run(value, call("bash"))

    assert env.terminated is True
    assert env.loop_exit == "context_overlong"
    assert [turn.kind for turn in env.turn_records] == ["step", "step"]
    assert [turn.truncated for turn in env.turn_records] == [False, True]


class FakeServerVLLM:
    """忠实模拟 TRL 1.8.0 server 分支:`prompts[::num_generations]` 去重 + 每 prompt n 条 + prompt-major 展平。

    completion token 取其 prompt 的 sum(可溯源到具体 history);logprobs 结构同
    `VLLMGeneration.generate` server 返回值(per-token top-k list)。
    """

    def __init__(self):
        self.num_generations_seen = []

    def generate(self, *, prompts, images, num_generations, profiler=None):
        del images, profiler
        self.num_generations_seen.append(num_generations)
        ordered = prompts[::num_generations]
        completion_ids, logprobs = [], []
        for p in ordered:
            for _ in range(num_generations):
                completion_ids.append([sum(p)])
                logprobs.append([[float(sum(p))]])
        # 单进程 process_slice:仅保留前 len(prompts) 条(同 VLLMGeneration.generate server 分支)
        prompts_out = [p for p in ordered for _ in range(num_generations)][: len(prompts)]
        return prompts_out, completion_ids[: len(prompts)], logprobs[: len(prompts)], None


def test_post_tool_regeneration_preserves_per_trajectory_lineage(monkeypatch) -> None:
    """K 条 distinct history 必须得到 K 条一一对应 continuation(#6673 回归)。

    修复前:server stride 去重把 K=3 塌缩到 history 0,所有 continuation sum=102 → 失败。
    修复后:helper 以 num_generations=1 逐条生成 → 通过。
    """
    value = trainer()
    value.use_vllm = True  # trainer() 默认 False;本测试走 vLLM 分支
    value.state = SimpleNamespace(global_step=0)
    value._last_loaded_step = 0  # 与 global_step 相等 → 跳过 weight-sync
    # profiling_context 需要这两个属性(report_to=[] 时只计时,无任何副作用)
    value.args = SimpleNamespace(report_to=[])
    value.accelerator = SimpleNamespace(is_main_process=True)
    value.vllm_generation = FakeServerVLLM()
    value.environments = []
    value._sync_tool_dicts = [{"bash": lambda: "obs"} for _ in range(3)]
    value._async_tool_dicts = [{} for _ in range(3)]
    # 修复前路径的忠实替身:TRL `_generate_single_turn` 在 server 模式以 num_generations=16 调用。
    # 修复后 loop 不再调用它(应被 _generate_tool_loop_turn 取代),此替身变为惰性。
    def legacy_single_turn(ids, images, fields):
        _, completion_ids, logprobs, _ = value.vllm_generation.generate(
            prompts=ids, images=images, num_generations=16
        )
        return completion_ids, [[lp[0] for lp in seq] for seq in logprobs]
    value._generate_single_turn = legacy_single_turn
    # 一轮 post-tool 生成后即退出:空 completion 归因 context_overlong,不再迭代
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: {})
    mask, completions, ids, logprobs, *_ = value._tool_call_loop(
        prompts=[[{"role": "user", "content": "fix"}] for _ in range(3)],
        prompt_ids=[[1], [2], [3]],
        completion_ids=[[11], [22], [33]],
        completions=[[call("bash")] for _ in range(3)],
        logprobs=[[0.0], [0.0], [0.0]],
        images=None,
        multimodal_fields={},
    )
    # 每条 trajectory:completion + suffix(90) + 派生自**自己** history 的 continuation
    # history = prompt + completion + suffix:sum 分别为 102/114/126
    assert ids == [[11, 90, 102], [22, 90, 114], [33, 90, 126]]
    # logprobs 与 trajectory 索引一一对应:[首轮, tool 段 0.0, continuation 自己的 logprob]
    assert logprobs == [[0.0, 0.0, 102.0], [0.0, 0.0, 114.0], [0.0, 0.0, 126.0]]
    assert mask == [[1, 0, 1]] * 3
    # 修复后 loop 必须走 num_generations=1(且不再调用 legacy 替身)
    assert value.vllm_generation.num_generations_seen == [1]


def test_parallel_tool_failure_isolated_to_failing_sample(monkeypatch) -> None:
    """单样本工具抛错时，仅该样本得到 error observation。"""
    envs = [Environment() for _ in range(3)]
    value = trainer()
    value._tool_parallel_workers = 8
    value.environments = envs

    def ok(env):
        def _ok():
            env._steps.append(object())
            return "ok"

        return _ok

    def bad():
        raise RuntimeError("boom")

    value._sync_tool_dicts = [
        {"bash": ok(envs[0])},
        {"bash": bad},
        {"bash": ok(envs[2])},
    ]
    value._async_tool_dicts = [{} for _ in range(3)]
    value._generate_single_turn = lambda ids, *args: (
        [[91, 99]] * len(ids),
        [[0.5, 0.5]] * len(ids),
    )
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: {})
    _, completions, _, _, count, failures, _ = value._tool_call_loop(
        prompts=[[{"role": "user", "content": "fix"}] for _ in range(3)],
        prompt_ids=[[1], [2], [3]],
        completion_ids=[[11, 99], [22, 99], [33, 99]],
        completions=[[call("bash")] for _ in range(3)],
        logprobs=[[0.0] for _ in range(3)],
        images=None,
        multimodal_fields={},
    )
    assert count == 3 and failures == 1
    observations = [completion[1]["content"] for completion in completions]
    assert (
        "boom" not in observations[0]
        and "boom" in observations[1]
        and "boom" not in observations[2]
    )
    assert envs[1].turn_records[-1].kind == "invalid_call"
    assert (
        envs[0].turn_records[-1].step_index == 0
        and envs[2].turn_records[-1].step_index == 0
    )
