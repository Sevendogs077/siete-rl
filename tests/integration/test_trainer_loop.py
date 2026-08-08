"""OpenHands loop 的无 GPU 状态机回归。"""

from __future__ import annotations

from types import SimpleNamespace

from siete_rl.trainer import SWEGRPOTrainer


class Environment:
    def __init__(self): self.terminated = False; self.loop_exit = None; self.turn_records = []; self._steps = []
    def _record_loop_exit(self, reason): self.loop_exit = reason


def trainer(*, maximum=5, protocol_errors=2):
    value = object.__new__(SWEGRPOTrainer)
    value.max_tool_calling_iterations = maximum; value.max_consecutive_protocol_errors = protocol_errors; value.max_completion_length = 64
    value.use_vllm = False; value.vllm_mode = "server"; value._is_vlm = False; value.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)); value._tokenizer = object(); value._get_tool_suffix_ids = lambda messages: [90] * len(messages)
    return value


def call(name, arguments=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments or {}}}]}


def run(value, completion):
    return value._tool_call_loop(prompts=[[{"role": "user", "content": "fix"}]], prompt_ids=[[1]], completion_ids=[[2]], completions=[[completion]], logprobs=None, images=None, multimodal_fields={})


def test_finish_keeps_model_tokens_and_adds_no_observation() -> None:
    env = Environment(); value = trainer(); value.environments = [env]; value._sync_tool_dicts = [{"finish": lambda: setattr(env, "terminated", True) or ""}]; value._async_tool_dicts = [{}]
    mask, completions, ids, _, count, failures, _ = run(value, call("finish"))
    assert count == 1 and failures == 0 and env.terminated
    assert completions == [[call("finish")]]
    assert ids == [[2]] and mask == [[1]]


def test_plain_message_gets_fake_user_and_continues_to_finish(monkeypatch) -> None:
    env = Environment(); value = trainer(); value.environments = [env]; value._sync_tool_dicts = [{"finish": lambda: setattr(env, "terminated", True) or ""}]; value._async_tool_dicts = [{}]
    value._generate_single_turn = lambda ids, *args: ([[91]], None)
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: call("finish"))
    mask, completions, ids, *_ = run(value, {"role": "assistant", "content": "I will inspect this."})
    assert "Please continue working" in completions[0][1]["content"]
    assert completions[0][-1] == call("finish")
    assert ids[0] == [2, 90, 91] and mask[0] == [1, 0, 1]


def test_protocol_error_retries_then_records_format_exhaustion(monkeypatch) -> None:
    env = Environment(); value = trainer(protocol_errors=1); value.environments = [env]; value._sync_tool_dicts = [{}]; value._async_tool_dicts = [{}]
    completion = {"role": "assistant", "content": "<function=finish>", "parse_error": "incomplete function call"}
    mask, completions, ids, *_ = run(value, completion)
    assert env.loop_exit == "format_exhausted"
    assert completions == [[completion]] and ids == [[2]] and mask == [[1]]


def test_turn_records_track_kinds_intervals_and_step_backfill(monkeypatch) -> None:
    # 3 turn：真实 bash 调用 → parse 错误恢复 → finish；fake 工具追加 _steps 模拟 _call_tool 的 Step 语义
    env = Environment(); value = trainer(); value.environments = [env]
    def bash(): env._steps.append(object()); return "obs"
    def finish(): env._steps.append(object()); env.terminated = True; return ""
    value._sync_tool_dicts = [{"bash": bash, "finish": finish}]; value._async_tool_dicts = [{}]
    generations = iter([([91, 92], None), ([93], None)])
    value._generate_single_turn = lambda ids, *args: ([next(generations)[0]], None)
    parses = iter([{"role": "assistant", "content": "oops", "parse_error": "bad call"}, call("finish")])
    monkeypatch.setattr("siete_rl.trainer.parse_response", lambda *args, **kwargs: next(parses))
    run(value, call("bash"))
    records = env.turn_records
    assert [r.kind for r in records] == ["step", "invalid_call", "step"]
    assert [r.step_index for r in records] == [0, None, 1]
    assert all(r.token_start < r.token_end for r in records)
    assert all(a.token_end <= b.token_start for a, b in zip(records, records[1:]))


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


def test_initial_turn_generation_path_is_not_overridden() -> None:
    """invariant 1 守护:首 turn 仍走 TRL `_generate_single_turn`(n=16 去重优化不动)。"""
    from trl import GRPOTrainer

    assert SWEGRPOTrainer._generate_single_turn is GRPOTrainer._generate_single_turn
