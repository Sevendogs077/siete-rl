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
