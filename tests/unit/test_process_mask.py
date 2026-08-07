from __future__ import annotations

import pytest

from siete_rl.models import Action, Observation, Step
from siete_rl.process_mask import (
    RULE_REGISTRY,
    DuplicateActionMask,
    InvalidCallMask,
    TurnRecord,
    action_signature,
    build_alpha,
    resolve_rules,
)


def _step(tool_name: str, arguments: dict) -> Step:
    return Step(
        index=0,
        action=Action(tool_name=tool_name, arguments=arguments),
        observation=Observation(text="ok", exit_code=0),
    )


BASH = {"command": "ls"}
EDITOR = {"command": "str_replace", "path": "/a.py", "old_str": "x" * 300}
ALL_RULES = resolve_rules(["invalid_call", "duplicate_action"])


def _alpha(turns, steps, n_tokens, rules):
    """build_alpha 返回 (alpha, masked_turns)，既有用例只关心 alpha。"""
    alpha, _ = build_alpha(turns, steps, n_tokens, rules)
    return alpha


class TestInvalidCallMask:
    rule = InvalidCallMask()

    def test_invalid_call_masked(self):
        turn = TurnRecord(0, 10, "invalid_call", None)
        assert self.rule.masked(turn, None) is True

    @pytest.mark.parametrize("kind", ["step", "plain_message"])
    def test_other_kinds_not_masked(self, kind: str):
        turn = TurnRecord(0, 10, kind, 0)
        assert self.rule.masked(turn, None) is False


class TestDuplicateActionMask:
    rule = DuplicateActionMask()

    @pytest.mark.parametrize("occurrence", [1, 2])
    def test_below_threshold_not_masked(self, occurrence: int):
        turn = TurnRecord(0, 10, "step", 0)
        assert self.rule.masked(turn, occurrence) is False

    @pytest.mark.parametrize("occurrence", [3, 4, 100])
    def test_at_or_above_threshold_masked(self, occurrence: int):
        turn = TurnRecord(0, 10, "step", 0)
        assert self.rule.masked(turn, occurrence) is True

    def test_none_occurrence_not_masked(self):
        turn = TurnRecord(0, 10, "step", 0)
        assert self.rule.masked(turn, None) is False


class TestBuildAlpha:
    def test_no_turns_all_ones(self):
        assert _alpha([], [], 5, ALL_RULES) == [1.0] * 5

    def test_invalid_call_turn_zeroed(self):
        turns = [TurnRecord(2, 4, "invalid_call", None)]
        assert _alpha(turns, [], 6, ALL_RULES) == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    def test_third_duplicate_bash_masked(self):
        # 成败均计数，签名只看 action，不看 observation
        steps = [_step("execute_bash", BASH) for _ in range(3)]
        turns = [
            TurnRecord(0, 2, "step", 0),
            TurnRecord(2, 4, "step", 1),
            TurnRecord(4, 6, "step", 2),
        ]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def test_different_signatures_counted_independently(self):
        steps = [
            _step("execute_bash", BASH),
            _step("execute_bash", {"command": "pwd"}),
            _step("execute_bash", BASH),
        ]
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0] * 6

    def test_editor_signature_includes_old_str(self):
        editor_b = {**EDITOR, "old_str": "y"}
        steps = [_step("str_replace_editor", EDITOR) for _ in range(2)]
        steps.append(_step("str_replace_editor", editor_b))
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0] * 6

    def test_editor_old_str_truncated_at_200(self):
        # old_str 只取前 200 字符：第 201 位起不同仍视为同一签名
        args_a = {**EDITOR, "old_str": "a" * 200 + "X"}
        args_b = {**EDITOR, "old_str": "a" * 200 + "Y"}
        assert action_signature(Action(tool_name="str_replace_editor", arguments=args_a)) == action_signature(
            Action(tool_name="str_replace_editor", arguments=args_b)
        )
        steps = [
            _step("str_replace_editor", args_a),
            _step("str_replace_editor", args_b),
            _step("str_replace_editor", args_a),
        ]
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def test_editor_duplicate_masked_from_third(self):
        steps = [_step("str_replace_editor", EDITOR) for _ in range(3)]
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def test_finish_not_counted(self):
        steps = [_step("execute_bash", BASH) for _ in range(2)]
        steps.append(_step("finish", {}))
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        alpha = _alpha(turns, steps, 6, ALL_RULES)
        assert alpha == [1.0] * 6

    def test_empty_rules_all_ones(self):
        turns = [TurnRecord(0, 2, "invalid_call", None)]
        assert _alpha(turns, [], 4, []) == [1.0] * 4

    def test_step_kind_without_step_index_not_counted(self):
        turns = [TurnRecord(0, 2, "step", None)]
        assert _alpha(turns, [], 4, ALL_RULES) == [1.0] * 4

    def test_token_end_clamped_to_n_tokens(self):
        turns = [TurnRecord(2, 100, "invalid_call", None)]
        assert _alpha(turns, [], 4, ALL_RULES) == [1.0, 1.0, 0.0, 0.0]

    def test_negative_token_start_raises(self):
        with pytest.raises(ValueError, match="invalid turn token range"):
            build_alpha([TurnRecord(-1, 2, "step", None)], [], 4, ALL_RULES)


class TestRegistry:
    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            resolve_rules(["bogus"])

    def test_registry_contents(self):
        assert set(RULE_REGISTRY) == {"invalid_call", "duplicate_action"}


from siete_rl.trainer import (  # noqa: E402
    _PARSE_ERROR_SENTINEL,
    _PLAIN_MESSAGE_SENTINEL,
    _classify_turn,
    _record_turn,
)


class TestConfigWiring:
    def test_yaml_generation_process_mask_rules(self):
        from siete_rl.config import load_config

        config, _, _ = load_config("configs/grpo_swegym_openhands_7b_lora.yaml")
        assert config.generation.process_mask_rules == ["invalid_call", "duplicate_action"]

    def test_generation_config_default_off(self):
        from siete_rl.config import GenerationConfig

        generation = GenerationConfig(
            max_completion_length=128,
            context_safety_margin=0,
            use_liger_kernel=True,
            max_tool_calling_iterations=4,
            max_consecutive_protocol_errors=3,
            temperature=1.0,
            top_p=1.0,
            top_k=20,
            repetition_penalty=1.1,
        )
        assert generation.process_mask_rules == []


class _FakeEnv:
    def __init__(self):
        self.turn_records: list[TurnRecord] = []


class TestClassifyTurn:
    def test_parse_error_sentinel_is_invalid_call(self):
        assert _classify_turn([_PARSE_ERROR_SENTINEL]) == "invalid_call"

    def test_plain_message_sentinel(self):
        assert _classify_turn([_PLAIN_MESSAGE_SENTINEL]) == "plain_message"

    def test_real_tool_call_is_step(self):
        calls = [{"type": "function", "function": {"name": "execute_bash", "arguments": {}}}]
        assert _classify_turn(calls) == "step"


class TestRecordTurn:
    def test_appends_record(self):
        env = _FakeEnv()
        _record_turn(env, 0, 5, "step", None)
        assert env.turn_records == [TurnRecord(0, 5, "step", None)]

    def test_appends_multiple_in_order(self):
        env = _FakeEnv()
        _record_turn(env, 0, 5, "step", 0)
        _record_turn(env, 5, 9, "invalid_call", None)
        assert env.turn_records == [
            TurnRecord(0, 5, "step", 0),
            TurnRecord(5, 9, "invalid_call", None),
        ]

    @pytest.mark.parametrize("start,end", [(5, 5), (6, 5), (0, 0)])
    def test_empty_range_skipped(self, start: int, end: int):
        env = _FakeEnv()
        _record_turn(env, start, end, "step", None)
        assert env.turn_records == []


from types import SimpleNamespace  # noqa: E402

from siete_rl.trainer import assemble_token_weights  # noqa: E402


def _mask_env(turn_records=(), steps=(), termination="submitted"):
    """假 env：只带 assemble_token_weights 读取的三个属性。"""
    return SimpleNamespace(
        turn_records=list(turn_records),
        _steps=list(steps),
        trajectory=SimpleNamespace(termination=termination),
    )


class TestAssembleTokenWeights:
    def test_normalization_preserves_base_mass(self):
        env = _mask_env([TurnRecord(2, 4, "invalid_call", None)])
        weights, stats = assemble_token_weights(
            env, base_mask=[1.0] * 6, n_tokens=6, rules=[InvalidCallMask()]
        )
        # c = Σbase/Σ(base×α) = 6/4 = 1.5，逐点缩放全部未 mask token，归一保持 Σweights == Σbase
        assert weights == [1.5, 1.5, 0.0, 0.0, 1.5, 1.5]
        assert stats["masked_turns"] == 1
        assert stats["masked_frac"] == pytest.approx(2 / 6)

    @pytest.mark.parametrize("termination", ["infra_error", "context_overlong"])
    def test_governance_termination_all_zero(self, termination: str):
        env = _mask_env(termination=termination)
        weights, _ = assemble_token_weights(
            env, base_mask=[1.0] * 6, n_tokens=6, rules=[InvalidCallMask()]
        )
        assert weights == [0.0] * 6

    @pytest.mark.parametrize("termination", ["iteration_cap", "submitted"])
    def test_other_terminations_unaffected(self, termination: str):
        env = _mask_env(termination=termination)
        weights, stats = assemble_token_weights(
            env, base_mask=[1.0] * 6, n_tokens=6, rules=[InvalidCallMask()]
        )
        assert weights == [1.0] * 6
        assert stats["masked_frac"] == 0.0

    def test_all_zero_base_mask_no_division_by_zero(self):
        env = _mask_env([TurnRecord(0, 2, "invalid_call", None)])
        weights, stats = assemble_token_weights(
            env, base_mask=[0.0] * 6, n_tokens=6, rules=[InvalidCallMask()]
        )
        assert weights == [0.0] * 6
        assert stats["masked_frac"] == 0.0

    def test_fully_masked_trajectory_all_zero(self):
        env = _mask_env([TurnRecord(0, 2, "invalid_call", None)])
        weights, stats = assemble_token_weights(
            env, base_mask=[1.0, 1.0], n_tokens=2, rules=[InvalidCallMask()]
        )
        assert weights == [0.0, 0.0]
        assert stats["masked_turns"] == 1
        assert stats["masked_frac"] == 1.0

    def test_masked_turns_counts_each_hit_turn(self):
        env = _mask_env(
            [
                TurnRecord(0, 2, "invalid_call", None),
                TurnRecord(2, 4, "step", None),
                TurnRecord(4, 6, "invalid_call", None),
            ]
        )
        _, stats = assemble_token_weights(
            env, base_mask=[1.0] * 6, n_tokens=6, rules=[InvalidCallMask()]
        )
        assert stats["masked_turns"] == 2

    def test_masked_turns_with_duplicate_action_rule(self):
        steps = [_step("execute_bash", BASH) for _ in range(3)]
        turns = [TurnRecord(i * 2, i * 2 + 2, "step", i) for i in range(3)]
        env = _mask_env(turns, steps)
        weights, stats = assemble_token_weights(
            env, base_mask=[1.0] * 6, n_tokens=6, rules=ALL_RULES
        )
        # 第 3 次同签名动作命中，c = 6/4
        assert weights == [1.5, 1.5, 1.5, 1.5, 0.0, 0.0]
        assert stats["masked_turns"] == 1


from collections import defaultdict  # noqa: E402

import torch  # noqa: E402
from trl import GRPOTrainer  # noqa: E402

from siete_rl.trainer import SWEGRPOTrainer  # noqa: E402


class _FakeTrainer(SWEGRPOTrainer):
    """跳过 GRPOTrainer 重型初始化，覆写所需属性由测试手动设置。"""

    def __init__(self) -> None:
        pass


def _bare_trainer(rules, environments):
    trainer = _FakeTrainer()
    trainer._process_mask_rules = rules
    if environments is not None:
        trainer.environments = environments
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.model = SimpleNamespace(training=True)
    return trainer


class TestGenerateAndScoreCompletionsOverride:
    def test_no_rules_returns_output_untouched(self, monkeypatch):
        output = {"completion_mask": torch.ones(1, 2, dtype=torch.long)}
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: output)
        # environments 故意不设置：无规则时覆写不应读取它
        trainer = _bare_trainer([], None)
        assert trainer._generate_and_score_completions([]) is output

    def test_misaligned_environments_raise(self, monkeypatch):
        output = {
            "completion_mask": torch.ones(2, 3, dtype=torch.long),
            "tool_mask": torch.ones(2, 3, dtype=torch.long),
        }
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: output)
        trainer = _bare_trainer([InvalidCallMask()], [_mask_env()])
        with pytest.raises(RuntimeError, match="aligned environments"):
            trainer._generate_and_score_completions([])

    def test_missing_tool_mask_raises(self, monkeypatch):
        output = {"completion_mask": torch.ones(1, 3, dtype=torch.long)}
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: output)
        trainer = _bare_trainer([InvalidCallMask()], [_mask_env()])
        with pytest.raises(RuntimeError, match="tool_mask"):
            trainer._generate_and_score_completions([])

    def test_token_weights_injected_and_metrics_appended(self, monkeypatch):
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
        tool_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]])
        output = {"completion_mask": completion_mask, "tool_mask": tool_mask}
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: output)
        rules = [InvalidCallMask()]
        envs = [
            _mask_env([TurnRecord(0, 2, "invalid_call", None)]),
            _mask_env(termination="infra_error"),
            # 规则全 mask 但非 governance 终止：不计入 governance_masked
            _mask_env([TurnRecord(0, 2, "invalid_call", None)]),
        ]
        trainer = _bare_trainer(rules, envs)
        result = trainer._generate_and_score_completions([])

        expected0, stats0 = assemble_token_weights(envs[0], base_mask=[1.0, 1.0, 1.0], n_tokens=3, rules=rules)
        expected1, stats1 = assemble_token_weights(envs[1], base_mask=[1.0, 0.0, 1.0, 1.0], n_tokens=4, rules=rules)
        expected2, stats2 = assemble_token_weights(envs[2], base_mask=[1.0, 1.0], n_tokens=2, rules=rules)
        assert result["token_weights"].tolist() == [expected0 + [0.0], expected1, expected2 + [0.0, 0.0]]

        frac_sum = stats0["masked_frac"] + stats1["masked_frac"] + stats2["masked_frac"]
        metrics = trainer._metrics["train"]
        assert metrics["process_mask/masked_token_frac"] == [pytest.approx(frac_sum / 3)]
        assert metrics["process_mask/masked_turns"] == [2.0]
        assert metrics["process_mask/governance_masked"] == [1.0]


class TestComputeLigerLossOverride:
    """process mask 开启时以 token_weights 替换 loss mask（经 tool_mask 通道传入父类）。"""

    def _capture_parent(self, monkeypatch):
        captured = {}

        def fake_compute_liger_loss(self, unwrapped_model, inputs):
            captured["inputs"] = inputs
            return "loss"

        monkeypatch.setattr(GRPOTrainer, "compute_liger_loss", fake_compute_liger_loss)
        return captured

    def test_rules_and_token_weights_replace_tool_mask(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer([InvalidCallMask()], None)
        token_weights = torch.ones(1, 3)
        inputs = {"completion_mask": torch.ones(1, 3), "token_weights": token_weights}
        assert trainer.compute_liger_loss(None, inputs) == "loss"
        # 同一对象传入；token_weights ⊆ completion_mask 支撑集使父类 loss_mask ≡ token_weights
        assert captured["inputs"]["tool_mask"] is token_weights
        # 原 dict 不被修改
        assert "tool_mask" not in inputs

    def test_no_rules_passes_tool_mask_through(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer([], None)
        tool_mask = torch.ones(1, 3)
        inputs = {"tool_mask": tool_mask, "token_weights": torch.zeros(1, 3)}
        trainer.compute_liger_loss(None, inputs)
        assert captured["inputs"]["tool_mask"] is tool_mask

    def test_rules_without_token_weights_passthrough(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer([InvalidCallMask()], None)
        inputs = {"completion_mask": torch.ones(1, 3)}
        trainer.compute_liger_loss(None, inputs)
        assert "tool_mask" not in captured["inputs"]


import json  # noqa: E402
from pathlib import Path  # noqa: E402

from siete_rl.config import load_config  # noqa: E402
from siete_rl.recording import STEP_METRIC_KEYS, RunRecorder  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/grpo_swegym_openhands_7b_lora.yaml"

PROCESS_MASK_STEP_METRICS = {
    "process_mask_masked_token_frac": "process_mask/masked_token_frac",
    "process_mask_masked_turns": "process_mask/masked_turns",
    "process_mask_governance_masked": "process_mask/governance_masked",
}


def _recorder(tmp_path: Path, run_id: str) -> RunRecorder:
    config, _, _ = load_config(CONFIG_PATH)
    output = config.output.model_copy(
        update={"output_root": (tmp_path / "outputs").as_posix(), "run_id": None}
    )
    config = config.model_copy(update={"output": output})
    return RunRecorder(config=config, seed=1, run_id=run_id)


class TestProcessMaskStepMetricKeys:
    def test_step_metric_keys_mapping(self):
        for field, source in PROCESS_MASK_STEP_METRICS.items():
            assert STEP_METRIC_KEYS.get(field) == source

    def test_metrics_row_contains_process_mask_fields(self, tmp_path: Path):
        recorder = _recorder(tmp_path, "pm-run")
        assert recorder.record_metrics(
            step=1,
            logs={
                "process_mask/masked_token_frac": 0.25,
                "process_mask/masked_turns": 3.0,
                "process_mask/governance_masked": 1.0,
            },
        )
        row = json.loads(recorder.metrics_path.read_text(encoding="utf-8").strip())
        assert row["process_mask_masked_token_frac"] == 0.25
        assert row["process_mask_masked_turns"] == 3.0
        assert row["process_mask_governance_masked"] == 1.0

    def test_metrics_row_fields_none_when_logs_absent(self, tmp_path: Path):
        recorder = _recorder(tmp_path, "pm-run")
        assert recorder.record_metrics(step=1, logs={"loss": 0.5})
        row = json.loads(recorder.metrics_path.read_text(encoding="utf-8").strip())
        assert row["process_mask_masked_token_frac"] is None
        assert row["process_mask_masked_turns"] is None
        assert row["process_mask_governance_masked"] is None
