from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from siete_rl.models import Action, Observation, Step
from siete_rl.process_mask import (
    ProcessMaskStats,
    TurnRecord,
    _action_key,
    build_process_token_weights,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/grpo_swegym_openhands_7b_lora.yaml"
)


class TestActionKey:
    def test_argument_key_order_does_not_change_key(self):
        left = Action(
            tool_name="str_replace_editor",
            arguments={"command": "view", "path": "/repo/a.py", "view_range": [1, 40]},
        )
        right = Action(
            tool_name="str_replace_editor",
            arguments={"view_range": [1, 40], "path": "/repo/a.py", "command": "view"},
        )
        assert _action_key(left) == _action_key(right)

    @pytest.mark.parametrize(
        ("field", "left", "right"),
        [
            ("view_range", [1, 40], [41, 80]),
            ("new_str", "x = 1", "x = 2"),
            ("insert_line", 10, 11),
            ("file_text", "a\n", "b\n"),
        ],
    )
    def test_every_editor_argument_participates(self, field, left, right):
        common = {"command": "view", "path": "/repo/a.py"}
        action_a = Action(tool_name="str_replace_editor", arguments={**common, field: left})
        action_b = Action(tool_name="str_replace_editor", arguments={**common, field: right})
        assert _action_key(action_a) != _action_key(action_b)

    def test_finish_has_no_repeat_key(self):
        assert _action_key(Action(tool_name="finish", arguments={})) is None


def _turns(kinds: list[str]) -> list[TurnRecord]:
    return [
        TurnRecord(token_start=2 * i, token_end=2 * i + 2, kind=kind, step_index=i if kind == "step" else None)
        for i, kind in enumerate(kinds)
    ]


def _bash_step(index: int, command: str) -> Step:
    return Step(
        index=index,
        action=Action(tool_name="execute_bash", arguments={"command": command}),
        observation=Observation(text="ok", exit_code=0),
    )


class TestBuildProcessTokenWeights:
    def test_positive_advantage_masks_third_consecutive_action(self):
        turns = _turns(["step", "step", "step"])
        steps = [_bash_step(i, "pytest -q") for i in range(3)]
        weights, stats = build_process_token_weights(
            turns=turns,
            steps=steps,
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0] * 6,
        )
        assert weights == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        assert stats.candidate_turns == 1
        assert stats.applied_turns == 1
        assert stats.retained_negative_turns == 0
        assert stats.masked_token_frac == pytest.approx(2 / 6)
        assert not stats.governance_masked

    def test_intervening_action_resets_streak(self):
        commands = ["pytest -q", "pytest -q", "sed -n '1,40p' a.py", "pytest -q"]
        turns = _turns(["step"] * 4)
        steps = [_bash_step(i, command) for i, command in enumerate(commands)]
        weights, stats = build_process_token_weights(
            turns=turns,
            steps=steps,
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0] * 8,
        )
        assert weights == [1.0] * 8
        assert stats.candidate_turns == 0

    def test_invalid_call_breaks_repeat_streak_and_is_candidate(self):
        turns = [
            TurnRecord(0, 2, "step", 0),
            TurnRecord(2, 4, "step", 1),
            TurnRecord(4, 6, "invalid_call", None),
            TurnRecord(6, 8, "step", 2),
        ]
        steps = [_bash_step(i, "pytest -q") for i in range(3)]
        weights, stats = build_process_token_weights(
            turns=turns,
            steps=steps,
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0] * 8,
        )
        assert weights == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        assert stats.candidate_turns == 1

    @pytest.mark.parametrize("advantage", [-0.5, 0.0])
    def test_nonpositive_advantage_retains_candidate_turns(self, advantage):
        turns = _turns(["step", "step", "step"])
        steps = [_bash_step(i, "pytest -q") for i in range(3)]
        weights, stats = build_process_token_weights(
            turns=turns,
            steps=steps,
            termination="submitted",
            advantage=advantage,
            base_mask=[1.0] * 6,
        )
        assert weights == [1.0] * 6
        assert stats.applied_turns == 0
        assert stats.retained_negative_turns == (1 if advantage < 0 else 0)

    @pytest.mark.parametrize("termination", ["infra_error", "context_overlong"])
    def test_governance_always_masks_whole_trajectory(self, termination):
        weights, stats = build_process_token_weights(
            turns=[],
            steps=[],
            termination=termination,
            advantage=-0.5,
            base_mask=[1.0, 0.0, 1.0],
        )
        assert weights == [0.0, 0.0, 0.0]
        assert stats.masked_token_frac == 1.0
        assert stats.governance_masked

    def test_output_never_contains_weight_above_one(self):
        turns = [TurnRecord(0, 2, "invalid_call", None)]
        weights, _ = build_process_token_weights(
            turns=turns,
            steps=[],
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0, 1.0, 1.0],
        )
        assert set(weights) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "turn",
    [
        TurnRecord(-1, 1, "invalid_call", None),
        TurnRecord(2, 1, "invalid_call", None),
        TurnRecord(0, 4, "invalid_call", None),
        TurnRecord(0, 1, "step", None),
        TurnRecord(0, 1, "plain_message", 0),
    ],
)
def test_invalid_turn_facts_fail_loud(turn):
    with pytest.raises(ValueError):
        build_process_token_weights(
            turns=[turn],
            steps=[],
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0, 1.0],
        )


def test_nonbinary_base_mask_fails_loud():
    with pytest.raises(ValueError, match="base_mask must be binary"):
        build_process_token_weights(
            turns=[],
            steps=[],
            termination="submitted",
            advantage=0.5,
            base_mask=[1.0, 0.5],
        )


from siete_rl.trainer import (  # noqa: E402
    _PARSE_ERROR_SENTINEL,
    _PLAIN_MESSAGE_SENTINEL,
    _classify_turn,
    _record_turn,
)


class TestConfigWiring:
    def test_yaml_requires_boolean_process_mask(self):
        from siete_rl.config import load_config

        config, _, _ = load_config(CONFIG_PATH)
        assert config.generation.use_process_mask is True
        assert not hasattr(config.generation, "process_mask_rules")

    def test_old_process_mask_rules_key_is_rejected(self, tmp_path):
        from siete_rl.config import load_config

        raw = yaml.safe_load(CONFIG_PATH.read_text())
        raw["generation"].pop("use_process_mask")
        raw["generation"]["process_mask_rules"] = ["invalid_call", "duplicate_action"]
        path = tmp_path / "old.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValidationError):
            load_config(path)

    @pytest.mark.parametrize("loss_type", ["bnpo", "dapo", "dr_grpo"])
    def test_non_grpo_loss_type_is_rejected(self, tmp_path, loss_type):
        from siete_rl.config import load_config

        raw = yaml.safe_load(CONFIG_PATH.read_text())
        raw["grpo"]["loss_type"] = loss_type
        path = tmp_path / "bad-loss.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValidationError):
            load_config(path)


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

def _mask_env(turn_records=(), steps=(), termination="submitted"):
    """假 env：只带 process-mask 接线读取的轨迹事实。"""
    return SimpleNamespace(
        turn_records=list(turn_records),
        _steps=list(steps),
        trajectory=SimpleNamespace(termination=termination),
    )


from collections import defaultdict  # noqa: E402

import torch  # noqa: E402
from trl import GRPOTrainer  # noqa: E402

from siete_rl.trainer import SWEGRPOTrainer  # noqa: E402


class _FakeTrainer(SWEGRPOTrainer):
    """跳过 GRPOTrainer 重型初始化，覆写所需属性由测试手动设置。"""

    def __init__(self) -> None:
        pass


def _bare_trainer(*, use_process_mask: bool, environments=None):
    trainer = _FakeTrainer()
    trainer._use_process_mask = use_process_mask
    if environments is not None:
        trainer.environments = environments
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.model = SimpleNamespace(training=True)
    return trainer


def test_process_mask_requires_liger(monkeypatch):
    monkeypatch.setattr(
        GRPOTrainer,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "use_liger_kernel", False),
    )
    with pytest.raises(ValueError, match="requires use_liger_kernel=true"):
        SWEGRPOTrainer(max_consecutive_protocol_errors=5, use_process_mask=True)


class TestGenerateAndScoreCompletionsOverride:
    def test_disabled_process_mask_returns_output_untouched(self, monkeypatch):
        output = {"completion_mask": torch.ones(1, 2, dtype=torch.long)}
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(use_process_mask=False)
        assert trainer._generate_and_score_completions([]) is output

    def test_misaligned_environments_raise(self, monkeypatch):
        output = {
            "completion_mask": torch.ones(2, 3, dtype=torch.long),
            "tool_mask": torch.ones(2, 3, dtype=torch.long),
            "advantages": torch.ones(2),
        }
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(use_process_mask=True, environments=[_mask_env()])
        with pytest.raises(RuntimeError, match="process mask requires"):
            trainer._generate_and_score_completions([])

    def test_trainer_passes_each_advantage_to_process_module(self, monkeypatch):
        completion_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
        tool_mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
        output = {
            "completion_mask": completion_mask,
            "tool_mask": tool_mask,
            "advantages": torch.tensor([0.5, -0.25]),
        }
        envs = [_mask_env(), _mask_env()]
        calls = []

        def fake_build(**kwargs):
            calls.append(kwargs)
            return list(kwargs["base_mask"]), ProcessMaskStats(0, 0, 0, 0.0, False)

        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        monkeypatch.setattr("siete_rl.trainer.build_process_token_weights", fake_build)
        trainer = _bare_trainer(use_process_mask=True, environments=envs)
        result = trainer._generate_and_score_completions([])

        assert result["token_weights"].tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        assert [call["advantage"] for call in calls] == [0.5, -0.25]
        assert calls[0]["termination"] == envs[0].trajectory.termination

    def test_truncated_governance_row_uses_full_padded_width(self, monkeypatch):
        output = {
            "completion_mask": torch.zeros(1, 4, dtype=torch.long),
            "tool_mask": torch.zeros(1, 4, dtype=torch.long),
            "advantages": torch.tensor([0.5]),
        }
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(
            use_process_mask=True,
            environments=[
                _mask_env(
                    turn_records=[TurnRecord(1, 3, "invalid_call", None)],
                    termination="context_overlong",
                )
            ],
        )

        result = trainer._generate_and_score_completions([])

        assert result["token_weights"].tolist() == [[0.0, 0.0, 0.0, 0.0]]
        assert trainer._metrics["train"]["process_mask/governance_masked"] == [1.0]

    def test_process_mask_stats_are_aggregated_without_recomputing_policy(self):
        trainer = _bare_trainer(use_process_mask=True)
        trainer._record_process_mask_metrics(
            [
                ProcessMaskStats(3, 2, 0, 0.25, False),
                ProcessMaskStats(4, 0, 4, 0.00, False),
                ProcessMaskStats(1, 0, 0, 1.00, True),
            ]
        )
        metrics = trainer._metrics["train"]
        assert metrics["process_mask/candidate_turns"] == [8.0]
        assert metrics["process_mask/applied_turns"] == [2.0]
        assert metrics["process_mask/retained_negative_turns"] == [4.0]
        assert metrics["process_mask/masked_token_frac"] == [pytest.approx(1.25 / 3)]
        assert metrics["process_mask/governance_masked"] == [1.0]

    @pytest.mark.parametrize("missing", ["tool_mask", "advantages"])
    def test_enabled_process_mask_requires_parent_fields(self, monkeypatch, missing):
        output = {
            "completion_mask": torch.ones(1, 3),
            "tool_mask": torch.ones(1, 3),
            "advantages": torch.tensor([0.5]),
        }
        output.pop(missing)
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: output)
        trainer = _bare_trainer(use_process_mask=True, environments=[_mask_env()])
        with pytest.raises(RuntimeError, match="process mask requires"):
            trainer._generate_and_score_completions([])

    @pytest.mark.parametrize(
        ("tool_mask", "advantages"),
        [
            (torch.ones(1, 3), torch.ones(2)),
            (torch.ones(2, 3), torch.ones(2, 1)),
        ],
        ids=["broadcast-tool-mask", "column-advantages"],
    )
    def test_enabled_process_mask_requires_exact_parent_shapes(
        self, monkeypatch, tool_mask, advantages
    ):
        output = {
            "completion_mask": torch.ones(2, 3),
            "tool_mask": tool_mask,
            "advantages": advantages,
        }
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(
            use_process_mask=True,
            environments=[_mask_env(), _mask_env()],
        )

        with pytest.raises(RuntimeError) as exc_info:
            trainer._generate_and_score_completions([])

        assert str(exc_info.value) == (
            "process mask requires aligned environments, tool_mask, and advantages"
        )


class TestComputeLigerLossOverride:
    """process mask 开启时以 token_weights 替换 loss mask（经 tool_mask 通道传入父类）。"""

    def _capture_parent(self, monkeypatch):
        captured = {}

        def fake_compute_liger_loss(self, unwrapped_model, inputs):
            captured["inputs"] = inputs
            return "loss"

        monkeypatch.setattr(GRPOTrainer, "compute_liger_loss", fake_compute_liger_loss)
        return captured

    def test_enabled_mask_and_token_weights_replace_tool_mask(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer(use_process_mask=True)
        token_weights = torch.ones(1, 3)
        inputs = {"completion_mask": torch.ones(1, 3), "token_weights": token_weights}
        assert trainer.compute_liger_loss(None, inputs) == "loss"
        # 同一对象传入；token_weights ⊆ completion_mask 支撑集使父类 loss_mask ≡ token_weights
        assert captured["inputs"]["tool_mask"] is token_weights
        # 原 dict 不被修改
        assert "tool_mask" not in inputs

    def test_disabled_mask_passes_tool_mask_through(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer(use_process_mask=False)
        tool_mask = torch.ones(1, 3)
        inputs = {"tool_mask": tool_mask, "token_weights": torch.zeros(1, 3)}
        trainer.compute_liger_loss(None, inputs)
        assert captured["inputs"]["tool_mask"] is tool_mask

    def test_enabled_mask_without_token_weights_passthrough(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer(use_process_mask=True)
        inputs = {"completion_mask": torch.ones(1, 3)}
        trainer.compute_liger_loss(None, inputs)
        assert "tool_mask" not in captured["inputs"]


import json  # noqa: E402

from siete_rl.config import load_config  # noqa: E402
from siete_rl.recording import STEP_METRIC_KEYS, RunRecorder  # noqa: E402


PROCESS_MASK_STEP_METRICS = {
    "process_mask_candidate_turns": "process_mask/candidate_turns",
    "process_mask_applied_turns": "process_mask/applied_turns",
    "process_mask_retained_negative_turns": "process_mask/retained_negative_turns",
    "process_mask_masked_token_frac": "process_mask/masked_token_frac",
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
                "process_mask/candidate_turns": 8.0,
                "process_mask/applied_turns": 2.0,
                "process_mask/retained_negative_turns": 4.0,
                "process_mask/masked_token_frac": 0.25,
                "process_mask/governance_masked": 1.0,
            },
        )
        row = json.loads(recorder.metrics_path.read_text(encoding="utf-8").strip())
        assert row["process_mask_candidate_turns"] == 8.0
        assert row["process_mask_applied_turns"] == 2.0
        assert row["process_mask_retained_negative_turns"] == 4.0
        assert row["process_mask_masked_token_frac"] == 0.25
        assert row["process_mask_governance_masked"] == 1.0

    def test_metrics_row_fields_none_when_logs_absent(self, tmp_path: Path):
        recorder = _recorder(tmp_path, "pm-run")
        assert recorder.record_metrics(step=1, logs={"loss": 0.5})
        row = json.loads(recorder.metrics_path.read_text(encoding="utf-8").strip())
        assert row["process_mask_candidate_turns"] is None
        assert row["process_mask_applied_turns"] is None
        assert row["process_mask_retained_negative_turns"] is None
        assert row["process_mask_masked_token_frac"] is None
        assert row["process_mask_governance_masked"] is None
