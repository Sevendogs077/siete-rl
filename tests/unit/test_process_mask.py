from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer

from siete_rl.models import Action, Observation, Settlement, Step
from siete_rl.process_mask import (
    CreditMaskStats,
    ProcessMaskStats,
    TurnRecord,
    _action_key,
    build_credit_token_weights,
    build_process_token_weights,
)
from siete_rl.trainer import SWEGRPOTrainer


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

    def test_finish_has_no_repeat_key(self):
        assert _action_key(Action(tool_name="finish", arguments={})) is None


def _turns(kinds: list[str]) -> list[TurnRecord]:
    return [
        TurnRecord(
            token_start=2 * i,
            token_end=2 * i + 2,
            kind=kind,
            step_index=i if kind == "step" else None,
        )
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

    def test_context_overlong_does_not_change_process_mask_policy(self):
        weights, stats = build_process_token_weights(
            turns=[],
            steps=[],
            termination="context_overlong",
            advantage=-0.5,
            base_mask=[1.0, 0.0, 1.0],
        )
        assert weights == [1.0, 0.0, 1.0]
        assert stats.masked_token_frac == 0.0


class TestBuildCreditTokenWeights:
    def test_infra_settlement_masks_entire_row(self):
        weights, stats = build_credit_token_weights(
            turns=[],
            termination="submitted",
            settlement=Settlement(status="infra_error"),
            base_mask=[1.0, 0.0, 1.0],
        )
        assert weights == [0.0, 0.0, 0.0]
        assert stats == CreditMaskStats(
            infra_rows=1, truncated_turns=0, masked_token_frac=1.0
        )

    def test_only_final_truncated_turn_is_masked(self):
        turns = [
            TurnRecord(0, 2, "step", 0),
            TurnRecord(2, 5, "plain_message", None, truncated=True),
        ]
        weights, stats = build_credit_token_weights(
            turns=turns,
            termination="context_overlong",
            settlement=Settlement(status="unresolved"),
            base_mask=[1.0] * 5,
        )
        assert weights == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert stats.truncated_turns == 1
        assert stats.masked_token_frac == pytest.approx(3 / 5)

    def test_truncated_executed_step_keeps_step_identity(self):
        turn = TurnRecord(0, 2, "step", 0, truncated=True)
        weights, _ = build_credit_token_weights(
            turns=[turn],
            termination="context_overlong",
            settlement=Settlement(status="agent_error"),
            base_mask=[1.0, 1.0],
        )
        assert weights == [0.0, 0.0]


@pytest.mark.parametrize("advantage", [0.5, -0.5])
def test_final_pending_action_at_iteration_cap_preserves_base_mask(advantage):
    base_mask = [1.0, 0.0, 1.0]

    weights, stats = build_process_token_weights(
        turns=[TurnRecord(0, 2, "pending_action", None)],
        steps=[],
        termination="iteration_cap",
        advantage=advantage,
        base_mask=base_mask,
    )

    assert weights == base_mask
    assert stats.candidate_turns == 0
    assert stats.applied_turns == 0
    assert stats.retained_negative_turns == 0


def _mask_env(
    turn_records=(),
    steps=(),
    termination="submitted",
    verification=None,
    settlement="unresolved",
    reward=0.0,
):
    """假 env：只带 process-mask 接线读取的轨迹事实。"""
    return SimpleNamespace(
        turn_records=list(turn_records),
        _steps=list(steps),
        trajectory=SimpleNamespace(
            termination=termination,
            settlement=Settlement(status=settlement),
        ),
        verification=verification,
        _reward=reward,
    )

class _FakeTrainer(SWEGRPOTrainer):
    """跳过 GRPOTrainer 重型初始化，覆写所需属性由测试手动设置。"""

    def __init__(self) -> None:
        pass


def _bare_trainer(*, use_process_mask: bool, environments=None):
    trainer = _FakeTrainer()
    trainer._use_process_mask = use_process_mask
    trainer._extra_reference_rewards = ()
    if environments is not None:
        trainer.environments = environments
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.model = SimpleNamespace(training=True)
    return trainer


@pytest.mark.parametrize("use_process_mask", [False, True])
def test_credit_mask_requires_liger(monkeypatch, use_process_mask):
    monkeypatch.setattr(
        GRPOTrainer,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "use_liger_kernel", False),
    )
    with pytest.raises(ValueError, match="credit mask requires use_liger_kernel=true"):
        SWEGRPOTrainer(
            max_consecutive_protocol_errors=5,
            use_process_mask=use_process_mask,
        )


class TestGenerateAndScoreCompletionsOverride:
    def test_extra_reference_rewards_give_degenerate_groups_gradient(
        self, monkeypatch
    ):
        output = {
            "completion_mask": torch.ones(8, 1, dtype=torch.long),
            "tool_mask": torch.ones(8, 1, dtype=torch.long),
            "advantages": torch.zeros(8),
        }
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(
            use_process_mask=False,
            environments=[
                *[_mask_env(reward=0.0) for _ in range(4)],
                *[_mask_env(reward=1.0) for _ in range(4)],
            ],
        )
        trainer._extra_reference_rewards = (0.0, 1.0)
        trainer.num_generations = 4
        trainer.scale_rewards = "none"

        result = trainer._generate_and_score_completions([])

        assert result["advantages"].tolist() == pytest.approx(
            [-1 / 6] * 4 + [1 / 6] * 4
        )

    def test_disabled_process_mask_still_applies_credit_mask(self, monkeypatch):
        output = {
            "completion_mask": torch.ones(1, 2, dtype=torch.long),
            "tool_mask": torch.ones(1, 2, dtype=torch.long),
            "advantages": torch.ones(1),
        }
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )
        trainer = _bare_trainer(
            use_process_mask=False,
            environments=[_mask_env(settlement="infra_error")],
        )
        result = trainer._generate_and_score_completions([])
        assert result["token_weights"].tolist() == [[0.0, 0.0]]

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
            return list(kwargs["base_mask"]), ProcessMaskStats(0, 0, 0, 0.0)

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

    def test_truncated_turn_masks_only_its_token_interval(self, monkeypatch):
        output = {
            "completion_mask": torch.ones(1, 4, dtype=torch.long),
            "tool_mask": torch.ones(1, 4, dtype=torch.long),
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
                    turn_records=[
                        TurnRecord(1, 3, "plain_message", None, truncated=True)
                    ],
                    termination="context_overlong",
                )
            ],
        )

        result = trainer._generate_and_score_completions([])

        assert result["token_weights"].tolist() == [[1.0, 0.0, 0.0, 1.0]]
        assert trainer._metrics["train"]["credit_mask/truncated_turns"] == [1.0]

    def test_process_mask_stats_are_aggregated_without_recomputing_policy(self):
        trainer = _bare_trainer(use_process_mask=True)
        trainer._record_process_mask_metrics(
            [
                ProcessMaskStats(3, 2, 0, 0.25),
                ProcessMaskStats(4, 0, 4, 0.00),
                ProcessMaskStats(1, 0, 0, 1.00),
            ]
        )
        metrics = trainer._metrics["train"]
        assert metrics["process_mask/candidate_turns"] == [8.0]
        assert metrics["process_mask/applied_turns"] == [2.0]
        assert metrics["process_mask/retained_negative_turns"] == [4.0]
        assert metrics["process_mask/masked_token_frac"] == [pytest.approx(1.25 / 3)]

    def test_recovered_positive_metrics_measure_effective_token_support(
        self, monkeypatch
    ):
        output = {
            "completion_mask": torch.tensor(
                [[1, 1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.long
            ),
            "tool_mask": torch.tensor(
                [[1, 0, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long
            ),
            "advantages": torch.tensor([0.5, 0.5, 0.5, 0.5]),
        }
        resolved = SimpleNamespace(result="resolved")
        envs = [
            _mask_env(termination="iteration_cap", verification=resolved),
            _mask_env(termination="format_exhausted", verification=resolved),
            _mask_env(termination="submitted", verification=resolved),
            _mask_env(termination="context_overlong", verification=resolved),
        ]
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: output,
        )

        trainer = _bare_trainer(use_process_mask=True, environments=envs)
        trainer._generate_and_score_completions([])

        metrics = trainer._metrics["train"]
        assert metrics["settlement/recovered_positive_rows"] == [3.0]
        assert metrics["settlement/recovered_positive_active_tokens"] == [4.0]


class TestComputeLigerLossOverride:
    """always-on token_weights 经 tool_mask 通道替换父类 loss mask。"""

    def _capture_parent(self, monkeypatch):
        captured = {}

        def fake_compute_liger_loss(self, unwrapped_model, inputs):
            captured["inputs"] = inputs
            return "loss"

        monkeypatch.setattr(GRPOTrainer, "compute_liger_loss", fake_compute_liger_loss)
        return captured

    @pytest.mark.parametrize("use_process_mask", [False, True])
    def test_token_weights_always_replace_tool_mask(
        self, monkeypatch, use_process_mask
    ):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer(use_process_mask=use_process_mask)
        token_weights = torch.ones(1, 3)
        inputs = {"completion_mask": torch.ones(1, 3), "token_weights": token_weights}
        assert trainer.compute_liger_loss(None, inputs) == "loss"
        # 同一对象传入；token_weights ⊆ completion_mask 支撑集使父类 loss_mask ≡ token_weights
        assert captured["inputs"]["tool_mask"] is token_weights
        # 原 dict 不被修改
        assert "tool_mask" not in inputs

    def test_without_token_weights_passes_parent_inputs_through(self, monkeypatch):
        captured = self._capture_parent(monkeypatch)
        trainer = _bare_trainer(use_process_mask=False)
        inputs = {"completion_mask": torch.ones(1, 3)}
        trainer.compute_liger_loss(None, inputs)
        assert "tool_mask" not in captured["inputs"]

    def test_all_zero_token_weights_bypass_model_connected_parent(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("fully censored microbatch must bypass parent loss")

        monkeypatch.setattr(GRPOTrainer, "compute_liger_loss", boom)
        trainer = _bare_trainer(use_process_mask=False)
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        loss = trainer.compute_liger_loss(
            SimpleNamespace(lm_head=SimpleNamespace(weight=parameter, bias=None)),
            {
                "completion_mask": torch.ones(1, 2),
                "token_weights": torch.zeros(1, 2),
            },
        )

        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert parameter.grad is None


def _real_liger_case(token_weights):
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

    trainer = _bare_trainer(use_process_mask=False)
    trainer.beta = 0.0
    trainer.current_gradient_accumulation_steps = 1
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None),
        gather=lambda value: value,
    )
    trainer.liger_loss = LigerFusedLinearGRPOLoss(
        beta=0.0,
        compiled=False,
        use_ref_model=False,
        chunk_size=4,
        loss_type="grpo",
    )
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.5, -0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.5, -0.5]],
        ]
    )
    trainer._get_last_hidden_state = lambda *args, **kwargs: hidden
    model = SimpleNamespace(lm_head=torch.nn.Linear(2, 3, bias=False))
    with torch.no_grad():
        model.lm_head.weight.copy_(
            torch.tensor([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.4]])
        )
    inputs = {
        "prompt_ids": torch.zeros((4, 1), dtype=torch.long),
        "prompt_mask": torch.ones((4, 1), dtype=torch.long),
        "completion_ids": torch.tensor(
            [[0, 1], [1, 2], [0, 1], [1, 2]], dtype=torch.long
        ),
        "completion_mask": torch.ones((4, 2)),
        "token_weights": torch.tensor(token_weights, dtype=torch.float32),
        "advantages": torch.tensor([1.0, 0.25, 1.0, 0.25]),
    }
    loss = trainer.compute_liger_loss(model, inputs)
    loss.backward()
    return loss.detach(), model.lm_head.weight.grad.detach().clone()


def test_fixed_g_liger_loss_and_gradient_scale_with_censored_rows():
    complete_loss, complete_grad = _real_liger_case([[1, 1]] * 4)
    censored_loss, censored_grad = _real_liger_case(
        [[1, 1], [1, 1], [0, 0], [0, 0]]
    )

    assert censored_loss.item() == pytest.approx(complete_loss.item() / 2)
    assert torch.allclose(censored_grad, complete_grad / 2, atol=1e-6, rtol=1e-6)
