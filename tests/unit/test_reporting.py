from __future__ import annotations

import json
from pathlib import Path

import pytest

from siete_rl.reporting import (
    load_group_rows,
    load_metric_rows,
    render_training_summary,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_summary() -> dict[str, object]:
    return {
        "run_id": "plot-run",
        "status": "completed",
        "results": {
            "reward": {
                "successes": 2,
                "attempts": 8,
                "nondegenerate_groups": 2,
            }
        },
        "train": {"steps_completed": 2, "steps_target": 2},
    }


def test_training_summary_is_a_real_png(tmp_path: Path) -> None:
    write_rows(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "rollouts_cumulative": 4,
                "reward_mean_group": 0.25,
                "reward_mean_ema": 0.25,
                "train_pass_rate_cumulative": 0.25,
                "group_degenerate": False,
                "nondegenerate_group_rate_cumulative": 1.0,
                "reward_std_group_population": 0.433,
                "grad_norm": 1.2,
                "kl": 0.02,
                "importance_sampling_ratio_mean": 1.01,
            },
            {
                "step": 2,
                "rollouts_cumulative": 8,
                "reward_mean_group": 0.25,
                "reward_mean_ema": 0.25,
                "train_pass_rate_cumulative": 0.25,
                "group_degenerate": False,
                "nondegenerate_group_rate_cumulative": 1.0,
                "reward_std_group_population": 0.433,
                "grad_norm": 0.8,
                "kl": 0.01,
                "importance_sampling_ratio_mean": 0.99,
            },
        ],
    )

    result = render_training_summary(tmp_path, run_summary())

    assert result == tmp_path / "training_summary.png"
    assert result.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.stat().st_size > 10_000


def test_training_summary_is_skipped_without_steps(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").touch()
    assert render_training_summary(tmp_path, run_summary()) is None


def test_group_series_keeps_every_group_between_optimizer_steps(
    tmp_path: Path,
) -> None:
    rewards = ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1])
    for index, values in enumerate(rewards):
        group_dir = (
            tmp_path
            / "rollouts"
            / f"batch-{index:04d}"
            / "group-0000"
        )
        group_dir.mkdir(parents=True)
        (group_dir / "group.json").write_text(
            json.dumps({"state": "completed", "rewards": values}),
            encoding="utf-8",
        )

    rows = load_group_rows(tmp_path)

    assert len(rows) == 3
    assert [row["rollouts_cumulative"] for row in rows] == [4, 8, 12]
    assert [row["reward_mean_group"] for row in rows] == [0.0, 0.25, 1.0]
    assert [row["group_degenerate"] for row in rows] == [True, False, True]
    assert rows[-1]["train_pass_rate_cumulative"] == pytest.approx(5 / 12)
    assert rows[-1]["reward_mean_ema"] == pytest.approx(0.24)
    assert rows[-1]["nondegenerate_group_rate_cumulative"] == pytest.approx(1 / 3)


def test_metric_reader_only_tolerates_a_truncated_last_line(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 1}\n{"step":', encoding="utf-8")
    assert load_metric_rows(metrics) == [{"step": 1}]

    metrics.write_text('{"step":\n{"step": 2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_metric_rows(metrics)
