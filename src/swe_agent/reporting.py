"""训练指标读取与终止态摘要图生成。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


REWARD_EMA_ALPHA = 0.2


def load_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取 metrics.jsonl；只容忍最后一行因进程退出而不完整。"""

    metric_path = Path(path)
    lines = metric_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(row, dict):
            raise ValueError(f"{metric_path}:{index + 1}: metric row must be an object")
        rows.append(row)
    return rows


def load_group_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    """从完整 group 索引重建 reward 序列，避免每 step 丢掉中间 group。"""

    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    rewards_seen: list[float] = []
    rollouts_cumulative = 0
    nondegenerate_groups = 0
    reward_ema: float | None = None
    paths = sorted(
        root.glob("rollouts/batch-*/group-*/group.json"),
        key=lambda path: (
            int(path.parents[1].name.removeprefix("batch-")),
            int(path.parent.name.removeprefix("group-")),
        ),
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewards = payload.get("rewards")
        if payload.get("state") != "completed" or not isinstance(rewards, list):
            continue
        numeric_rewards = [_number(value) for value in rewards]
        if not numeric_rewards or any(value is None for value in numeric_rewards):
            continue
        group_rewards = [float(value) for value in numeric_rewards if value is not None]
        reward_mean = fmean(group_rewards)
        reward_std = pstdev(group_rewards) if len(group_rewards) > 1 else 0.0
        degenerate = math.isclose(reward_std, 0.0)
        rewards_seen.extend(group_rewards)
        rollouts_cumulative += len(group_rewards)
        nondegenerate_groups += int(not degenerate)
        reward_ema = (
            reward_mean
            if reward_ema is None
            else REWARD_EMA_ALPHA * reward_mean
            + (1.0 - REWARD_EMA_ALPHA) * reward_ema
        )
        group_count = len(rows) + 1
        rows.append(
            {
                "rollouts_cumulative": rollouts_cumulative,
                "reward_mean_group": reward_mean,
                "reward_mean_ema": reward_ema,
                # 累计平均奖励；二元奖励下等价于 resolved 通过率，layered 奖励下为部分得分的均值
                "train_pass_rate_cumulative": fmean(rewards_seen),
                "group_degenerate": degenerate,
                "nondegenerate_group_rate_cumulative": (
                    nondegenerate_groups / group_count
                ),
                "reward_std_group_population": reward_std,
            }
        )
    return rows


def render_training_summary(
    output_dir: str | Path,
    run: dict[str, Any],
    *,
    filename: str = "training_summary.png",
) -> Path | None:
    """生成固定三联图；没有 optimizer step 时返回 None。"""

    root = Path(output_dir)
    step_rows = load_metric_rows(root / "metrics.jsonl")
    if not step_rows:
        return None
    group_rows = load_group_rows(root) or step_rows

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    step_x = [
        _nan(_number(row.get("rollouts_cumulative"))) for row in step_rows
    ]
    group_x = [
        _nan(_number(row.get("rollouts_cumulative"))) for row in group_rows
    ]
    reward_group = [
        _nan(_number(row.get("reward_mean_group"))) for row in group_rows
    ]
    reward_ema = [
        _nan(_number(row.get("reward_mean_ema"))) for row in group_rows
    ]
    reward_cumulative = [
        _nan(_number(row.get("train_pass_rate_cumulative")))
        for row in group_rows
    ]
    nondegenerate = [
        math.nan
        if row.get("group_degenerate") is None
        else float(not bool(row["group_degenerate"]))
        for row in group_rows
    ]
    nondegenerate_cumulative = [
        _nan(_number(row.get("nondegenerate_group_rate_cumulative")))
        for row in group_rows
    ]
    reward_std = [
        _nan(_number(row.get("reward_std_group_population")))
        for row in group_rows
    ]
    grad_norm = [_nan(_number(row.get("grad_norm"))) for row in step_rows]
    kl = [_nan(_number(row.get("kl"))) for row in step_rows]
    is_ratio = [
        _nan(_number(row.get("importance_sampling_ratio_mean")))
        for row in step_rows
    ]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0f172a",
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    figure.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].scatter(
        group_x,
        reward_group,
        color="#93c5fd",
        edgecolor="#2563eb",
        linewidth=0.8,
        s=34,
        label="Group mean reward",
        zorder=3,
    )
    axes[0].plot(
        group_x,
        reward_ema,
        color="#2563eb",
        linewidth=2.2,
        label="EMA (alpha=0.2)",
    )
    axes[0].plot(
        group_x,
        reward_cumulative,
        color="#d97706",
        linewidth=1.8,
        linestyle="--",
        label="Cumulative mean reward",
    )
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("Train reward", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False, ncol=3)

    axes[1].step(
        group_x,
        nondegenerate,
        where="mid",
        color="#64748b",
        linewidth=1.3,
        label="Non-degenerate group (0/1)",
    )
    axes[1].plot(
        group_x,
        nondegenerate_cumulative,
        color="#2563eb",
        linewidth=2.2,
        label="Cumulative non-degenerate rate",
    )
    axes[1].plot(
        group_x,
        reward_std,
        color="#d97706",
        linewidth=1.6,
        linestyle=":",
        label="Group reward std (population)",
    )
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_ylabel("Signal")
    axes[1].set_title("GRPO learning signal", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=False, ncol=3)

    grad_line = axes[2].plot(
        step_x,
        grad_norm,
        color="#2563eb",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="Grad norm",
    )
    positive_grad = [
        value for value in grad_norm if math.isfinite(value) and value > 0
    ]
    if positive_grad:
        axes[2].set_yscale("log")
    axes[2].set_ylabel("Grad norm")
    axes[2].set_title("Optimization health", loc="left", fontweight="bold")
    health_axis = axes[2].twinx()
    health_axis.spines["top"].set_visible(False)
    kl_line = health_axis.plot(
        step_x,
        kl,
        color="#d97706",
        linewidth=1.7,
        linestyle="--",
        label="KL",
    )
    is_line = health_axis.plot(
        step_x,
        is_ratio,
        color="#64748b",
        linewidth=1.5,
        linestyle=":",
        label="IS ratio mean",
    )
    health_axis.set_ylabel("KL / IS ratio")
    axes[2].legend(
        grad_line + kl_line + is_line,
        [line.get_label() for line in grad_line + kl_line + is_line],
        loc="upper right",
        frameon=False,
        ncol=3,
    )
    axes[2].set_xlabel("Cumulative rollouts")

    reward = run["results"]["reward"]
    title = (
        f"{run['run_id']} · {run['status']} · "
        f"{run['train']['steps_completed']}/{run['train']['steps_target']} steps · "
        f"{reward['successes']}/{reward['attempts']} successes · "
        f"{reward['nondegenerate_groups']} non-degenerate groups"
    )
    figure.suptitle(title, x=0.08, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.95), h_pad=2.0)

    destination = root / filename
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=150,
            facecolor=figure.get_facecolor(),
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nan(value: float | None) -> float:
    return math.nan if value is None else value
