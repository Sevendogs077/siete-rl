#!/usr/bin/env bash
# 将固定外部 7B 参考与选定的本地发布评测绘制为 README 对比图。
# 本地输入：outputs/_selected/<run>/evals/<eval>/candidate/{rollout-state.json, official-report.json}
# 输出：docs/assets/results-chart-{light,dark}.png（透明底，按主题切换）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

.venv/bin/python - <<'EOF'
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SELECTED = Path("outputs/_selected")
RELEASE_RUN = SELECTED / "20260811T171350Z-086c"
LOCAL_LABEL = "SieteRL-Agent-7B-v0"
OUT_TEMPLATE = "docs/assets/results-chart-{theme}.png"

# 固定到原始发布来源；数值是各来源报告的 SWE-bench Verified resolved rate。
EXTERNAL_ROWS = [
    ("Qwen2.5-Coder-7B-Instruct", 1.8),
    ("SWE-Gym-OpenHands-7B-Agent", 10.6),
    ("SkyRL-Agent-7B-v0", 14.6),
]

THEMES = {
    "light": {
        "accent": "#04648c",
        "text": "#24292f",
        "secondary": "#6e7781",
        "track": "#e7ebef",
        "reference": "#89939e",
        "guide": "#d8dde3",
    },
    "dark": {
        "accent": "#34a4c4",
        "text": "#e6edf3",
        "secondary": "#8b949e",
        "track": "#30363d",
        "reference": "#8b949e",
        "guide": "#484f58",
    },
}

RAIL_LEN = 51.0
GUTTER = 3.0
RUN_STEP = 1.12
GROUP_GAP = 0.0


def latest_candidate(run_dir: Path) -> Path | None:
    """按 metadata 选择最新完成且 outcome 齐全的 candidate 评测。"""

    evals = run_dir / "evals"
    if not evals.is_dir():
        return None
    best: tuple[str, str, Path] | None = None
    for entry in sorted(evals.iterdir()):
        candidate = entry / "candidate"
        metadata_path = entry / "metadata.json"
        state = candidate / "rollout-state.json"
        report = candidate / "official-report.json"
        if not metadata_path.is_file() or not report.is_file() or not state.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "completed":
                continue
            finished_at = metadata["finished_at"]
            selected_count = metadata["dataset"]["selected_count"]
            count = len(json.loads(state.read_text(encoding="utf-8"))["outcomes"])
            if (
                not isinstance(finished_at, str)
                or not finished_at
                or not isinstance(selected_count, int)
                or selected_count < 1
                or count != selected_count
            ):
                continue
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            continue
        if best is None or (finished_at, entry.name) >= (best[0], best[1]):
            best = (finished_at, entry.name, candidate)
    return best[2] if best else None


def local_result(run_dir: Path) -> tuple[str, float, str, int] | None:
    candidate = latest_candidate(run_dir)
    if candidate is None:
        return None
    outcomes = json.loads(
        (candidate / "rollout-state.json").read_text(encoding="utf-8")
    )["outcomes"]
    resolved = json.loads(
        (candidate / "official-report.json").read_text(encoding="utf-8")
    )["resolved_ids"]
    total = len(outcomes)
    if total == 0:
        return None
    count = len(resolved)
    return LOCAL_LABEL, 100.0 * count / total, f"{count} / {total}", count


result = local_result(RELEASE_RUN)
if result is None:
    raise SystemExit(f"no completed candidate eval under {RELEASE_RUN}")
local_rows = [result]


def render(theme: str) -> None:
    t = THEMES[theme]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Inter",
        "SF Pro Text",
        "Segoe UI",
        "Arial",
        "Lato",
        "DejaVu Sans",
    ]

    rows = [
        {"label": label, "rate": rate, "detail": "reported", "local": False}
        for label, rate in EXTERNAL_ROWS
    ] + [
        {"label": label, "rate": rate, "detail": detail, "local": True}
        for label, rate, detail, _ in local_rows
    ]
    max_rate = max(row["rate"] for row in rows)
    scale_max = max(20, int(math.ceil(max_rate * 1.2 / 5.0) * 5))

    positions = []
    for i in range(len(rows)):
        y = (len(rows) - 1 - i) * RUN_STEP
        if i < len(EXTERNAL_ROWS):
            y += GROUP_GAP
        positions.append(float(y))

    axis_dy = 0.58
    height_data = positions[0] + 0.32
    fig, ax = plt.subplots(figsize=(8.6, 0.42 + 0.40 * len(rows)), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(positions[-1] - axis_dy - 0.12, height_data)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_width_data(artist) -> float:
        fig.canvas.draw()
        ext = artist.get_window_extent(renderer)
        return inv.transform((ext.x1, 0))[0] - inv.transform((ext.x0, 0))[0]

    label_texts = [
        ax.text(0, 0, row["label"], ha="left", va="center", fontsize=10.5, color=t["text"])
        for row in rows
    ]
    label_w = max(text_width_data(txt) for txt in label_texts)
    rail_x0 = label_w + 2 * GUTTER
    rail_x1 = rail_x0 + RAIL_LEN
    metric_x = rail_x1 + GUTTER

    bbox = ax.get_window_extent()
    ppy = bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ppx = bbox.width / 100.0
    rail_h = 7.0 / ppy
    rx = (rail_h / 2) * ppy / ppx

    guide_bottom = positions[-1] - 0.25
    guide_top = positions[0] + 0.25
    tick_step = 5 if scale_max <= 30 else 10
    for tick in range(0, scale_max + 1, tick_step):
        x = rail_x0 + tick / scale_max * RAIL_LEN
        ax.plot([x, x], [guide_bottom, guide_top], color=t["guide"], linewidth=0.45, zorder=0)
        ax.text(
            x,
            positions[-1] - axis_dy,
            f"{tick}%",
            ha="center",
            va="center",
            fontsize=7.5,
            color=t["secondary"],
        )

    for row, y, txt in zip(rows, positions, label_texts):
        track = FancyBboxPatch(
            (rail_x0, y - rail_h / 2),
            RAIL_LEN,
            rail_h,
            boxstyle=f"round,pad=0,rounding_size={rx:.5f}",
            mutation_aspect=ppx / ppy,
            facecolor=t["track"],
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(track)
        width = row["rate"] / scale_max * RAIL_LEN
        fill = FancyBboxPatch(
            (rail_x0, y - rail_h / 2),
            width,
            rail_h,
            boxstyle=f"round,pad=0,rounding_size={rx:.5f}",
            mutation_aspect=ppx / ppy,
            facecolor=t["accent"] if row["local"] else t["reference"],
            edgecolor="none",
            zorder=2,
        )
        fill.set_clip_path(track)
        ax.add_patch(fill)

        txt.set_position((label_w + GUTTER, y))
        txt.set_ha("right")
        ax.text(
            metric_x,
            y + 0.12,
            f"{row['rate']:.1f}%",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=t["accent"] if row["local"] else t["text"],
        )
        ax.text(
            metric_x,
            y - 0.21,
            row["detail"],
            ha="left",
            va="center",
            fontsize=7.5,
            color=t["secondary"],
        )

    ax.set_xlim(GUTTER - 1, metric_x + 10)
    out = OUT_TEMPLATE.format(theme=theme)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


for theme in THEMES:
    render(theme)
for label, rate in EXTERNAL_ROWS:
    print(f"  external: {label}: {rate:.1f}%")
for label, rate, detail, _ in local_rows:
    print(f"  local: {label}: {rate:.1f}% ({detail})")
EOF
