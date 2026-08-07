#!/usr/bin/env bash
# 读取 outputs/_selected/ 下各 run 的评测结果，生成 README 用的 benchmark strip 图。
# 输入：outputs/_selected/<run>/evals/<eval>/candidate/{rollout-state.json, official-report.json}
# 输出：docs/assets/results-chart-{light,dark}.png（透明底，按主题切换）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

.venv/bin/python - <<'EOF'
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SELECTED = Path("outputs/_selected")
OUT_TEMPLATE = "docs/assets/results-chart-{theme}.png"

# 强调色取自 docs/assets/logo-*.png 实测主色；legend 与 rail 共享同一组色板常量
THEMES = {
    "light": {
        "accent": "#04648c",
        "text": "#24292f",
        "secondary": "#6e7781",
        "grays": ["#c9cfd7", "#9aa3ad", "#6b7280", "#3f4650"],
    },
    "dark": {
        "accent": "#34a4c4",
        "text": "#e6edf3",
        "secondary": "#8b949e",
        "grays": ["#4a525b", "#636b75", "#7d8590", "#9ba3ad"],
    },
}

SEGMENTS = [
    ("resolved", "Resolved"),
    ("submitted_unresolved", "Unresolved"),
    ("overlong", "Context Limit"),
    ("itercap", "Iteration Cap"),
    ("infra", "Infra Error"),
]

RAIL_LEN = 52.0      # rail 长度（数据单位）
GUTTER = 3.0         # label|rail 与 rail|metric 的固定间距
RUN_STEP = 1.0       # run 间纵向节奏
LEGEND_DY = 0.55     # legend 距首行 rail 中心的距离


def run_label(run_dir: Path) -> str:
    """展示实验演进：只标相对上一级的算法增量。"""

    try:
        import yaml

        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if config.get("generation", {}).get("max_repeat_action"):
            return "+ Repeat Guard"
        if config.get("grpo", {}).get("reward_type") == "layered":
            return "+ Layered Reward"
        return "Vanilla GRPO"
    except Exception:
        return run_dir.name


def latest_candidate(run_dir: Path) -> Path | None:
    """选 outcome 数最多的 eval（完整评测），并列取最新；跳过部分任务的试跑。"""

    evals = run_dir / "evals"
    if not evals.is_dir():
        return None
    best: tuple[int, str, Path] | None = None
    for entry in sorted(evals.iterdir()):
        candidate = entry / "candidate"
        state = candidate / "rollout-state.json"
        if not (candidate / "official-report.json").is_file() or not state.is_file():
            continue
        try:
            count = len(json.loads(state.read_text(encoding="utf-8"))["outcomes"])
        except Exception:
            continue
        if best is None or (count, entry.name) >= (best[0], best[1]):
            best = (count, entry.name, candidate)
    return best[2] if best else None


def outcome_buckets(run_dir: Path) -> dict[str, int] | None:
    """每题恰好归入一桶：infra > resolved > submitted > overlong > itercap。"""

    candidate = latest_candidate(run_dir)
    if candidate is None:
        return None
    outcomes = json.loads((candidate / "rollout-state.json").read_text(encoding="utf-8"))["outcomes"]
    resolved = set(json.loads((candidate / "official-report.json").read_text(encoding="utf-8"))["resolved_ids"])
    buckets = {key: 0 for key, _ in SEGMENTS}
    for o in outcomes:
        if o["infrastructure_error"] is not None:
            bucket = "infra"
        elif o["instance_id"] in resolved:
            bucket = "resolved"
        elif o["termination"] == "submitted":
            bucket = "submitted_unresolved"
        elif o["termination"] == "context_overlong":
            bucket = "overlong"
        else:
            bucket = "itercap"
        buckets[bucket] += 1
    return buckets


rows = []
for run_dir in sorted((p for p in SELECTED.iterdir() if p.is_dir()), key=lambda p: p.name):
    buckets = outcome_buckets(run_dir)
    if buckets is None:
        print(f"skip {run_dir.name}: no completed eval candidate")
        continue
    rows.append((run_label(run_dir), buckets))

if not rows:
    raise SystemExit("no evaluated runs under outputs/_selected/")


def segment_color(t: dict, i: int) -> str:
    return t["accent"] if i == 0 else t["grays"][i - 1]


def render(theme: str) -> None:
    t = THEMES[theme]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "SF Pro Text", "Segoe UI", "Arial", "Lato", "DejaVu Sans"]
    n = len(rows)
    run_y0 = 0.30
    legend_y = run_y0 + (n - 1) * RUN_STEP + LEGEND_DY
    height_data = legend_y + 0.28
    fig, ax = plt.subplots(figsize=(8.6, 0.40 + 0.30 * n), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, height_data)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_width_data(artist) -> float:
        fig.canvas.draw()
        ext = artist.get_window_extent(renderer)
        return inv.transform((ext.x1, 0))[0] - inv.transform((ext.x0, 0))[0]

    # label 列宽按最长 label 实测，右对齐贴 rail；rail/metric 依次排布
    label_texts = [
        ax.text(0, 0, label, ha="left", va="center", fontsize=10.5, color=t["text"])
        for label, _ in rows
    ]
    label_w = max(text_width_data(txt) for txt in label_texts)
    rail_x0 = label_w + 2 * GUTTER
    rail_x1 = rail_x0 + RAIL_LEN
    metric_cx = rail_x1 + GUTTER + 4.2  # 4.2 ≈ "100.0%" 半宽

    # rail 厚度按像素标定：约 6px（200dpi），仅两端极轻圆角
    bbox = ax.get_window_extent()
    ppy = bbox.height / height_data
    ppx = bbox.width / 100.0
    rail_h = 6.0 / ppy
    rx = (rail_h / 2) * ppy / ppx

    # 共享 legend：总跨度与 rail 严格一致（两端对齐、间距均布）
    swatch_w, swatch_gap = 1.2, 0.7
    legend_texts = [
        ax.text(0, legend_y, name, ha="left", va="center", fontsize=7.8, color=t["secondary"])
        for _, name in SEGMENTS
    ]
    item_w = [swatch_w + swatch_gap + text_width_data(txt) for txt in legend_texts]
    spacing = (RAIL_LEN - sum(item_w)) / (len(SEGMENTS) - 1)
    cursor = rail_x0
    for i, txt in enumerate(legend_texts):
        ax.plot([cursor, cursor + swatch_w], [legend_y, legend_y],
                color=segment_color(t, i), linewidth=1.3, solid_capstyle="butt")
        txt.set_position((cursor + swatch_w + swatch_gap, legend_y))
        cursor += item_w[i] + spacing

    for i, ((label, buckets), txt) in enumerate(zip(rows, label_texts)):
        y = run_y0 + (n - 1 - i) * RUN_STEP
        total = sum(buckets.values())
        resolved = buckets["resolved"]

        track = FancyBboxPatch(
            (rail_x0, y - rail_h / 2), RAIL_LEN, rail_h,
            boxstyle=f"round,pad=0,rounding_size={rx:.5f}",
            mutation_aspect=ppx / ppy,
            facecolor=t["grays"][0], edgecolor="none",
        )
        ax.add_patch(track)

        left = rail_x0
        for j, (key, _) in enumerate(SEGMENTS):
            count = buckets[key]
            if count == 0:
                continue
            width = count / total * RAIL_LEN
            (rect,) = ax.barh(y, width, left=left, height=rail_h,
                              color=segment_color(t, j), edgecolor="none")
            rect.set_clip_path(track)
            left += width

        txt.set_position((label_w + GUTTER, y))
        txt.set_ha("right")
        ax.text(metric_cx, y + 0.10, f"{resolved / total:.1%}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=t["accent"])
        ax.text(metric_cx, y - 0.16, f"{resolved} / {total}", ha="center", va="center",
                fontsize=7.8, color=t["secondary"])

    ax.set_xlim(-1, metric_cx + 8)
    out = OUT_TEMPLATE.format(theme=theme)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


for theme in THEMES:
    render(theme)
for label, buckets in rows:
    print(" ", label, buckets)
EOF
