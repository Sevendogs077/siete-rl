#!/usr/bin/env bash
# 训练后评测薄入口；所有资源与生命周期检查均由 siete_rl.eval 负责。
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export EVAL_ROLLOUT_WORKERS="${EVAL_ROLLOUT_WORKERS-16}"
export EVAL_HARNESS_WORKERS="${EVAL_HARNESS_WORKERS-4}"
export EVAL_HARNESS_PYTHON="${EVAL_HARNESS_PYTHON:-$PWD/.external/swe-bench-venv/bin/python}"
exec uv run --no-sync python -m siete_rl.eval "$@"
