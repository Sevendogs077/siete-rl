#!/usr/bin/env bash
# 薄启动器：supervisor 负责 vLLM 生命周期、GPU 拆分与容器清扫。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${GRPO_CONFIG:-$PROJECT_ROOT/configs/grpo_swegym_openhands_7b_lora.yaml}"
exec .venv/bin/siete-rl grpo --config "$config_path"
