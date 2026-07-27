#!/usr/bin/env bash
# 薄启动器：supervisor 负责 vLLM 生命周期、GPU 拆分与容器清扫。
set -euo pipefail

config_path="${GRPO_CONFIG:-/home/2025user/zyp/work/2607_trl_swe_agent/configs/grpo_swegym_openhands_7b_lora.yaml}"
exec .venv/bin/swe_agent grpo --config "$config_path"
