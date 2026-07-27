#!/usr/bin/env bash
# 唯一全套资格检查入口：config / dataset / assets / docker / tokenizer / GPU。
set -euo pipefail

config_path="${GRPO_CONFIG:-/home/2025user/zyp/work/2607_trl_swe_agent/configs/grpo_swegym_openhands_7b_lora.yaml}"
exec .venv/bin/python -m swe_agent.qualify --config "$config_path" "$@"
