#!/usr/bin/env bash
# 唯一全套资格检查入口：config / dataset / assets / docker / tokenizer / GPU。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${GRPO_CONFIG:-$PROJECT_ROOT/configs/grpo_swegym_openhands_7b_lora.yaml}"
exec .venv/bin/python -m siete_rl.qualify --config "$config_path" "$@"
