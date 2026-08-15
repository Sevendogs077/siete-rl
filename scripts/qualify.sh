#!/usr/bin/env bash
# 唯一全套资格检查入口：config / dataset / assets / docker / tokenizer / GPU。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${GRPO_CONFIG:-$PROJECT_ROOT/configs/stage1.yaml}"
exec .venv/bin/python -m siete_rl.qualify --config "$config_path" "$@"
