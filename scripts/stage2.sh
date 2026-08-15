#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "用法: bash scripts/stage2.sh outputs/<stage1-run-id>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ADAPTER_PATH="$1"
export GRPO_CONFIG="$SCRIPT_DIR/../configs/stage2.yaml"
exec "$SCRIPT_DIR/grpo.sh"
