#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "用法: bash scripts/stage1.sh [checkpoint-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset MODEL_ADAPTER_PATH
export RESUME_FROM_CHECKPOINT="${1:-}"
export GRPO_CONFIG="$SCRIPT_DIR/../configs/stage1.yaml"
exec "$SCRIPT_DIR/grpo.sh"
