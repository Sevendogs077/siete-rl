#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "用法: bash scripts/stage2.sh outputs/<stage1-run-id> [checkpoint-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ADAPTER_PATH="$1"
export RESUME_FROM_CHECKPOINT="${2:-}"
export GRPO_CONFIG="$SCRIPT_DIR/../configs/stage2.yaml"
exec "$SCRIPT_DIR/grpo.sh"
