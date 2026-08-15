#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset MODEL_ADAPTER_PATH
export GRPO_CONFIG="$SCRIPT_DIR/../configs/stage1.yaml"
exec "$SCRIPT_DIR/grpo.sh"
