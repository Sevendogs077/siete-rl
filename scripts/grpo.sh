#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/grpo.sh <complete-config.yaml>" >&2
  exit 2
fi

exec .venv/bin/swe-agent grpo --config "$1"
