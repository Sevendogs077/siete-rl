#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export DOCKER_HOST=unix:///run/docker-swegym/docker.sock
test -S /run/docker-swegym/docker.sock
exec .venv/bin/siete-rl prepare \
  --config configs/grpo_swegym_openhands_7b_lora.yaml
