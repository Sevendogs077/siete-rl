#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/grpo.sh <complete-config.yaml>" >&2
  exit 2
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" != *,* || "${CUDA_VISIBLE_DEVICES#*,}" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must select server_gpu,trainer_gpu" >&2
  exit 2
fi

server_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
trainer_gpu="${CUDA_VISIBLE_DEVICES#*,}"
model_path=$(.venv/bin/python -c \
  'import sys; from swe_agent.config import load_config; print(load_config(sys.argv[1])[0].model.model_path)' \
  "$1")
export SWE_AGENT_RUN_ID="${SWE_AGENT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

cleanup() {
  local status=$?
  trap - EXIT
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    mapfile -t containers < <(docker ps -aq --filter "label=swe-agent.run_id=$SWE_AGENT_RUN_ID")
    [[ ${#containers[@]} -eq 0 ]] || docker rm -f "${containers[@]}" >/dev/null
  fi
  exit "$status"
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="$server_gpu" .venv/bin/trl vllm-serve \
  --model "$model_path" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code &
server_pid=$!

CUDA_VISIBLE_DEVICES="$trainer_gpu" .venv/bin/swe-agent grpo --config "$1"
