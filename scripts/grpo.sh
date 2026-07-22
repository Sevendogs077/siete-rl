#!/usr/bin/env bash
set -euo pipefail

export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"

if [[ $# -ne 0 ]]; then
  echo "usage: CUDA_VISIBLE_DEVICES=0,1 GRPO_CONFIG=path/to/config.yaml scripts/grpo.sh" >&2
  exit 2
fi

config_path="${GRPO_CONFIG:-/home/2025user/zyp/work/2607_trl_swe_agent/configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r server_gpu trainer_gpu _ <<< "$cuda_devices"
if [[ -z "$server_gpu" || -z "$trainer_gpu" ]]; then
  echo "CUDA_VISIBLE_DEVICES must provide at least two GPUs" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="$trainer_gpu" .venv/bin/python -c \
  'from swe_agent.train import _require_single_visible_gpu; _require_single_visible_gpu()'
export CUDA_VISIBLE_DEVICES="$trainer_gpu"

config_output=$(.venv/bin/python -c '
import sys
from urllib.parse import urlparse

from swe_agent.config import load_config

config, _, _ = load_config(sys.argv[1])
if config.vllm.mode != "server" or config.vllm.server_base_url is None:
    raise ValueError("grpo.sh requires a vLLM server configuration")
url = urlparse(config.vllm.server_base_url)
if url.scheme != "http" or url.hostname is None or url.port is None:
    raise ValueError("vllm.server_base_url must be an http URL with an explicit port")
for value in (
    config.model.model_path,
    config.model.dtype,
    str(config.model.trust_remote_code).lower(),
    str(config.vllm.tensor_parallel_size or 1),
    str(config.vllm.gpu_memory_utilization),
    str(config.vllm.max_model_length),
    url.hostname,
    str(url.port),
):
    print(value)
' "$config_path")
mapfile -t config_values <<< "$config_output"
model_path="${config_values[0]}"
model_dtype="${config_values[1]}"
trust_remote_code="${config_values[2]}"
tensor_parallel_size="${config_values[3]}"
gpu_memory_utilization="${config_values[4]}"
max_model_length="${config_values[5]}"
server_host="${config_values[6]}"
server_port="${config_values[7]}"
export SWE_AGENT_RUN_ID="${SWE_AGENT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
server_pid=""

cleanup() {
  local status=$?
  local cleanup_failed=0
  local container_output
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]] && kill -0 -- "-$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || cleanup_failed=1
    for _ in {1..100}; do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || cleanup_failed=1
    fi
  fi
  [[ -z "$server_pid" ]] || wait "$server_pid" 2>/dev/null || true

  if ! container_output=$(docker ps -aq --filter "label=swe_agent.run_id=$SWE_AGENT_RUN_ID"); then
    cleanup_failed=1
  elif [[ -n "$container_output" ]]; then
    mapfile -t containers <<< "$container_output"
    docker rm -f "${containers[@]}" >/dev/null || cleanup_failed=1
  fi
  if [[ $cleanup_failed -ne 0 && $status -eq 0 ]]; then
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

vllm_args=(
  --model "$model_path"
  --tensor-parallel-size "$tensor_parallel_size"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --dtype "$model_dtype"
  --max-model-len "$max_model_length"
  --host "$server_host"
  --port "$server_port"
)
if [[ "$trust_remote_code" == "true" ]]; then
  vllm_args+=(--trust-remote-code)
fi

setsid env CUDA_VISIBLE_DEVICES="$server_gpu" .venv/bin/trl vllm-serve "${vllm_args[@]}" &
server_pid=$!

server_url="http://$server_host:$server_port"
vllm_ready_timeout_sec=1200
for ((attempt = 1; attempt <= vllm_ready_timeout_sec; attempt++)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    echo "vLLM server exited before becoming ready" >&2
    exit 1
  fi
  if curl --fail --silent --show-error "$server_url/health" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$trainer_gpu" .venv/bin/swe_agent grpo --config "$config_path"
    exit $?
  fi
  sleep 1
done

echo "vLLM server did not become ready within ${vllm_ready_timeout_sec} seconds" >&2
exit 1
