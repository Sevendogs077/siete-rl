#!/usr/bin/env bash
# 薄启动器：supervisor 负责 vLLM 生命周期、GPU 拆分与容器清扫。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${GRPO_CONFIG:-$PROJECT_ROOT/configs/stage1.yaml}"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
else
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#devices[@]}" -ne 4 ]] || [[ -z "${devices[0]}" || -z "${devices[1]}" || -z "${devices[2]}" || -z "${devices[3]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES 必须恰好包含四个非空条目" >&2
    exit 2
  fi
fi
exec .venv/bin/siete-rl grpo --config "$config_path"
