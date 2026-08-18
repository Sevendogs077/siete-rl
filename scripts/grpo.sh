#!/usr/bin/env bash
# 薄启动器：supervisor 负责二/四卡 colocate worker 生命周期与容器清扫。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${GRPO_CONFIG:-$PROJECT_ROOT/configs/stage1.yaml}"
gpu_count="${GPU_COUNT:-4}"
case "$gpu_count" in
  2) default_devices="0,1" ;;
  4) default_devices="0,1,2,3" ;;
  *) echo "GPU_COUNT 必须是 2 或 4" >&2; exit 2 ;;
esac
export GPU_COUNT="$gpu_count"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$default_devices"
else
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#devices[@]}" -ne "$gpu_count" ]]; then
    echo "CUDA_VISIBLE_DEVICES 必须恰好包含 $gpu_count 个非空条目" >&2
    exit 2
  fi
  for device in "${devices[@]}"; do
    if [[ -z "$device" ]]; then
      echo "CUDA_VISIBLE_DEVICES 必须恰好包含 $gpu_count 个非空条目" >&2
      exit 2
    fi
  done
fi
exec .venv/bin/siete-rl grpo --config "$config_path"
