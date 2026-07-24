#!/usr/bin/env bash
# 将 SWE-Gym subset 全部实例镜像拉取到专用 docker-swegym daemon。
# 安全约束：只操作 /run/docker-swegym/docker.sock，绝不触碰共享 daemon。
set -euo pipefail

export DOCKER_HOST="unix:///run/docker-swegym/docker.sock"
if [[ ! -S /run/docker-swegym/docker.sock ]]; then
  echo "docker-swegym socket 不存在，拒绝继续（绝不回落到共享 daemon）" >&2
  exit 2
fi

PARQUET="${PARQUET:-data/swegym/SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd/data/train-00000-of-00001.parquet}"
MIRROR="${MIRROR:-docker.1panel.live}"
CONCURRENCY="${CONCURRENCY:-2}"
RETRIES="${RETRIES:-3}"
STATE_DIR="${STATE_DIR:-data/swegym/pull}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/pull.log"
FAILED="$STATE_DIR/pull_failed.txt"
: > "$FAILED"

mapfile -t tags < <(.venv/bin/python - "$PARQUET" <<'EOF'
import sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
for i in t.column('instance_id').to_pylist():
    print(f'docker.io/xingyaoww/sweb.eval.x86_64.{i.replace("__", "_s_")}:latest')
EOF
)
echo "[$(date -Is)] total ${#tags[@]} tags, mirror=$MIRROR, concurrency=$CONCURRENCY" | tee -a "$LOG"

pull_one() {
  local tag="$1"
  local mirror_ref="$MIRROR/${tag#docker.io/}"
  if docker image inspect "$tag" >/dev/null 2>&1; then
    echo "[$(date -Is)] SKIP $tag" >> "$LOG"
    return 0
  fi
  for attempt in $(seq 1 "$RETRIES"); do
    if docker pull "$mirror_ref" >> "$LOG" 2>&1; then
      docker tag "$mirror_ref" "$tag"
      docker rmi "$mirror_ref" >/dev/null 2>&1 || true
      # docker pull 的详细输出只写入日志；每个镜像完成时在终端给出简洁提示。
      echo "[$(date -Is)] OK   $tag (attempt $attempt)" | tee -a "$LOG"
      return 0
    fi
    echo "[$(date -Is)] RETRY $tag (attempt $attempt failed)" >> "$LOG"
    sleep $((attempt * 10))
  done
  echo "[$(date -Is)] FAIL $tag" >> "$LOG"
  echo "$tag" >> "$FAILED"
  return 1
}
export -f pull_one
export LOG FAILED MIRROR RETRIES

printf '%s\n' "${tags[@]}" | xargs -P "$CONCURRENCY" -I{} bash -c 'pull_one "$@"' _ {} || true

ok=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^docker.io/xingyaoww/sweb.eval.x86_64\.' || true)
echo "[$(date -Is)] done: $ok/${#tags[@]} images present; failures listed in $FAILED" | tee -a "$LOG"
