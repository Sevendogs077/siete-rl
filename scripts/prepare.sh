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
from swe_agent.asset_generation import image_tag_for
t = pq.read_table(sys.argv[1])
for i in t.column('instance_id').to_pylist():
    print(image_tag_for(i))
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

ok=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^xingyaoww/sweb.eval.x86_64\.' || true)
echo "[$(date -Is)] done: $ok/${#tags[@]} images present; failures listed in $FAILED" | tee -a "$LOG"

# SWE-Gym 官方旧版 mypy 规格会用 sed 修改 tracked requirements；从锁定源码重建这五个镜像。
scripts/rebuild_swegym_mypy_images.sh

echo "[$(date -Is)] generating task assets" | tee -a "$LOG"
.venv/bin/python - "$PARQUET" "$MIRROR" <<'EOF'
import sys
from pathlib import Path

import pyarrow.parquet as pq

from swe_agent.asset_generation import fetch_registry_digest, generate_task_assets, image_tag_for
from swe_agent.docker import SubprocessDockerClient
from swe_agent.swegym import OFFICIAL_REVISION, SUBSET_REVISION

parquet = sys.argv[1]
mirror = sys.argv[2]
rows = pq.read_table(parquet, columns=["instance_id"]).to_pylist()
client = SubprocessDockerClient()
official = Path(f"data/swegym/SWE-Gym__SWE-Gym/{OFFICIAL_REVISION}/data/train-00000-of-00001.parquet")
for row in rows:
    task_id = row["instance_id"]
    tag = image_tag_for(task_id)
    inspected = client.run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], timeout_sec=30)
    if inspected.exit_code != 0:
        raise SystemExit(f"image missing in dedicated daemon, run prepare.sh pull first: {tag}")
    image_id = inspected.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise SystemExit(f"docker image inspect 未返回 sha256 镜像 ID: {tag} -> {image_id!r}")
    generate_task_assets(
        task_id=task_id,
        official_path=official,
        subset_path=Path(parquet),
        assets_dir=Path("assets/swegym"),
        image=tag,
        image_id=image_id,
        registry_digest=fetch_registry_digest(mirror, task_id),
        official_revision=OFFICIAL_REVISION,
        subset_revision=SUBSET_REVISION,
    )
    print(f"assets OK {task_id}")
EOF
