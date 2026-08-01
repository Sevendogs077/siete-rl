#!/usr/bin/env bash
# 从锁定的 SWE-Gym harness 源码重建五个旧版 mypy 镜像，避免构建过程修改 /testbed。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EXPECTED_DOCKER_HOST="unix:///run/docker-swegym/docker.sock"
export DOCKER_HOST="${DOCKER_HOST:-$EXPECTED_DOCKER_HOST}"
if [[ "$DOCKER_HOST" != "$EXPECTED_DOCKER_HOST" || ! -S /run/docker-swegym/docker.sock ]]; then
  echo "只允许使用专用 docker-swegym daemon: $EXPECTED_DOCKER_HOST" >&2
  exit 2
fi

HARNESS_COMMIT="242429c188fcfd06aad13fce9a54d450470bf0ac"
STATE_DIR="data/swegym/build/mypy-image-build/$HARNESS_COMMIT"
HARNESS_DIR="$STATE_DIR/SWE-Bench-Fork"
HARNESS_ARCHIVE="$STATE_DIR/SWE-Bench-Fork.tar.gz"
HARNESS_VENV="$STATE_DIR/.venv"
DATASET_JSON="$STATE_DIR/tasks.json"
OFFICIAL_PARQUET="data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb/data/train-00000-of-00001.parquet"
MIRROR="${MIRROR:-dockerproxy.net}"

TASKS=(
  python__mypy-10032
  python__mypy-11204
  python__mypy-11290
  python__mypy-11585
  python__mypy-11725
)

declare -A BASE_COMMITS=(
  [python__mypy-10032]="82dd59ee8da387c41b9edfe577c88d172f2c7091"
  [python__mypy-11204]="8e82171a3e68a5180fab267cad4d2b7cfa1f5cdc"
  [python__mypy-11290]="e6b91bdc5c253cefba940b0864a8257d833f0d8b"
  [python__mypy-11585]="4996d571272adde83a3de2689c0147ca1be23f2c"
  [python__mypy-11725]="4d2ff58e502b52a908f41c18fbbe7cbb32997504"
)

canonical_image() {
  printf 'docker.io/xingyaoww/sweb.eval.x86_64.%s:latest' "${1/__/_s_}"
}

image_is_clean() {
  local task_id="$1"
  local image
  image="$(canonical_image "$task_id")"
  docker image inspect "$image" >/dev/null 2>&1 || return 1
  docker run --rm --pull never --network none --entrypoint /bin/bash "$image" -lc '
    test "$(git -C /testbed rev-parse HEAD)" = "$1" &&
    test -z "$(git -C /testbed status --porcelain)" &&
    /opt/miniconda3/envs/testbed/bin/python -c \
      "import importlib.metadata as m; assert m.version(\"types-typing-extensions\") == \"3.7.3\""
  ' _ "${BASE_COMMITS[$task_id]}" >/dev/null 2>&1
}

all_clean=true
for task_id in "${TASKS[@]}"; do
  if ! image_is_clean "$task_id"; then
    all_clean=false
    break
  fi
done
if [[ "$all_clean" == true ]]; then
  echo "五个旧版 mypy 镜像已经干净，跳过重建"
  exit 0
fi

if ! docker image inspect ubuntu:22.04 >/dev/null 2>&1; then
  docker pull "$MIRROR/library/ubuntu:22.04"
  docker tag "$MIRROR/library/ubuntu:22.04" ubuntu:22.04
fi

mkdir -p "$STATE_DIR"
if [[ ! -f "$HARNESS_DIR/.source-commit" ]] || [[ "$(<"$HARNESS_DIR/.source-commit")" != "$HARNESS_COMMIT" ]]; then
  curl --fail --location --retry 3 \
    --output "$HARNESS_ARCHIVE" \
    "https://codeload.github.com/SWE-Gym/SWE-Bench-Fork/tar.gz/$HARNESS_COMMIT"
  mkdir -p "$HARNESS_DIR"
  tar -xzf "$HARNESS_ARCHIVE" --strip-components=1 -C "$HARNESS_DIR"
  printf '%s\n' "$HARNESS_COMMIT" > "$HARNESS_DIR/.source-commit"
fi

GHFAST_IP="${GHFAST_IP:-}"
for _ in 1 2 3; do
  [[ -n "$GHFAST_IP" ]] && break
  GHFAST_IP="$( { getent ahostsv4 ghfast.top 2>/dev/null || true; } | awk '$2 == "STREAM" {print $1; exit}')"
done
if [[ -z "$GHFAST_IP" ]]; then
  echo "无法解析 ghfast.top，不能安全地为 Docker build 固定单次 DNS 结果" >&2
  exit 3
fi

.venv/bin/python - \
  "$HARNESS_DIR/swebench/harness/constants.py" \
  "$HARNESS_DIR/swebench/harness/docker_build.py" \
  "$HARNESS_DIR/swebench/harness/dockerfiles.py" \
  "$HARNESS_DIR/swebench/harness/test_spec.py" \
  "$GHFAST_IP" <<'PY'
from pathlib import Path
import re
import sys

constants_path = Path(sys.argv[1])
docker_build_path = Path(sys.argv[2])
dockerfiles_path = Path(sys.argv[3])
test_spec_path = Path(sys.argv[4])
ghfast_ip = sys.argv[5]
text = constants_path.read_text(encoding="utf-8")
dirty = "sed -i '1i types-typing-extensions==3.7.3' test-requirements.txt"
clean = "python -m pip install types-typing-extensions==3.7.3"
if dirty in text:
    if text.count(dirty) != 1:
        raise SystemExit(f"expected exactly one dirty mypy install command, found {text.count(dirty)}")
    constants_path.write_text(text.replace(dirty, clean), encoding="utf-8")
elif clean not in text:
    raise SystemExit("locked SWE-Gym harness no longer contains the expected mypy install command")

text = docker_build_path.read_text(encoding="utf-8")
isolated = "USE_HOST_NETWORK = False"
host = "USE_HOST_NETWORK = True"
if isolated in text:
    if text.count(isolated) != 1:
        raise SystemExit(f"expected exactly one harness network switch, found {text.count(isolated)}")
    docker_build_path.write_text(text.replace(isolated, host), encoding="utf-8")
elif host not in text:
    raise SystemExit("locked SWE-Gym harness no longer contains the expected build network switch")

text = docker_build_path.read_text(encoding="utf-8")
extra_hosts_pattern = r'extra_hosts=\{"ghfast\.top": "[0-9.]+"\},'
extra_hosts = f'extra_hosts={{"ghfast.top": "{ghfast_ip}"}},'
if re.search(extra_hosts_pattern, text):
    docker_build_path.write_text(
        re.sub(extra_hosts_pattern, extra_hosts, text, count=1),
        encoding="utf-8",
    )
else:
    network_line = '            network_mode="host" if USE_HOST_NETWORK else None,'
    build_network = f'{network_line}\n            container_limits='
    if text.count(build_network) != 1:
        raise SystemExit(f"expected exactly one image-build network argument, found {text.count(build_network)}")
    docker_build_path.write_text(
        text.replace(build_network, f"{network_line}\n            {extra_hosts}\n            container_limits="),
        encoding="utf-8",
    )

text = dockerfiles_path.read_text(encoding="utf-8")
official_miniconda = "https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh"
mirror_miniconda = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh"
if official_miniconda in text:
    if text.count(official_miniconda) != 1:
        raise SystemExit(f"expected exactly one Miniconda URL, found {text.count(official_miniconda)}")
    dockerfiles_path.write_text(text.replace(official_miniconda, mirror_miniconda), encoding="utf-8")
elif mirror_miniconda not in text:
    raise SystemExit("locked SWE-Gym harness no longer contains the expected Miniconda URL")

text = dockerfiles_path.read_text(encoding="utf-8")
official_channels = "RUN conda config --append channels conda-forge"
legacy_mirror_channels = """RUN printf 'channels:\\n  - conda-forge\\n  - defaults\\nshow_channel_urls: true\\ndefault_channels:\\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\\ncustom_channels:\\n  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\\n' > /root/.condarc"""
mirror_channels = f"""{legacy_mirror_channels}
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"""
if official_channels in text:
    if text.count(official_channels) != 1:
        raise SystemExit(f"expected exactly one Conda channel command, found {text.count(official_channels)}")
    dockerfiles_path.write_text(text.replace(official_channels, mirror_channels), encoding="utf-8")
elif mirror_channels in text:
    pass
elif legacy_mirror_channels in text:
    dockerfiles_path.write_text(text.replace(legacy_mirror_channels, mirror_channels), encoding="utf-8")
else:
    raise SystemExit("locked SWE-Gym harness no longer contains the expected Conda channel command")

text = test_spec_path.read_text(encoding="utf-8")
official_clone = '''        f"git clone -o origin https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",'''
legacy_mirror_clone = '''        f"git clone -o origin https://gitclone.com/github.com/{repo}.git {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",'''
shallow_mirror_clone = '''        f"git init {repo_directory}",
        f"git -C {repo_directory} remote add origin https://ghfast.top/https://github.com/{repo}.git",
        f"git -C {repo_directory} fetch --depth=1 origin {base_commit}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        "git checkout --detach FETCH_HEAD",
        "git config --global url.https://ghfast.top/https://github.com/.insteadOf https://github.com/",'''
inline_mirror_clone = shallow_mirror_clone.replace(
    '        f"git -C {repo_directory} fetch --depth=1 origin {base_commit}",',
    f'        f"git -c http.curloptResolve=ghfast.top:443:{ghfast_ip} -C {{repo_directory}} fetch --depth=1 origin {{base_commit}}",',
)
global_resolve_line = r'        "git config --global --add http\.curloptResolve ghfast\.top:443:[0-9.]+",\n'
text = re.sub(global_resolve_line, "", text, count=1)
inline_pattern = r"git -c http\.curloptResolve=ghfast\.top:443:[0-9.]+ -C"
if re.search(inline_pattern, text):
    text = re.sub(
        inline_pattern,
        f"git -c http.curloptResolve=ghfast.top:443:{ghfast_ip} -C",
        text,
        count=1,
    )
    test_spec_path.write_text(text, encoding="utf-8")
elif official_clone in text:
    test_spec_path.write_text(text.replace(official_clone, inline_mirror_clone), encoding="utf-8")
elif legacy_mirror_clone in text:
    test_spec_path.write_text(text.replace(legacy_mirror_clone, inline_mirror_clone), encoding="utf-8")
elif shallow_mirror_clone in text:
    test_spec_path.write_text(text.replace(shallow_mirror_clone, inline_mirror_clone), encoding="utf-8")
elif inline_mirror_clone not in text:
    raise SystemExit("locked SWE-Gym harness no longer contains the expected repository setup commands")
PY

.venv/bin/python - "$OFFICIAL_PARQUET" "$DATASET_JSON" "${TASKS[@]}" <<'PY'
from pathlib import Path
import json
import sys
import pyarrow.parquet as pq

source, target, *task_ids = sys.argv[1:]
wanted = set(task_ids)
rows = [row for row in pq.read_table(source).to_pylist() if row["instance_id"] in wanted]
found = {row["instance_id"] for row in rows}
if found != wanted:
    raise SystemExit(f"official dataset task mismatch: missing={sorted(wanted - found)}")
Path(target).write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
PY

if [[ ! -x "$HARNESS_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$HARNESS_VENV"
fi
uv pip install --python "$HARNESS_VENV/bin/python" "$HARNESS_DIR"

(
  cd "$HARNESS_DIR"
  "$PROJECT_ROOT/$HARNESS_VENV/bin/python" -m swebench.harness.prepare_images \
    --dataset_name "$PROJECT_ROOT/$DATASET_JSON" \
    --split train \
    --instance_ids "${TASKS[@]}" \
    --max_workers 1 \
    --force_rebuild true
)

for task_id in "${TASKS[@]}"; do
  built="sweb.eval.x86_64.${task_id}:latest"
  canonical="$(canonical_image "$task_id")"
  docker image inspect "$built" >/dev/null
  docker tag "$built" "$canonical"
  if ! image_is_clean "$task_id"; then
    echo "重建后镜像仍不满足干净工作区契约: $canonical" >&2
    exit 1
  fi
  echo "mypy image OK: $task_id"
done
