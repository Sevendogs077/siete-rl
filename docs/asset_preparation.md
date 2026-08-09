# 资产准备

训练与评测依赖的外部资产均不在仓库内，本文档给出逐项获取方式。所有路径相对仓库根目录。完成后运行 `bash scripts/qualify.sh` 做全套验收。

## 硬件与系统

- 操作系统：Linux x86_64，已安装 Docker 与 NVIDIA 驱动。
- GPU：训练需要 2 张（vLLM server 与 trainer 各一张，由启动器自动拆分），评测需要 1 张。参考配置为 A100 80GB；trainer 卡实测峰值约 42GB，48GB 级显卡可用。
- 磁盘：至少 400 GB 空闲。任务镜像合计约 284 GB，另有模型约 15 GB 与数据文件。
- Python 3.12（仓库 `.python-version` 锁定 3.12.13）与 [uv](https://docs.astral.sh/uv/)。

## 1. Python 环境

```bash
uv sync
```

依赖锁定在 `uv.lock`，torch / vLLM 使用 cu129 轮子。

## 2. 数据集

三个 Hugging Face 数据集，按固定 revision 下载到 `data/swegym/`。训练需要前两个，评测需要第三个。

```bash
# SWE-Gym 官方全集（提供 gold patch、test patch、F2P/P2P 清单）
huggingface-cli download SWE-Gym/SWE-Gym --repo-type dataset \
  --revision bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb \
  --local-dir data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb

# SWE-Gym 训练子集（100 个任务，提供 eval_script 与镜像名）
huggingface-cli download SumanthRH/SWE-Gym-Subset --repo-type dataset \
  --revision 3f22e68f673027edbaebe3424e4c20ae580563fd \
  --local-dir data/swegym/SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd

# SWE-bench Verified（评测用 500 个任务）
huggingface-cli download SWE-bench/SWE-bench_Verified --repo-type dataset \
  --revision 91aa3ed51b709be6457e12d00300a6a596d4c6a3 \
  --local-dir data/swegym/SWE-Bench__SWE-bench_Verified/91aa3ed51b709be6457e12d00300a6a596d4c6a3
```

国内网络可前置 `HF_ENDPOINT=https://hf-mirror.com`。revision 与配置文件中的 `official_revision` / `subset_revision` 一一对应，改动任意一处都会导致加载失败。

## 3. 基座模型

模型为 [NovaSky-AI/SWE-Gym-OpenHands-7B-Agent](https://huggingface.co/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent)。配置默认从 ModelScope 缓存路径读取：

```bash
pip install modelscope
modelscope download --model NovaSky-AI/SWE-Gym-OpenHands-7B-Agent
```

也可以从 Hugging Face 下载到任意位置，然后修改 `configs/*.yaml` 中的 `model.model_path` 与 `model.tokenizer_path`。

## 4. 专用 Docker daemon

所有容器操作都钉死在专用 daemon `unix:///run/docker-swegym/docker.sock`，代码拒绝回落到共享 daemon（`src/siete_rl/docker.py`）。因此需要单独运行一个 dockerd 实例，需要 root 权限。

systemd 单元示例（`/etc/systemd/system/docker-swegym.service`），`--data-root` 请指向有 400 GB 以上空闲的磁盘：

```ini
[Unit]
Description=Dedicated Docker daemon for siete-rl
After=network.target

[Service]
ExecStartPre=/bin/mkdir -p /run/docker-swegym
ExecStart=/usr/bin/dockerd \
  --host=unix:///run/docker-swegym/docker.sock \
  --data-root=/var/lib/docker-swegym \
  --exec-root=/run/docker-swegym/exec \
  --pidfile=/run/docker-swegym/docker.pid
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now docker-swegym
```

临时使用也可以直接前台启动：`sudo dockerd --host=unix:///run/docker-swegym/docker.sock --data-root=/var/lib/docker-swegym`。

## 5. 任务镜像

镜像只能从 prepare / 手动拉取进入专用 daemon，配置中 `pull_policy: never`，运行期不会联网拉取。

### 训练镜像（100 个）

```bash
bash scripts/prepare.sh
```

脚本从子集 parquet 推出 100 个 `xingyaoww/sweb.eval.x86_64.*` 镜像名，逐个拉取并生成 `assets/swegym/<task_id>/`。默认镜像源为 `docker.1panel.live`，可用环境变量覆盖，例如直连官方源：`MIRROR=docker.io bash scripts/prepare.sh`。并发与重试次数由 `CONCURRENCY`、`RETRIES` 控制。

### 评测镜像（500 个）

镜像清单文件不在仓库内，由 Verified parquet 重新生成，生成后用 sha256 自检：

```bash
mkdir -p data/swegym/verified_pull
.venv/bin/python - <<'EOF'
import pyarrow.parquet as pq

rows = pq.read_table(
    "data/swegym/SWE-Bench__SWE-bench_Verified/91aa3ed51b709be6457e12d00300a6a596d4c6a3/data/test-00000-of-00001.parquet",
    columns=["instance_id"],
).to_pylist()
tags = sorted(
    f"swebench/sweb.eval.x86_64.{r['instance_id'].lower()}:latest".replace("__", "_1776_")
    for r in rows
)
with open("data/swegym/verified_pull/images-x86_64.txt", "w") as f:
    f.write("\n".join(tags) + "\n")
EOF
sha256sum data/swegym/verified_pull/images-x86_64.txt
# 应为 b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a
```

然后拉取：

```bash
export DOCKER_HOST=unix:///run/docker-swegym/docker.sock
xargs -a data/swegym/verified_pull/images-x86_64.txt -P 2 -I{} docker pull {}
```

## 6. SWE-bench harness（仅评测需要）

评测通过固定 revision 的 SWE-bench harness 评分，要求源码树干净：

```bash
git clone https://github.com/SWE-bench/SWE-bench .external/swe-bench
git -C .external/swe-bench checkout f7bbbb2ccdf479001d6467c9e34af59e44a840f9

python -m venv .external/swe-bench-venv
.external/swe-bench-venv/bin/pip install -e .external/swe-bench
```

评测时通过环境变量指定 harness Python（无默认值，缺失即报错）：

```bash
export EVAL_HARNESS_PYTHON="$PWD/.external/swe-bench-venv/bin/python"
```

harness 路径默认为 `.external/swe-bench`，可用 `EVAL_HARNESS_ROOT` 覆盖。`EVAL_TASK_IDS` 可将评测限制到部分任务（逗号分隔），默认全部 500 题。

rollout 和 official harness 默认均为单 worker。A100 80 GB 环境建议从 4 开始：

```bash
EVAL_ROLLOUT_WORKERS=4 EVAL_HARNESS_WORKERS=4 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh outputs/<run-id>
```

`EVAL_ROLLOUT_WORKERS` 并发运行相互隔离的 agent environment/container，共享同一个 vLLM server；`EVAL_HARNESS_WORKERS` 直接控制 SWE-bench official harness。两者都必须是正整数。并发预算按同时运行的 eval 进程累加，例如两个 `EVAL_ROLLOUT_WORKERS=4` 的进程最多会同时运行 8 个 rollout container。

## 7. 验收

```bash
bash scripts/qualify.sh
```

qualify 依次检查配置、数据集、资产、daemon、镜像、模型与 GPU，全部通过即就绪。之后按 README 的命令启动训练与评测。
