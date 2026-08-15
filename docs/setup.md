# Setup

This guide takes a clean checkout to a qualified SieteRL runtime. Run every command from the repository root.

## Requirements

- Linux x86_64 with Docker and an NVIDIA driver.
- Python 3.12 and [uv](https://docs.astral.sh/uv/). The repository pins Python 3.12.13.
- Four GPUs for training with the qualified `colocate` configuration, or one GPU for evaluation. The reference system uses A100 80 GB GPUs.
- At least 400 GB of free disk space: task images use about 284 GB, and the base model uses about 15 GB.
- Root access to start a dedicated Docker daemon.

## 1. Install the project

```bash
uv sync
```

Dependencies are locked in `uv.lock`; PyTorch and vLLM use CUDA 12.9 wheels.

## 2. Download the data

`scripts/prepare.sh` downloads the four pinned training sources automatically. Evaluation requires SWE-bench Verified separately.

```bash
# The 500-task evaluation split
uv run --no-sync hf download SWE-bench/SWE-bench_Verified --repo-type dataset \
  --revision 91aa3ed51b709be6457e12d00300a6a596d4c6a3 \
  --local-dir data/swegym/SWE-Bench__SWE-bench_Verified/91aa3ed51b709be6457e12d00300a6a596d4c6a3
```

Set `HF_ENDPOINT=https://hf-mirror.com` before this command if a mirror is required.

## 3. Download the base model

```bash
uv run --no-sync hf download NovaSky-AI/SWE-Gym-OpenHands-7B-Agent \
  --local-dir models/SWE-Gym-OpenHands-7B-Agent

export MODEL_PATH="$PWD/models/SWE-Gym-OpenHands-7B-Agent"
export TOKENIZER_PATH="$MODEL_PATH"
```

Keep both variables in the shell used for qualification and training. Without them, the default config reads the equivalent ModelScope cache path.

## 4. Start the dedicated Docker daemon

SieteRL only uses `unix:///run/docker-swegym/docker.sock` and never falls back to a shared daemon. Start `dockerd` in a separate terminal and place its data root on the disk with at least 400 GB free:

```bash
sudo mkdir -p /run/docker-swegym /var/lib/docker-swegym
sudo dockerd \
  --host=unix:///run/docker-swegym/docker.sock \
  --data-root=/var/lib/docker-swegym \
  --exec-root=/run/docker-swegym/exec \
  --pidfile=/run/docker-swegym/docker.pid
```

<details>
<summary><strong>Optional persistent systemd service</strong></summary>

Create `/etc/systemd/system/docker-swegym.service`:

```ini
[Unit]
Description=Dedicated Docker daemon for SieteRL
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

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now docker-swegym
```

</details>

## 5. Prepare training assets

```bash
bash scripts/prepare.sh
```

The script builds the pinned 82-task Stage 1 plus 220-task Stage 2 course, pulls its 302 `xingyaoww/sweb.eval.x86_64.*` images into the dedicated daemon, and generates `assets/swegym/<task-id>/`. It tries Docker Hub first and falls back to `dockerproxy.net`; runtime pulling is disabled by `pull_policy: never`. Re-running the command reuses local images and refreshes their manifests.

## 6. Prepare evaluation

### Pull the 500 task images

Generate the pinned image list from the Verified parquet file:

```bash
mkdir -p data/swegym/verified_pull
uv run --no-sync python - <<'PY'
import pyarrow.parquet as pq

rows = pq.read_table(
    "data/swegym/SWE-Bench__SWE-bench_Verified/91aa3ed51b709be6457e12d00300a6a596d4c6a3/data/test-00000-of-00001.parquet",
    columns=["instance_id"],
).to_pylist()
tags = sorted(
    f"swebench/sweb.eval.x86_64.{row['instance_id'].lower()}:latest".replace(
        "__", "_1776_"
    )
    for row in rows
)
with open("data/swegym/verified_pull/images-x86_64.txt", "w") as output:
    output.write("\n".join(tags) + "\n")
PY

sha256sum data/swegym/verified_pull/images-x86_64.txt
# Expected: b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a
```

Pull the verified list into the dedicated daemon:

```bash
export DOCKER_HOST=unix:///run/docker-swegym/docker.sock
xargs -a data/swegym/verified_pull/images-x86_64.txt -P 2 -I{} docker pull {}
```

### Install the evaluation harness

```bash
mkdir -p .external
git clone https://github.com/SWE-bench/SWE-bench .external/swe-bench
git -C .external/swe-bench checkout f7bbbb2ccdf479001d6467c9e34af59e44a840f9

uv venv --python 3.12 .external/swe-bench-venv
uv pip install --python .external/swe-bench-venv/bin/python \
  -e .external/swe-bench

export EVAL_HARNESS_PYTHON="$PWD/.external/swe-bench-venv/bin/python"
```

`EVAL_HARNESS_PYTHON` is required and has no default. The harness source tree must remain clean; its default location is `.external/swe-bench`, and `EVAL_HARNESS_ROOT` overrides that path.

## 7. Validate the runtime

```bash
bash scripts/qualify.sh
```

Qualification loads the actual training table and runtime assets, then checks the dedicated daemon, task image identities, model, tokenizer, and GPU topology. Do not start training until every check passes.

## 8. Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/grpo.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh outputs/<run-id>
```

### W&B

The default config logs training metrics to W&B. For an online run:

```bash
export WANDB_API_KEY=<your-key>
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/grpo.sh
```

Set `wandb.mode: offline` for local runs, or `wandb.enabled: false` to turn it off.

`scripts/eval.sh` defaults to 16 rollout workers and 4 official-harness workers. Override them to match available CPU and memory:

```bash
EVAL_ROLLOUT_WORKERS=8 EVAL_HARNESS_WORKERS=2 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh outputs/<run-id>
```

`EVAL_ROLLOUT_WORKERS` runs isolated agent containers against one vLLM server; `EVAL_HARNESS_WORKERS` controls official scoring. Both must be positive integers. `EVAL_TASK_IDS` accepts a comma-separated task subset and defaults to all 500 tasks. Direct `python -m siete_rl.eval` calls use conservative `1/1` worker defaults. Counts add across concurrent evaluations, so two default runs may start 32 rollout containers.
