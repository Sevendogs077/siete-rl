# 多任务 GRPO 适配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把训练管线从"单任务（moto-7023）"推广到"SWE-Gym subset 全部 100 个任务"，为正式多任务 GRPO 训练铺路。

**Architecture:** 以"资产驱动"为核心：每个任务的镜像指纹与评测资产自包含在 `assets/swegym/<task_id>/`；配置只选任务不记镜像；运行时沿用 TRL 单环境工厂 + `env.reset(task_id)` 按样本切换任务；记录层按 reward kwargs 透传的 `task_id` 逐 group 归档。

**Tech Stack:** TRL 1.8 GRPOTrainer（environment_factory + RepeatSampler）、pydantic 配置、pyarrow 数据集、Docker（docker-swegym 专用 daemon）。

---

## 关键设计决策（先于任务，理解全局）

1. **资产驱动镜像信息**：`docker.image / expected_image_id / expected_registry_digest` 从配置删除，改为从 `assets/swegym/<task_id>/selected_instance.json`（image_name）与 `manifest.json`（expected_image_id、expected_registry_digest）读取。配置只保留共享 docker 旋钮（cpus/memory/pids/平台/网络）。
2. **任务选择**：`dataset` 段改为 `tasks_dir`（资产根目录）+ `task_ids`（`null`=全部 | 显式列表）+ `max_tasks`（`null` | int，按排序截断，便于小规模试跑）。
3. **单环境工厂即可多任务**：TRL 非 dict 的 `environment_factory` 下，每个样本 `env.reset(**row)`，row 带 `task_id`；`SWEEnvironment.reset` 已按 `task_id` 查 `TaskContext`——**核心路径不用动 TRL 接线**。
4. **逐 group 记录 task_id**：TRL reward func 会把 dataset 列作为 kwargs 广播传入，dataset 行里加 `task_id` 列后，`_recording_reward` 可收到 `task_ids: list[str]`，写入 `batch.json/group.json`（每组同 prompt 即同任务）。
5. **数据集行数与采样器**：多任务后不变量改为 `generation_batch_size // num_generations <= len(dataset)`；`shuffle_dataset: true` 终于生效。
6. **镜像 digest**：镜像站回 tag 导致 daemon 内 `RepoDigests` 为空；资产生成器从镜像站 manifest API 抓取 digest 写入 manifest（moto-7212 已验证可行）。`inspect_image` 对空 RepoDigests 本已放行。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/swe_agent/assets_gen.py` | 从 parquet + daemon 生成全部任务资产（新建） | 创建 |
| `scripts/prepare.sh` | 资产预生成入口（在现有拉镜像脚本基础上扩展第二职责或新建 `prepare_assets.sh`，二选一，见 Task 1） | 修改/创建 |
| `src/swe_agent/config.py` | `dataset` 段重设计、`docker` 段删镜像三字段 | 修改 |
| `src/swe_agent/swegym.py` | 多任务 TaskContext、按资产构造 Environment | 修改 |
| `src/swe_agent/recording.py` | batch/group 记录 per-group task_id | 修改 |
| `src/swe_agent/train.py` | dataset 构建多行、prompt 长度校验改逐行、`_recording_reward` 透传 task_id | 修改 |
| `src/swe_agent/qualify.py` | 检查循环覆盖全部任务 | 修改 |
| `configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml` | dataset/docker 段新形态 | 修改 |
| `tests/unit/test_swegym.py`、`test_config.py`、`test_recording.py`、`test_train.py`、`test_qualify.py`、`test_assets_gen.py`（新） | 同步 | 修改/创建 |

---

### Task 1: 资产生成器（100 个任务的 assets 全量生成）

**Files:**
- Create: `src/swe_agent/assets_gen.py`
- Create: `tests/unit/test_assets_gen.py`
- Modify: `scripts/prepare.sh`（追加资产生成阶段；该脚本已负责拉镜像）

设计：输入 official/subset parquet + 任务清单（subset parquet 全部行），对每个任务产出与 moto-7023 同构的六个文件。幂等（已存在且哈希一致则跳过）。digest 通过镜像站 manifest API 获取（`MIRROR` 环境变量，默认 `docker.1panel.live`）；image_id 从专用 daemon `docker image inspect` 获取（镜像须已由 prepare.sh 拉取，否则报错并列出缺失清单）。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_assets_gen.py
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swe_agent.assets_gen import generate_task_assets


def _rows(instance_id: str) -> list[dict]:
    return [
        {
            "instance_id": instance_id,
            "repo": "getmoto/moto",
            "base_commit": "a" * 40,
            "version": "4.2",
            "problem_statement": "problem",
            "patch": "gold-diff",
            "test_patch": "test-diff",
            "FAIL_TO_PASS": ["t1"],
            "PASS_TO_PASS": ["t2"],
            "eval_script": "make init\npytest -q\n",
        }
    ]


def test_generate_task_assets_produces_six_consistent_files(tmp_path: Path) -> None:
    official = tmp_path / "official.parquet"
    subset = tmp_path / "subset.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), official)
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), subset)

    written = generate_task_assets(
        task_id="owner__repo-1",
        official_path=official,
        subset_path=subset,
        assets_dir=tmp_path / "assets",
        image="docker.io/x/owner_s_repo-1:latest",
        image_id="sha256:" + "1" * 64,
        registry_digest="sha256:" + "2" * 64,
    )

    root = tmp_path / "assets" / "owner__repo-1"
    assert {p.name for p in written} == {
        "selected_instance.json",
        "eval_script.sh",
        "eval_script.offline.sh",
        "gold.patch",
        "test.patch",
        "manifest.json",
    }
    assert (root / "gold.patch").read_text() == "gold-diff"
    assert "PIP_NO_INDEX=1" in (root / "eval_script.offline.sh").read_text()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["image_name"] == "docker.io/x/owner_s_repo-1:latest"
    assert manifest["expected_image_id"] == "sha256:" + "1" * 64
    assert manifest["expected_registry_digest"] == "sha256:" + "2" * 64
    assert set(manifest["files"]) == {
        "selected_instance.json",
        "eval_script.sh",
        "eval_script.offline.sh",
        "gold.patch",
        "test.patch",
    }


def test_generate_task_assets_is_idempotent(tmp_path: Path) -> None:
    official = tmp_path / "official.parquet"
    subset = tmp_path / "subset.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), official)
    pq.write_table(pa.Table.from_pylist(_rows("owner__repo-1")), subset)
    kwargs = dict(
        task_id="owner__repo-1",
        official_path=official,
        subset_path=subset,
        assets_dir=tmp_path / "assets",
        image="docker.io/x/owner_s_repo-1:latest",
        image_id="sha256:" + "1" * 64,
        registry_digest="sha256:" + "2" * 64,
    )
    first = generate_task_assets(**kwargs)
    second = generate_task_assets(**kwargs)
    assert second == first
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_assets_gen.py -q`
Expected: FAIL `ModuleNotFoundError: swe_agent.assets_gen`

- [ ] **Step 3: 实现 `src/swe_agent/assets_gen.py`**

```python
"""从锁定 parquet 与 daemon 事实生成单个/全部任务资产；深度校验由 qualify 负责。"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from swe_agent.swegym import _read_exact_row, transform_eval_script_offline


ASSET_FILES = (
    "selected_instance.json",
    "eval_script.sh",
    "eval_script.offline.sh",
    "gold.patch",
    "test.patch",
)


def generate_task_assets(
    *,
    task_id: str,
    official_path: Path,
    subset_path: Path,
    assets_dir: Path,
    image: str,
    image_id: str,
    registry_digest: str,
) -> list[Path]:
    official = _read_exact_row(official_path, task_id)
    subset = _read_exact_row(subset_path, task_id)
    root = assets_dir / task_id
    root.mkdir(parents=True, exist_ok=True)

    texts = {
        "eval_script.sh": subset["eval_script"],
        "eval_script.offline.sh": transform_eval_script_offline(subset["eval_script"]),
        "gold.patch": official["patch"],
        "test.patch": official["test_patch"],
    }
    fields = (
        "repo", "base_commit", "version", "problem_statement",
        "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    )
    selected = {"instance_id": task_id, "eval_script": subset["eval_script"], "image_name": image}
    for field in fields:
        value = official[field]
        selected[field] = value.tolist() if hasattr(value, "tolist") else value
    texts["selected_instance.json"] = json.dumps(selected, ensure_ascii=False, indent=2)

    written: list[Path] = []
    for name, text in texts.items():
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        written.append(path)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "1",
        "task_id": task_id,
        "repo_name": official["repo"],
        "base_commit": official["base_commit"],
        "image_name": image,
        "expected_image_id": image_id,
        "expected_registry_digest": registry_digest,
        "files": {name: sha(root / name) for name in ASSET_FILES},
        "datasets": {
            "official": {"revision": official_path.parent.parent.name, "sha256": sha(official_path)},
            "subset": {"revision": subset_path.parent.parent.name, "sha256": sha(subset_path)},
        },
    }
    manifest_path = root / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != manifest_text:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    written.append(manifest_path)
    return written


def image_tag_for(task_id: str) -> str:
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{task_id.replace('__', '_s_')}:latest"


def fetch_registry_digest(mirror: str, task_id: str) -> str:
    path = f"xingyaoww/sweb.eval.x86_64.{task_id.replace('__', '_s_')}"
    request = urllib.request.Request(
        f"https://{mirror}/v2/{path}/manifests/latest",
        headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json,"
                      "application/vnd.docker.distribution.manifest.list.v2+json"
        },
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        digest = response.headers["Docker-Content-Digest"]
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"mirror returned invalid digest for {task_id}: {digest}")
    return digest
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/test_assets_gen.py -q`
Expected: 2 passed

- [ ] **Step 5: prepare.sh 追加资产生成阶段**

在 `scripts/prepare.sh` 末尾（拉镜像完成之后）追加：

```bash
echo "[$(date -Is)] generating task assets" | tee -a "$LOG"
.venv/bin/python - "$PARQUET" <<'EOF'
import json, subprocess, sys
from pathlib import Path

import pyarrow.parquet as pq

from swe_agent.assets_gen import fetch_registry_digest, generate_task_assets, image_tag_for
from swe_agent.docker import SubprocessDockerClient

parquet = sys.argv[1]
rows = pq.read_table(parquet, columns=["instance_id"]).to_pylist()
client = SubprocessDockerClient()
official = Path("data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb/data/train-00000-of-00001.parquet")
for row in rows:
    task_id = row["instance_id"]
    tag = image_tag_for(task_id)
    inspected = client.run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], timeout_sec=30)
    if inspected.exit_code != 0:
        raise SystemExit(f"image missing in dedicated daemon, run prepare.sh pull first: {tag}")
    generate_task_assets(
        task_id=task_id,
        official_path=official,
        subset_path=Path(parquet),
        assets_dir=Path("assets/swegym"),
        image=tag,
        image_id=inspected.stdout.strip(),
        registry_digest=fetch_registry_digest("docker.1panel.live", task_id),
    )
    print(f"assets OK {task_id}")
EOF
```

- [ ] **Step 6: 对 100 个任务全量执行并抽查**

Run: `bash scripts/prepare.sh`（镜像已齐则秒过拉取阶段，进入生成）
Expected: 输出 100 行 `assets OK <task_id>`；`ls assets/swegym | wc -l` 为 101（含 moto-7023）
抽查: `GRPO_CONFIG=configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml bash scripts/qualify.sh` 仍全过

- [ ] **Step 7: Commit**

```bash
git add src/swe_agent/assets_gen.py tests/unit/test_assets_gen.py scripts/prepare.sh
git commit -m "feat: 新增任务资产生成器并接入 prepare 流程"
```

---

### Task 2: config 重设计（dataset 选任务、docker 删镜像字段）

**Files:**
- Modify: `src/swe_agent/config.py:21-42`
- Modify: `configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: 先改测试（失败）**

`tests/unit/test_config.py` 的契约测试追加：

```python
    assert config.dataset.tasks_dir.endswith("assets/swegym")
    assert config.dataset.task_ids is None or len(config.dataset.task_ids) >= 1
    assert not hasattr(config.docker, "image")
```

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: FAIL（`DatasetConfig` 无 `tasks_dir` / `DockerConfig` 仍有 `image`）

- [ ] **Step 2: 改 schema**

`src/swe_agent/config.py`：

```python
class DatasetConfig(StrictConfig):
    tasks_dir: str = Field(min_length=1)
    task_ids: tuple[str, ...] | None = None
    max_tasks: int | None = Field(default=None, ge=1)
    official_path: str
    official_revision: str = Field(min_length=40, max_length=40)
    subset_path: str
    subset_revision: str = Field(min_length=40, max_length=40)


class DockerConfig(StrictConfig):
    platform: Literal["linux/amd64"]
    pull_policy: Literal["never"]
    network_mode: Literal["none"]
    host_mounts: Literal[False]
    cpus: float = Field(gt=0)
    memory: str = Field(min_length=2)
    pids_limit: int = Field(gt=0)
    exec_timeout_sec: int = Field(gt=0)
    verifier_timeout_sec: int = Field(gt=0)
```

`load_config` 的 resolve 段把 `official_path/subset_path/assets_dir` 替换为 `tasks_dir`：

```python
            "dataset": config.dataset.model_copy(
                update={
                    "official_path": resolve_path(project_root, config.dataset.official_path),
                    "subset_path": resolve_path(project_root, config.dataset.subset_path),
                    "tasks_dir": resolve_path(project_root, config.dataset.tasks_dir),
                }
            ),
```

yaml 的 `dataset` 段改为：

```yaml
dataset:
  tasks_dir: assets/swegym
  task_ids: null        # null=全部；试跑可写 [getmoto__moto-7023, getmoto__moto-7212]
  max_tasks: null       # null=不截断；试跑可写 8
  official_path: data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb/data/train-00000-of-00001.parquet
  official_revision: bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb
  subset_path: data/swegym/SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd/data/train-00000-of-00001.parquet
  subset_revision: 3f22e68f673027edbaebe3424e4c20ae580563fd
```

`docker` 段删除 `image / expected_image_id / expected_registry_digest` 三行。

- [ ] **Step 3: 跑测试确认通过并全量回归**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: PASS（其余失败项允许在 Task 3-6 陆续修复，本步只要求 test_config 全绿）

- [ ] **Step 4: Commit**

```bash
git add src/swe_agent/config.py configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml tests/unit/test_config.py
git commit -m "feat: dataset 配置改为任务选择制并下沉镜像信息到任务资产"
```

---

### Task 3: swegym 多任务 TaskContext

**Files:**
- Modify: `src/swe_agent/swegym.py`
- Test: `tests/unit/test_swegym.py`

- [ ] **Step 1: 先改测试（失败）**

```python
def test_task_context_covers_all_selected_tasks() -> None:
    config, project_root, _ = load_config(CONFIG_PATH)
    context = load_task_context(config, project_root)
    assert len(context) >= 2
    sample, evaluation = context["getmoto__moto-7023"]
    assert sample.environment.image_name.endswith("getmoto_s_moto-7023:latest")
    assert "PIP_NO_INDEX=1" in evaluation.offline_eval_script
    assert len(build_training_dataset(context)) == len(context)
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_swegym.py -q`
Expected: FAIL（`build_training_dataset` 签名/context 长度）

- [ ] **Step 3: 实现**

`swegym.py` 要点（替换 `load_qualified_instance` / `load_task_context` / `build_training_dataset`）：

```python
def select_task_ids(config: ProjectConfig, project_root: Path) -> list[str]:
    tasks_dir = _resolve(project_root, config.dataset.tasks_dir)
    if config.dataset.task_ids is not None:
        selected = sorted(config.dataset.task_ids)
    else:
        selected = sorted(path.name for path in tasks_dir.iterdir() if path.is_dir())
    if config.dataset.max_tasks is not None:
        selected = selected[: config.dataset.max_tasks]
    if not selected:
        raise SWEGymContractError(f"no task assets found under {tasks_dir}")
    return selected


def load_task_instance(
    config: ProjectConfig, project_root: Path, task_id: str
) -> tuple[Sample, Evaluation]:
    official = _read_exact_row(_resolve(project_root, config.dataset.official_path), task_id)
    subset = _read_exact_row(_resolve(project_root, config.dataset.subset_path), task_id)
    eval_script = subset.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script.strip():
        raise SWEGymContractError(f"{task_id}: derived dataset is missing eval_script")
    assets_dir = _resolve(project_root, config.dataset.tasks_dir) / task_id
    offline = (assets_dir / "eval_script.offline.sh").read_text(encoding="utf-8")
    manifest = _read_object(assets_dir / "manifest.json")
    task = Task(
        task_id=task_id,
        repo_name=str(official["repo"]),
        base_commit=str(official["base_commit"]),
        problem_statement=str(official["problem_statement"]),
    )
    environment = Environment(
        environment_id=f"swegym:{task_id}",
        task_id=task_id,
        image_name=manifest["image_name"],
        expected_image_id=manifest["expected_image_id"],
        expected_registry_digest=manifest["expected_registry_digest"],
        workdir="/testbed",
        cpus=config.docker.cpus,
        memory=config.docker.memory,
        pids_limit=config.docker.pids_limit,
        exec_timeout_sec=config.docker.exec_timeout_sec,
        verifier_timeout_sec=config.docker.verifier_timeout_sec,
    )
    return Sample(task=task, environment=environment), Evaluation(offline_eval_script=offline)


def load_task_context(config: ProjectConfig, project_root: Path) -> TaskContext:
    return MappingProxyType(
        {
            task_id: load_task_instance(config, project_root, task_id)
            for task_id in select_task_ids(config, project_root)
        }
    )


def build_training_dataset(context: TaskContext) -> Dataset:
    return Dataset.from_list(
        [
            {"task_id": task_id, "prompt": build_prompt(sample.task)}
            for task_id, (sample, _) in context.items()
        ]
    )
```

`load_qualified_instance` 删除（引用处：`tests/unit/test_docker.py` 的 domain fixture、`verifier.py` 无引用、train.py 用 `load_task_context`——domain fixture 改为 `load_task_instance(config, root, "getmoto__moto-7023")`）。

- [ ] **Step 4: 跑测试并回归**

Run: `.venv/bin/python -m pytest tests/unit/test_swegym.py tests/unit/test_docker.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swe_agent/swegym.py tests/unit/test_swegym.py tests/unit/test_docker.py
git commit -m "feat: TaskContext 支持多任务并按任务资产构造环境"
```

---

### Task 4: train.py 集成（多行 dataset、逐行 prompt 校验、reward 透传 task_id）

**Files:**
- Modify: `src/swe_agent/train.py`
- Modify: `src/swe_agent/recording.py`
- Test: `tests/unit/test_train.py`、`tests/unit/test_recording.py`

- [ ] **Step 1: dataset 构建与 prompt 校验**

`_run_once` 中：

```python
        task_context = load_task_context(config, project_root)
        dataset = build_training_dataset(task_context)
        recorder.log(f"task context ready: {len(task_context)} tasks")
        for row in dataset:
            _validate_rendered_prompt_length(trainer, row["prompt"], config.chat.max_prompt_length)
```

删除原 `sample, _ = task_context[config.dataset.task_id]` 与单行 `prompt` 逻辑；`sandbox_factory`/`verifier_factory` 不变（沙箱按 env 自带的 sample 构造，已是逐任务）。

- [ ] **Step 2: recording 记录 per-group task_id**

`recording.py` 的 `begin_group` 增加参数：

```python
    def begin_group(self, prompt: object, rollout_count: int, *, task_id: str) -> list[Path]:
        ...
        self.batch = {..., "task_id": task_id, ...}
        self.group = {..., "task_id": task_id, ...}
```

（schema 字段名沿用现有 `"task_id"`，值从 `self.config.dataset.task_id` 改为参数传入。）

`_recording_reward` 改为从 kwargs 取 TRL 广播的 dataset 列：

```python
    def reward(prompts, completions, environments, task_id, **kwargs):
        # TRL 将 dataset 的 task_id 列按 completion 广播为 list[str]
        task_ids = list(task_id)
        if len(set(task_ids)) != 1:
            raise RecordingRuntimeError(f"group mixes tasks: {sorted(set(task_ids))}")
        recorder.begin_group(prompts[0], len(completions), task_id=task_ids[0])
```

注意：TRL reward kwargs 的 dataset 列名与值形态（单值 vs list）需在实现时以 `test_recording_reward` 的 FakeRecorder 断言锁定——测试里 `task_id=["t"]*4` 传入。

- [ ] **Step 3: 采样不变量更新**

`tests/unit/test_config.py` 契约测试的采样器行改为：

```python
    assert config.grpo.generation_batch_size // config.grpo.num_generations <= 100
```

（100 = subset 行数上限；多任务后约束来自数据集行数而非 1。）

- [ ] **Step 4: 跑测试并全量回归**

Run: `.venv/bin/python -m pytest tests/unit tests/integration/test_trl_interfaces.py tests/integration/test_trainer_loop.py -q`
Expected: 全绿

- [ ] **Step 5: Commit**

```bash
git add src/swe_agent/train.py src/swe_agent/recording.py tests/unit/test_train.py tests/unit/test_recording.py tests/unit/test_config.py
git commit -m "feat: 训练入口接入多任务数据集并按组记录任务身份"
```

---

### Task 5: qualify 覆盖全部任务

**Files:**
- Modify: `src/swe_agent/qualify.py`
- Test: `tests/unit/test_qualify.py`

- [ ] **Step 1: dataset/assets 检查改为循环全部选中任务**

```python
def check_dataset(config: ProjectConfig, project_root: Path) -> list[Check]:
    checks: list[Check] = []
    for task_id in select_task_ids(config, project_root):
        checks.extend(_check_dataset_one(config, project_root, task_id))
    return checks
```

（把现有单任务逻辑原样提取为 `_check_dataset_one`，`check_assets` 同理拆 `_check_assets_one`；docker.image 检查循环每个任务的 manifest 镜像。）

- [ ] **Step 2: 测试 + 全量 qualify 实测**

Run: `.venv/bin/python -m pytest tests/unit/test_qualify.py -q`，然后
`GRPO_CONFIG=configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml bash scripts/qualify.sh`
Expected: 全部任务 PASS（100+ 项检查），`== qualify 通过 ==`

- [ ] **Step 3: Commit**

```bash
git add src/swe_agent/qualify.py tests/unit/test_qualify.py
git commit -m "feat: qualify 检查覆盖全部选中任务"
```

---

### Task 6: 多任务试跑（8 任务 pilot）

**Files:**
- Modify: `configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml`（临时 `max_tasks: 8`）

- [ ] **Step 1: 小规模配置**

yaml：`dataset.max_tasks: 8`，`grpo.max_steps: 4`（其余不变）。采样器此时每组 unique prompt=1（`generation_batch_size == num_generations`），8 任务 × 重复采样。

- [ ] **Step 2: 启动 run 并观察**

Run: `CUDA_VISIBLE_DEVICES=0,1 scripts/grpo.sh`
观察点：
- `rollouts/batch-*/group.json` 的 `task_id` 出现 ≥2 个不同值；
- 每组 `rewards` 记录与 trajectory 落盘正常；
- 无新增 OOM/沙箱错误（不同任务镜像轮换创建容器）；
- `run.json.cleanup` 无残留容器。

- [ ] **Step 3: 恢复正式值并 Commit**

`max_tasks` 回 `null`、`max_steps` 回 `24`，commit 试跑结论到 notes（不动 notes.md 规则，由用户决定）。

---

## Self-Review 结论

- **Spec 覆盖**：资产生成(1)、配置(2)、加载(3)、训练集成(4)、资格(5)、试跑(6) 全覆盖；"镜像 digest 获取"在 Task 1 的 `fetch_registry_digest` 有着落。
- **Placeholder 扫描**：所有步骤含具体代码或具体命令；Task 6 属运行验证，给出明确观察点。
- **类型一致性**：`select_task_ids` / `load_task_instance` / `build_training_dataset(context)` / `begin_group(..., task_id=)` 签名跨任务一致；`task_ids: list[str]` 在 reward kwargs 的形态由 Task 4 Step 2 的测试锁定。
- **遗留风险**：① TRL reward kwargs 对 dataset 列的广播形态（单值/list）需在 Task 4 实现时以真实 trainer 行为复核，若 TRL 传的是去重值则改用 `environment.trajectory.task_id` 记录；② 100 任务资产首次生成时 digest 抓取受镜像站限速，失败任务需可重跑（生成器幂等已保证）。
