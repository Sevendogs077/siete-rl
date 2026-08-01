# SWE-bench Verified 训练后评测最小计划

> 状态：2026-07-31 修订。Verified 500 个 Docker 镜像已 500/500 就绪（见 §7.6）；
> 资产阶段完成，进入评测实施阶段。
>
> 目标：尽量不改变现有 GRPO 训练闭环，在本项目完整 Agent Loop 下评测训练后的
> LoRA 权重，并用 SWE-bench official harness 得到可用于论文或简历的结果。
>
> 2026-07-31 修订要点（经代码核查与外部仓库调研后收敛）：
>
> 1. 明确 scaffold 版本决策：评测固定使用评测时工作区当前代码的 prompt/工具/
>    parser 语义，不恢复训练时 scaffold（§5）；正式结论将在 scaffold 定稿后
>    重新训练并重新评测。
> 2. 新增 docker.py base-commit 校验与官方 TestSpec 镜像 HEAD 冲突的适配（§4.3）。
> 3. 明确 vLLM serving 方案：原生 `vllm serve --enable-lora`，同一 server 同时
>    serve base 与 LoRA adapter，`--eval-base` 边际成本为零（§4.4）。
> 4. 补充 official harness 离线化细节与空 patch 排除规则（§6）。
> 5. 更新镜像资产状态与阻塞项（§7.6、§11），新增实施门禁（§10）。

## 1. 最终决策

### 1.1 默认只评测 final

不建立 canonical base registry，也不为每个 run 自动重复评测 base。

默认行为：

```text
eval-base = False
只运行 final 或用户显式指定的 checkpoint
```

只有用户在命令行显式传入 `--eval-base True` 时，才在同一次评测中先运行该 run
对应的 base，再运行目标 candidate，并输出 base/final 对比。base 与 candidate
共用同一个 vLLM server（§4.4），base 评测不需要第二次启动推理服务。

评测仍记录模型、代码、prompt、sampling、数据和 harness revision，目的是让结果
可查阅，而不是建立自动协议校验系统。其中"代码"指**评测执行时工作区的
git commit 与 dirty 状态**（prompt/工具/parser 语义来源，见 §5），不是训练
run 的 code_commit。

最小输出：

```text
eval_outputs/
  runs/<training-run-id>/<candidate>/
    metadata.json
    predictions.jsonl
    official-report.json
    base/                 # 仅 --eval-base True 时存在
    comparison.json       # 仅 --eval-base True 时存在
```

`metadata.json` 的 schema 即评测协议契约，至少包含：training run 路径与
candidate、adapter 的 base_model_name_or_path、评测工作区 git commit/dirty、
sampling 参数与 seed、turn/completion/context 上限、数据集 parquet sha256、
harness revision、镜像 manifest sha256、vLLM serving 方式（adapter-serve 或
merge-serve）与数值一致性 smoke 结果。后续任何 eval 代码修改不得改变已有
metadata 的语义。

### 1.2 正式主张的边界

默认只跑 final 时，结果写成：

> 本项目 GRPO final 在本项目 Agent Loop 和既定推理预算下取得 Y%。

只有使用 `--eval-base True` 得到同配置 base 后，才能进一步写"相对 base 提升
Y-X 个百分点"。论文或榜单数字只作端到端系统参考，不能替代本地 base 来计算
本项目的 GRPO 因果增益。

注意：当前阶段评测使用评测时的工作区 scaffold（§5），而训练 rollout 使用的是
训练时的工作区 scaffold；两者可能因 scaffold 未冻结而不同。因此第一阶段的
base/final 差值严格说是"权重差 + scaffold 漂移"的混合效应，引用时必须注明。
该限制在 scaffold 定稿并重训后消除。

如果要单独证明 scaffold 的贡献，需要以后增加"同一权重、不同 scaffold"的
ablation；不属于第一阶段。

### 1.3 Verified 不能用于挑 checkpoint

- smoke task 只验证管线，不报告模型能力；
- 正式 Verified 500 之前预先指定 `final` 或一个 `checkpoint-N`；
- 不查看多个 checkpoint 的 Verified 结果后选择最高者；
- 开发期 checkpoint 选择应使用训练 reward、SWE-Gym holdout 或另建 dev 集。

## 2. 第一阶段范围

最小闭环：

```text
一个 Verified gold-patch/environment smoke
→ final 或预指定 checkpoint smoke
→ final 的 Verified inference
→ official grading
→ 输出 final 结果
→ 可选：--eval-base True 时再输出同次 base/final 对比
```

不建设：

- 通用评测平台；
- 独立 Docker asset manager；
- 自定义 Verified verifier；
- per-task `task.json/eval.sh/grader.json` 资产生成；
- 自动删除、prune 或镜像所有权数据库；
- 复杂 resume/retry/reporting 系统（official harness 自身支持已完成实例跳过，
  见 §6，不再另建）；
- 多模型或多 checkpoint 自动 sweep；
- 训练时 scaffold 版本恢复机制（评测固定使用当前工作区代码，见 §5）；
- 训练代码大规模重构。

## 3. 最小用户入口

目标接口：

```bash
# 默认只评测 final
CUDA_VISIBLE_DEVICES=2 scripts/eval.sh /absolute/path/to/run final

# 显式要求同一次额外评测 base
CUDA_VISIBLE_DEVICES=2 scripts/eval.sh \
  /absolute/path/to/run final --eval-base True

# 允许显式预指定 checkpoint；不自动选优
CUDA_VISIBLE_DEVICES=2 scripts/eval.sh /absolute/path/to/run checkpoint-24

# 单任务 plumbing smoke
EVAL_TASK_IDS=astropy__astropy-14539 \
CUDA_VISIBLE_DEVICES=2 \
scripts/eval.sh /absolute/path/to/run final
```

入口接受普通 `outputs/` 和 `_archive/` 中的真实绝对路径，不假设 run 一定位于
`outputs/<run-id>`。（已核实：`outputs/` 下近期 run 均无 adapter，带 final
adapter 的 run 全部位于 `outputs/_archive/`。）

候选解析：

- `final`：run **根目录**下的 adapter 文件对（`adapter_config.json` +
  `adapter_model.safetensors`）。已核实不存在 `final/` 子目录，训练
  `save_model` 直接写入 run 根目录；
- `checkpoint-N`：run 根目录下 `checkpoint-N/` 中的 adapter 文件对；
- `--eval-base True`：从目标 adapter 的 `adapter_config.json` 的
  `base_model_name_or_path` 解析 base，不猜测；
- `--eval-base` 默认严格为 `False`；
- 不存在或不完整时 fail closed；
- 不做临时 merged model，不修改训练 checkpoint。

GPU 使用约束：

- 评测只需 **1 张空闲 GPU**（vLLM server，无 trainer 进程、无 NCCL 权重同步）；
- 示例中的 `CUDA_VISIBLE_DEVICES=2` 仅作示意，实际以当时空闲卡为准；
- 必须避开正在训练占用的卡；四卡全部被训练占用时评测排队等待，不抢占；
- 容器侧按 `swe_agent.run_id` label 隔离（`docker.py:359-379`），eval 使用
  自己的 run_id，不会触碰训练容器。

## 4. Agent Loop 最小复用与推理服务

### 4.1 复用 `_tool_call_loop`

现有完整状态机位于 `SWEGRPOTrainer._tool_call_loop`
（`src/swe_agent/trainer.py:83-380`），包含：

- assistant 普通消息与 fixed fake user；
- tool-call parse error 恢复；
- bash/editor/finish 执行；
- termination 轮询；
- completion/context 超长处理；
- completion token IDs 与 tool mask；
- Loop exit reason。

第一阶段不提取 `agent_loop.py`。薄 evaluator 完成首次 generation 后直接复用现有
`_tool_call_loop`，不调用 `GRPOTrainer.evaluate()`，也不执行 reward、reference
model、logprob 或 loss 计算。

已核实的复用依据：

- `trainer.py` 不 import recording/reward，reward/ref/loss 均在 loop 之外；
- `num_generations` 与权重同步只出现在 `_generate_single_turn`，evaluator
  自实现 eval 版生成函数即可绕开；
- `tests/integration/test_trainer_loop.py:15-19` 已有
  `object.__new__(SWEGRPOTrainer)` 绕过全部 GRPO 初始化、手工注入依赖跑
  无 GPU loop 回归的先例，evaluator 照此模式复用。

实施前用固定 scripted generation 做 characterization test，比较：

- 完整 messages；
- tool call 及 observation 顺序；
- termination；
- completion token IDs；
- tool mask；
- parse-error/plain-message/finish/overlong 分支。

只有直接复用被实测证明不可维护时，才允许提取共享 Loop；届时最小边界只包含现有
状态机和它直接依赖的 action/suffix helper，训练入口、环境和 recorder 不随之重构。

### 4.2 环境与 patch 收集

当前环境 `_finalize()`（`environment.py:193-239`）会在
`termination=="submitted"` 且 patch 非空时立即调用 SWE-Gym verifier
（`environment.py:229-236`）。评测不应把 Verified grading 塞进这里；rollout
结束只收集 messages、termination 和 frozen patch，官方 grading 在所有
inference 完成后独立执行。

已核实的环境事实：

- `verifier_factory` 是惰性调用（仅在 `_finalize` 内），eval 不调用
  `_finalize` 即完全绕过 SWE-Gym verifier，无需 dummy verifier；
- `terminated` / `frozen_patch` / `trajectory` / `verification` 均已有公开
  property（`environment.py:66-82`），patch 冻结逻辑（`tools.py:75-78` →
  `docker.py:226-237` 的 `git add -N` + `git diff --binary`）可直接复用；
- 资源关闭无公开入口（`_close_rollout()`/`_close()` 为私有）。若确实需要，
  只允许增加一个 eval 专用的最小"关闭 rollout 并快照"公开接口，不改变训练
  `_finalize()` 路径。

### 4.3 官方镜像与 base-commit 校验冲突（新增，实施第一雷）

`DockerSandbox.open()` 的 `_verify_base_contract`（`docker.py:269-280`）硬性
要求容器 `HEAD == task.base_commit` 且 worktree 干净。而 §7.3 已实测官方
TestSpec 镜像的初始 HEAD 是镜像内名为 `SWE-bench` 的额外 prep commit（只改
`pyproject.toml`），其父提交才是任务 `base_commit`。

因此官方 Verified 镜像按现状会在 `open()` 直接抛 `ContainerCreateError`。
适配必须在 eval 侧完成，不改训练路径，候选方案（实施时二选一，取更小者）：

1. eval 专用 `DockerSandbox` 子类，放宽 `_verify_base_contract`：允许
   `HEAD^ == base_commit`（官方 prep commit 形态），其余校验保留；
2. 装载任务时以镜像实际 HEAD 构造 Task，`open()` 后立即在容器内
   `git checkout <base_commit>`，再进入 Agent Loop。

注意 SWE-bench official grading 与容器内 HEAD 无关（harness 新建容器、写入
patch 后跑 eval.sh），放宽 rollout 侧校验不影响 official grading 有效性。

另外 `inspect_image`（`docker.py:324-346`）要求 `expected_image_id` 精确匹配，
eval 装载任务时需先 `docker image inspect` 填充该字段（纯 eval 侧代码）。

### 4.4 vLLM serving：base 与 adapter 同 server 双列出（新增）

训练期的 `trl vllm-serve` **不能用于 eval**：已核实其脚本
（`trl/scripts/vllm_serve.py`）无 LoRA 支持，且用 `HfArgumentParser`
严格解析参数，无法透传 `--enable-lora`；训练期 LoRA 进 server 依赖 trainer
进程持有 PEFT 权重走 NCCL 同步，eval 没有 trainer 进程。

eval 使用原生 vLLM（项目依赖 vllm==0.22.1，已核实 `arg_utils.py` 含
`--enable-lora`）：

```bash
vllm serve <base_model_path> \
  --enable-lora \
  --lora-modules candidate=<adapter_path> \
  --max-lora-rank <adapter 实际 rank> \
  --gpu-memory-utilization <按空闲卡余量> \
  --port <port>
```

依据 [vLLM LoRA 官方文档](https://docs.vllm.ai/en/stable/features/lora.html)：

- `/v1/models` 同时列出 base model 与 adapter；请求时用 `model` 参数选择
  base 或 `candidate`。**同一 server 天然支持 §1.1 的 `--eval-base`：base
  评测零额外 serving 成本，且 base/final 共享完全相同的 server 配置**；
- `--max-lora-rank` 设为 adapter 实际 rank（从 `adapter_config.json` 读取），
  设太大浪费显存；
- 运行时动态加载 API（`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` +
  `/v1/load_lora_adapter`）本阶段不用，仅作将来 RL 迭代评测的备注。

数值一致性门禁（必须执行）：runtime LoRA 与 merge 参考系存在已记录的发散案例
（[vllm#47026](https://github.com/vllm-project/vllm/issues/47026)、
[vllm#17766](https://github.com/vllm-project/vllm/issues/17766)、
[vllm#38606](https://github.com/vllm-project/vllm/issues/38606)）。正式
inference 前做一次 smoke：同一组固定 prompt、temperature=0，对比
adapter-serve 输出与 HF+PEFT（merge 参考系）输出；若出现实质差异，改用
merge 后 serve（merge 是 W+BA 一次性舍入，且推理更快；代价是失去同 server
评 base 的便利，`--eval-base` 需第二次 serve base 权重）。smoke 结果写入
metadata.json。

注意：eval 推理栈（原生 vllm serve）与训练 rollout 推理栈（trl vllm-serve +
NCCL 同步）不是同一个，characterization test 锁的是 Loop 逻辑等价，锁不了两个
serving 栈的数值等价；这是本项目固有结构，以上 smoke 是唯一兜底，不为此统一
推理栈。

## 5. 配置继承规则（2026-07-31 修订）

**决策：评测固定使用评测时工作区当前代码的 prompt、工具和 parser 语义。**
不恢复训练 run 对应的 scaffold 版本（已核实：prompt/工具语义在代码
`prompts.py`/`tool_protocol.py` 中，不在 config；历史 run 实录的 system
prompt 与当前代码已不同）。理由：scaffold 尚未定稿，本阶段评测目标是验证
管线与获得参考结果；正式结论将在 scaffold 定稿后重新训练并以同一份定稿
scaffold 评测，届时 train/eval scaffold 一致性天然成立。metadata.json 必须
记录评测工作区的 git commit 与 dirty 状态，使每次评测的 scaffold 可追溯。

从 run 的 `config.yaml`（完整 resolved dump，可 `load_config()` round-trip）
继承：

- base/adapter/tokenizer provenance；
- sampling 参数与 seed；
- completion/context 上限；
- Agent Loop turns、protocol-error 上限；
- observation 截断和 Docker 资源上限。

不继承：

- optimizer、学习率和 scheduler；
- GRPO `num_generations`；
- KL、reward 配置；
- gradient accumulation/checkpointing；
- save/logging strategy；
- 训练双 GPU/vLLM 同步拓扑（eval 单 GPU 原生 vllm serve，见 §4.4）；
- 训练时的 prompt/工具/parser 代码版本（见上）。

每个 candidate 必须使用同一评测协议；禁止为表现较弱的模型单独增加 turns、
rollout 数或更换 sampling 参数。该纪律的落点是 metadata.json schema（§1.1），
任何协议变更体现为新 schema 版本，而不是隐式改代码。

## 6. 数据、环境和泄漏边界

固定目标：

```text
dataset: princeton-nlp/SWE-bench_Verified
split: test
instances: 500
HF revision: 91aa3ed51b709be6457e12d00300a6a596d4c6a3
parquet sha256: 43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21
harness revision: f7bbbb2ccdf479001d6467c9e34af59e44a840f9
```

固定 parquet 已持久化到：

```text
data/swegym/SWE-Bench__SWE-bench_Verified/
  91aa3ed51b709be6457e12d00300a6a596d4c6a3/
  data/test-00000-of-00001.parquet

size: 2,090,470 bytes
sha256: 43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21
```

本地核验结果：

- Verified 为 500 个唯一实例；
- 500 个实例均有 `test_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS`；
- 当前 SWE-Gym 2,438 与 Verified 的 instance ID、repo、problem statement、
  base commit 精确交集均为 0；
- 当前 SkyRL 100-task 子集与 Verified 的 instance ID/repo 交集为 0；
- 491 条本地 OpenHands SFT trajectory 对应 SWE-Gym，未发现 Verified 精确文本交集。

这只能证明当前可见训练数据没有直接交集，不能证明 base model 预训练阶段不存在
benchmark contamination。

传给模型的 public task 内容不得包含：

- gold patch；
- test patch；
- `FAIL_TO_PASS`、`PASS_TO_PASS`；
- official eval script 或 grader 结果。

SWE-Gym/SkyRL 与 SWE-bench Verified 是两套环境资产：

- 本项目 SkyRL 100-task：`xingyaoww/...` 镜像和本地 `eval_script`；
- Verified：official harness 的 `swebench/sweb.eval...` TestSpec 镜像和官方 grading。

二者可以存放在同一个 SWE-Gym 专用 Docker daemon，但不能把 SWE-Gym verifier
当作 Verified official grading。

### 6.1 official harness 离线化与调用细节（已按 pinned revision 源码核实）

以下行为均已对照 harness revision `f7bbbb2` 的
`run_evaluation.py` / `utils.py` 源码确认：

- **daemon 指向**：harness 用 `docker.from_env()` 建 client，通过环境变量
  `DOCKER_HOST=unix:///run/docker-swegym/docker.sock` 指向专用 daemon，
  不修改任何 daemon 配置；
- **数据集离线**：`load_swebench_dataset` 直接接受本地 `.json/.jsonl/.parquet`
  路径，`--dataset_name` 传 §6 的固定 parquet 绝对路径即可全程离线，不访问
  Hugging Face；
- **镜像复用**：`run_instances` 对本地已存在的 instance image 打印
  "Found N existing instance images. Will reuse them."并跳过 pull；image key
  必须与 official 规则逐字符一致（`swebench/sweb.eval.x86_64.<id小写、__→_1776_>:latest`），
  §7 的 retag 已按此规则完成并经 500/500 核对；
- **镜像保留**：`--cache_level instance --clean false`。源码语义：clean=False
  时只删除"本次新建且高于 cache level"的镜像，已存在镜像一律保留；
  cache_level=instance 时 instance image 处于保留层级。该组合不删除任何
  已有镜像，与 OpenHands `eval_infer.sh` 用法一致；
- **空 patch 排除**：`model_patch` 为 `""` 或 `None` 的 prediction 会被
  `get_dataset_from_preds` 静默排除出评测并计入未解决。predictions.jsonl
  生成时必须保证 patch 提取兜底（finish 未触发或 patch 为空时仍写出记录并
  显式计数），空 patch 数量写入 metadata.json；
- **predictions 格式**：每行 JSON，字段 `instance_id`、
  `model_name_or_path`、`model_patch`；
- **grading 断点续跑**：已有 `report.json` 的实例默认跳过
  （exclude_completed），grading 中断后重跑不重复已完成的实例；
- **资源参数**：`--max_workers` 默认 4（官方建议 ≤ 75% CPU 核数），
  `--timeout` 默认 1800s/instance；500 实例 grading 的墙钟时间按
  `500 × 平均测试时长 / workers` 估算，实施时按机器核数设定 workers；
- grading 本身（log 解析 + 容器内跑 eval.sh）完全离线。

## 7. Docker 资产准备与缓存

### 7.1 指定 daemon 和实际存储位置

唯一允许使用的 daemon：

```text
socket: unix:///run/docker-swegym/docker.sock
Docker Root Dir: /home/2025user/zyp/.docker-swegym
storage driver: overlay2
backing filesystem: /home/2025user 所在大容量 ext4 磁盘
```

`/run/docker-swegym/docker.sock` 只是 socket；镜像 layer 实际写入
`/home/2025user/zyp/.docker-swegym`，不会进入系统默认 `/var/lib/docker`。

不得使用默认 Docker daemon。所有 Docker 命令都必须显式带：

```bash
docker -H unix:///run/docker-swegym/docker.sock ...
```

official harness 通过 `DOCKER_HOST` 环境变量指向同一 daemon（§6.1）。

### 7.2 smoke task 与准确镜像

smoke task：

```text
instance_id: astropy__astropy-14539
repo: astropy/astropy
base_commit: c0a24c1dc957a3b565294213f435fefb2ec99714
TestSpec image:
swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest
platform: linux/x86_64
```

镜像名由固定版本 official `make_test_spec()` 实际生成，不手工推导。

### 7.3 镜像站下载

使用大陆 Docker Hub 镜像中转站。专用 Docker daemon 的 `docker info` 未显示
HTTP/HTTPS proxy，Docker CLI 通过 Unix socket 请求该 daemon，因此全量下载不修改、
不清除用户 shell 中的代理变量，由无代理 daemon 直接访问镜像站。

实测情况：

- `docker.1panel.live`：smoke 镜像下载成功，但后续任务差异层反复超时；
- `dockerproxy.net`：同一失败任务实测完整下载成功，作为全量下载源；
- 两者均采用显式改写后的完整 Docker Hub namespace/path，不修改 daemon 配置。

文档来源：[1Panel 镜像加速说明](https://1panel.cn/docs/v2/user_manual/containers/setting/)；
[Docker Proxy 使用文档](https://dockerproxy.net/docs)。

全量下载命令使用：

```bash
docker -H unix:///run/docker-swegym/docker.sock pull \
  dockerproxy.net/swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest
```

下载完成后添加 official tag，避免 harness 再访问 Docker Hub：

```bash
docker -H unix:///run/docker-swegym/docker.sock tag \
  dockerproxy.net/swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest \
  swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest
```

保留 mirror tag 和 official tag；二者引用同一 image layers，不会复制一份镜像数据。

2026-07-30 实测结果：

```text
首次 smoke pull source:
docker.1panel.live/swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest

pull digest:
sha256:a8d0f9829ec24dfb23a2f0097a245ee60faf1b396b33b3af5c22d7ac5f3c00ab

local image ID:
sha256:290a743498af81faf833324ccb3dfaf877e1d4fdd60594efc1a5f4835601316e

local uncompressed size:
2,695,538,814 bytes（约 2.70 GB）
```

mirror tag 与 official tag 已通过 `docker image inspect` 确认为同一 image ID。
第一次 1Panel smoke pull 曾使用命令级 `env -u`，其作用域仅限该子进程，没有修改用户
shell 或任何代理配置。后续 Verified 500 全量 pull 不再操作这些变量；专用 daemon
的 `docker info` 未显示 daemon 级 HTTP/HTTPS proxy。

使用 official tag 运行 `--network none --rm` 容器成功：

```text
Python 3.9.20
/testbed exists
image initial HEAD: c06bee2ac1f505eb6530511662ce4695a69003eb
HEAD parent: c0a24c1dc957a3b565294213f435fefb2ec99714
```

初始 HEAD 是镜像内名为 `SWE-bench` 的单独 commit，只修改 `pyproject.toml`
两行中的一行；它的父提交正是任务 `base_commit`。因此镜像内容包含准确任务基线，
但 gold patch 和官方 eval script 尚未实际执行，不能只凭容器启动宣称 grading
已闭环。（该 HEAD 形态与 `docker.py` base-commit 校验的冲突及适配见 §4.3。）

### 7.4 全量 Verified 资产

本阶段直接下载 Verified 500 个 TestSpec instance image，不再采用 smoke 后惰性下载。

镜像名按固定 official harness 的 `TestSpec.instance_image_key` 生成：

```text
swebench/sweb.eval.x86_64.<lowercase-instance-id>:latest
并将 instance ID 中的 "__" 替换为 "_1776_"
```

本地 Verified parquet 实测为 500 个任务、500 个唯一 official image key。

固定 image manifest：

```text
path:
data/swegym/verified_pull/images-x86_64.txt

lines:
500

sha256:
b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a
```

每个任务执行：

```text
1. 从 dockerproxy.net/swebench/... pull；
2. 在专用 daemon 中保留 mirror tag；
3. 添加 swebench/... official tag；
4. 已存在 official tag 时跳过，以便中断后继续；
5. 使用 12 路并发；单个镜像失败最多重试 5 次；
6. 批次失败后从已有 official tag 继续，不重复下载完成项。
```

不修改 daemon 配置，不使用默认 Docker daemon，不删除已有 SWE-Gym 镜像。

2026-07-30 全量下载运行状态：

```text
tmux session:
swebench_verified_pull

log:
data/swegym/verified_pull/pull.log

pass-end count:
data/swegym/verified_pull/ready_count.txt

final status:
data/swegym/verified_pull/status.txt

启动 detached session 时:
21/500 official tags ready
```

普通 `nohup` 子进程会被当前受管执行环境回收，因此最终改用 detached tmux。

**2026-07-31 完成确认（已核实）**：下载于 2026-07-31 04:04 (+0800) 在第 1 个
pass 即全部拉齐，`status.txt=complete`、`ready_count.txt=500`；manifest sha256
与上述固定值一致；manifest 500 行与本地 official tag 逐一比对零缺失；下载
进程已正常退出（tmux session 已不存在）。

### 7.5 本地镜像保留方式

这里没有额外缓存目录或新管理器。"缓存"就是：

1. 镜像通过专用 daemon 下载一次；
2. layer 保留在 `/home/2025user/zyp/.docker-swegym`；
3. base/final 每次创建 fresh container，但复用本地只读 image layers；
4. official harness 使用 `--cache_level instance --clean false`，不在每次 grading 后
   删除 instance image（语义已按源码核实，§6.1）；
5. 不自动 prune，不删除已有 SWE-Gym 镜像。

这与本地 SWE-Gym 的基本使用方式一致：镜像常驻，任务运行时创建/销毁容器。

本阶段已下载全部 500 个 Verified instance image。后续所有 final，以及显式
`--eval-base True` 的 base，共用这些本地镜像。

mirror tag 清理：daemon 中残留 498 个 `dockerproxy.net/swebench/...` 别名
tag，与 official tag 共享 layer。允许（但不要求）用 `docker rmi` 仅删除这些
mirror 别名 tag——删 tag 不删 layer，不影响 official tag 与 SWE-Gym 镜像；
磁盘余量充足（分区剩余约 3.1T），不紧急。

### 7.6 资产确认清单

- [x] 专用 daemon Root Dir 已确认；
- [x] daemon 未显示 HTTP/HTTPS proxy 配置；
- [x] TestSpec 和准确镜像名已确认；
- [x] Verified 500-row parquet 已持久化并通过 SHA256 校验；
- [x] 镜像站 pull 完整成功；
- [x] mirror tag 已添加 official tag；
- [x] image inspect 确认两个 tag 指向同一 image ID；
- [x] `--network none` 容器启动，并确认 `/testbed` HEAD 的父提交为任务 base commit；
- [x] 500/500 个 mirror image 下载完成（2026-07-31 04:04）；
- [x] 500/500 个 official tag 存在（与 manifest 零缺失比对）；
- [x] 核对缺失、失败和本地总占用（缺失 0；daemon 总占用约 280GB，含 498 个
  可回收 mirror 别名 tag；分区剩余约 3.1T）；
- [ ] official gold-patch 单任务 grading 成功。

## 8. 论文和发布基线备忘

### 8.1 SWE-Gym 论文

来源：[Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/html/2412.21139v2)

统一条件：

- benchmark：完整 SWE-bench Verified 500；
- scaffold：OpenHands CodeActAgent 2.1；
- 工具：bash + editor，browser disabled；
- evaluation temperature：0；
- 上限：100 interaction turns 或 32k context；
- SFT 数据：491 条 SWE-Gym 成功轨迹；
- base：Qwen2.5-Coder-Instruct；
- 下表普通 zero-shot/fine-tuned 是首次单 rollout resolve rate，不是 verifier best-of-N。

| 模型大小 | zero-shot | SWE-Gym SFT | 绝对提升 |
|---|---:|---:|---:|
| 7B | 1.8% ± 1.1 | 10.6% ± 2.1 | +8.8 |
| 14B | 4.0% ± 1.6 | 16.4% ± 2.0 | +12.4 |
| 32B | 7.0% ± 1.3 | 20.6% ± 2.1 | +13.6 |

论文另报：

| 系统 | Verified |
|---|---:|
| 32B SWE-Gym Agent + learned verifier | 32.0% |

32.0% 使用多候选和 verifier 进行 inference-time scaling，不能与单 rollout 的
10.6%、11.0%、14.6% 或本项目 pass@1 直接比较。

`SWE-Gym/OpenHands-7B-Agent` Hugging Face 仓库存在，但当前模型卡没有记录成绩；
10.6% 的原始成绩证据来自论文，不来自模型卡。

### 8.2 SkyRL-v0

来源：[SkyRL 固定发布 README](https://raw.githubusercontent.com/NovaSky-AI/SkyRL/a0d50c482436af7fac8caffa4533616a78431d66/README.md)

| SkyRL 模型 | Base | Base Performance | RL 后 |
|---|---|---:|---:|
| SkyRL-Agent-7B-v0 | OpenHands-7B-Agent | 11.0% | 14.6% |
| SkyRL-Agent-8B-v0 | Qwen3-8B no thinking | 3.6% | 9.4% |
| SkyRL-Agent-14B-v0 | Qwen3-14B thinking | 18.0% | 21.6% |

README 声称 benchmark 为 SWE-Bench Verified。固定 OpenHands 子模块默认
CodeActAgent、`N_RUNS=1`、最大 100 iterations，但发布结果的完整执行命令、
decoding 参数和精确模型 revision 没有充分归档。因此：

- 11.0% → 14.6% 可作为 SkyRL 作者发布内的 RL 对照；
- 对本项目只能作为端到端系统参考；
- 不能把 14.6%-10.6% 解释为 SkyRL RL 相对 SWE-Gym SFT 的纯权重提升；
- 目前未在 SWE-bench official experiments tree 中确认相应公开 submission。

### 8.3 官方排行榜

[SWE-bench Verified 官方页面](https://www.swebench.com/verified.html) 明确说明：

- Full leaderboard 混合任意 agent、RAG、multi-rollout、review 等系统；
- Bash Only 使用统一 mini-SWE-agent，才更接近模型横向比较；
- 即使同为 mini-SWE-agent，不同 major release 也不一定可比。

因此官方榜单值可作为系统级参考，但每个引用都必须记录 agent/scaffold、版本、
rollout 数、预算和 selection/verifier 规则。不得只抄百分比。

## 9. 最小文件改动预算

预计第一阶段最多：

| 文件 | 动作 | 用途 |
|---|---|---|
| `scripts/eval.sh` | 创建 | 薄 Bash 入口（`exec .venv/bin/python -m swe_agent.eval "$@"`） |
| `src/swe_agent/eval.py` | 创建 | candidate 解析、Verified 任务装载、sandbox base-contract 适配、纯 inference、prediction 输出、official harness 调用、vLLM server 生命周期与 raw token 客户端 |
| `tests/test_eval.py` | 创建 | CLI 默认值、candidate、Loop characterization 和 smoke |
| `src/swe_agent/environment.py` | 尽量不改；必要时只加一个接口 | 关闭 eval rollout 并取得快照 |

对 eval.py 体量的现实预期：它单文件承载四块新逻辑（§4.3 的 sandbox 适配、
Verified 任务装载——`swegym.py` 的 TaskContext 是 SWE-Gym 专用无法复用、
harness 调用、vLLM serving/客户端），不会很"薄"，预计数百行量级。保持单文件
是为了克制模块数量，不为守住"薄"砍功能。

明确不创建 `eval/` 多模块目录、`docker_assets.py`、`verifier.py`、
`reporting.py`、`prepare_eval.sh` 或共享 `agent_loop.py`。

## 10. 实施顺序与门禁

1. ~~完成 Verified 500 个镜像及 official tag~~（已完成，§7.6）。做一个离线
   容器/gold smoke。
2. 实现薄 evaluator（含 §4.3 sandbox 适配与 §4.4 vLLM serving），并锁定
   Loop messages/token/mask/termination 等价性。
3. 默认只评测目标 run 的 final 或预指定 checkpoint。
4. 进行正式 Verified inference 和 official grading，输出 final 与外部参考表。
5. 仅当用户传入 `--eval-base True` 时，在同一次命令中额外评测 base 并输出差值。

进入完整 Verified 500 前必须满足：

- smoke gold patch resolved（official harness 对 gold prediction 判 resolved）；
- smoke task 容器经 eval 适配路径 `open()` 成功（§4.3 base-contract）；
- LoRA 数值一致性 smoke 通过：adapter-serve 与 HF+PEFT merge 参考系在
  temperature=0、同一组固定 prompt 下输出一致，或已改用 merge-serve 并在
  metadata 中记录（§4.4）；
- `--eval-base` 未提供时不运行 base；
- `--eval-base True` 时 base/final 使用同一 server、同一评测配置（§4.4）；
- 模型输入无 private grader 字段；
- predictions.jsonl 无静默空 patch：空 patch 显式计数并写入 metadata（§6.1）；
- harness 全程离线：本地 parquet + `DOCKER_HOST` 指向专用 daemon + 本地镜像
  复用（§6.1）；
- 不从 Verified 选择 checkpoint；
- official harness 直接产生 report；
- 评测使用 1 张空闲 GPU，不抢占训练卡；专用 Docker daemon 得到确认；
- 不影响训练进程和其他人的 GPU/Docker 容器。

## 11. 当前阻塞项

- ~~Verified 500 个 Docker 镜像下载~~：已完成（2026-07-31 04:04，500/500，
  manifest 零缺失，§7.6）；
- 当前服务器上 official gold-patch grading 尚未成功（§10 门禁第一条）；
- 单 GPU 原生 vLLM base + LoRA adapter 的 raw-token Loop 尚未 smoke（§4.4，
  含数值一致性门禁）；
- §4.3 的 base-commit 校验适配尚未实现，官方镜像按现状无法通过
  `DockerSandbox.open()`；
- 必须指定一个已完成、可作为正式目标的 run：现有 `_archive/` 中全部已完成
  run（含 `20260726T130807Z-b5db`）的 base 均为 Qwen2.5-Coder-7B-Instruct，
  不是 OpenHands-7B-Agent；首批 OpenHands-7B-Agent base 的 run 仍在训练中。
  b5db 可用于全流程管线 smoke；
- base model 预训练污染无法由本项目本地数据审计排除；
- 评测时间表受训练排队约束：四张 GPU 均可能被训练占用，评测需等待空闲卡
  （§3 GPU 使用约束）。
