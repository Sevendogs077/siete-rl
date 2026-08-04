# SWE-bench Verified 训练后评测实施计划

> 状态：2026-08-04 收敛修订。Verified 500 个 Docker 镜像已 500/500 就绪；
> 当前只实施独立评测闭环，不扩建通用评测平台，不修改训练逻辑。

## 0. 唯一目标与完成定义

唯一目标：使用本项目 Agent Loop 和 SWE-bench official harness 评测一个已指定
训练 run 的 final LoRA，产出可复核的 SWE-bench Verified 结果，并按用户选择与
同协议本地 base 及此前记录的公开 baseline 放在同一份比较结果中。

计划完成且全部门禁通过后，可以准确地说：

> 项目新增了一个独立于训练闭环的 post-training eval 系统；它能在固定评测协议下
> 运行目标 LoRA，并可选运行对应 base，收集 patch，调用固定 revision 的
> SWE-bench official harness，并输出 resolved 数和 resolve rate；运行 base 时再输出
> base/final 差值。

结果主张分两层：

1. **严格主比较**：同一次 evaluator、同一任务集、同一 scaffold、同一预算和同一
   serving 协议下的本地 base 与 final。可以报告 `final - local_base`，表述为
   “该 LoRA 权重在本项目固定 eval 协议下相对其 base 的差值”。
2. **外部参考比较**：SWE-Gym、SkyRL 等已记录数字。由于 scaffold、预算或发布信息
   不完全相同，只能并列表述为端到端系统参考，不声称严格 apples-to-apples，也不
   用其差值冒充本项目 LoRA 的因果增益。

在完整 500 任务 inference 和 official grading 完成前，只能说“实现并通过 smoke”，
不能提前声称已经得到有效的完整 benchmark 结果。

## 1. 对现有代码库和训练逻辑的边界

### 1.1 允许的文件变化

第一阶段只新增：

| 文件 | 用途 |
|---|---|
| `scripts/eval.sh` | 薄入口；以项目环境启动 evaluator |
| `src/swe_agent/eval.py` | run 解析、Verified 任务装载、eval-only sandbox/environment 适配、vLLM 生命周期、Agent inference、prediction 与 official harness 调用 |
| `tests/test_eval.py` | 纯 eval 单元测试和 characterization test |

计划文件本身之外，不修改现有训练文件。特别禁止修改：

- `src/swe_agent/train.py`；
- `src/swe_agent/trainer.py` 中的训练 Loop；
- `src/swe_agent/config.py` 的训练 schema；
- reward、verifier、recording、supervisor 和现有训练脚本；
- `src/swe_agent/docker.py` 的训练期 sandbox 合同。

eval 通过导入和复用既有能力工作，不实例化完整 `GRPOTrainer`，不创建 optimizer、
reference model、reward 计算或 NCCL 同步。评测使用独立进程、run 内独立的 `evals/<eval-timestamp>/`
输出子目录、独立 Docker run label 和一张明确空闲的 GPU；四张卡都在训练时等待，不
抢占训练资源。

### 1.2 为什么不会破坏训练逻辑

- 新入口不被 `scripts/grpo.sh`、`src/swe_agent/train.py` 或训练 supervisor 调用；
- eval-only Docker 适配定义在 `eval.py`，不放宽训练期 base/image 校验；
- eval-only environment 不调用训练用 `_finalize()`，因此不会调用 SWE-Gym verifier、
  不生成 reward；它只快照 patch/termination 并关闭 rollout container；
- 复用 `_tool_call_loop` 仅作为现有方法的调用者，不修改其实现；
- 新增回归门禁要求现有 unit/integration tests 全部继续通过。

因此，按本计划实施不会改变训练语义。实际风险集中在 eval 自己失败或资源未清理，
对应门禁见 §7；这些失败必须 fail closed，不能影响正在运行的训练。

## 2. 最小用户入口与输出

第一阶段只支持 run 根目录中的 `final` adapter，不做 checkpoint 自动发现、选优或
多模型 sweep。入口接受 `outputs/` 或 `_archive/` 中的真实绝对 run 路径：

```bash
# 单任务 plumbing smoke
EVAL_TASK_IDS=astropy__astropy-14539 \
CUDA_VISIBLE_DEVICES=2 \
scripts/eval.sh /absolute/path/to/run

# 默认只评测 final
EVAL_BASE=False CUDA_VISIBLE_DEVICES=2 scripts/eval.sh /absolute/path/to/run

# 显式评测 local base 和 final
EVAL_BASE=True CUDA_VISIBLE_DEVICES=2 scripts/eval.sh /absolute/path/to/run
```

run 根目录必须同时存在 `adapter_config.json` 与 `adapter_model.safetensors`；base 只从
`adapter_config.json.base_model_name_or_path` 解析，不猜测、不建立 registry、不修改
训练 checkpoint。

`EVAL_BASE` 默认严格为 `False`，只接受字符串 `True` 或 `False`，其他值立即报错。
`False` 时只评测 final；`True` 时先评测 local base，再评测 final。单任务 smoke
遵循相同开关，不隐式增加 base 评测。

最小输出：

```text
<run-root>/evals/<eval-timestamp>/
  metadata.json           # 会话级协议：工作区 commit/dirty、数据/harness revision、
                          # turn/context 预算、sampler、engine 参数、任务顺序、GPU
  base/                   # 仅 EVAL_BASE=True 时存在
    metadata.json         # serve 方式（shared-LoRA-engine 或 base-only fallback）
    predictions.jsonl
    official-report.json
  candidate/
    metadata.json         # runtime LoRA 或 merged fallback、adapter parity 结果
    predictions.jsonl
    official-report.json
  comparison.json
```

`<run-root>` 是被评测训练 run 的根目录（`outputs/` 或 `_archive/` 下）。eval 只在 run
内新增 `evals/` 子目录，不改动任何训练产物；每次评测生成新的 `<eval-timestamp>`
目录，旧评测结果不可变，不覆盖、不选优。临时 merged model 等中间产物也只能位于
该次评测的 `candidate/` 下。

`comparison.json` 始终包含 final 结果和 §6 的外部参考表。`EVAL_BASE=True` 时再包含
base 的 evaluated、resolved、resolve rate、绝对百分点差和逐任务
`base_only/final_only/both/neither` 计数；`False` 时明确记录 `local_base=not_run`，
不得生成本地差值。基础设施失败单独记录，不得计为 unresolved，也不得从分母静默删除。

## 3. 固定评测协议与可比性

### 3.1 本地 base/final 协议

`EVAL_BASE=True` 时，base 与 final 必须固定：

- 同一 Verified 500 数据 revision 和任务顺序；
- 同一评测工作区 commit/dirty 状态，即同一 prompt、工具、parser；
- 同一 tokenizer、context/completion 上限、turn 上限和 Docker 资源上限；
- 同一 vLLM 版本、sampler backend、engine 参数和请求并发度；
- `temperature=0`，采用 greedy decoding。

评测 sampling 不再继承训练时 `temperature=1.0`。训练 sampling 是优化过程配置，
不是评测协议；greedy evaluation 能减少在线 serving 的采样随机性，也与已记录的
SWE-Gym 公开 baseline 更接近。turn/context 上限仍使用目标 run 的已解析配置，并在
metadata 和外部对比中明确披露；不为 final 单独增加预算。配置的读取来源与缺失
行为由实测决定，见 §8.1 第 1 条。

vLLM [默认不承诺跨进程、跨批次的逐 token 完全可复现](https://docs.vllm.ai/en/v0.22.0/usage/reproducibility/)。因此固定 seed 仍记录为输入，
但不把它描述为强一致性保证；运行 base 时固定任务顺序和并发度，禁止 base/final
使用不同调度配置。

### 3.2 scaffold 边界

base 与 final 都使用评测执行时的当前工作区 scaffold，因此两者差值不是
“权重差 + scaffold 漂移”的混合量；它是在同一当前 scaffold 下的权重对比。

训练 rollout 可能使用较早的 scaffold，这会限制“训练过程完全复现”的主张，但不
破坏 `EVAL_BASE=True` 时的 base/final 同协议比较。metadata 必须记录评测工作区
git commit 和 dirty 状态，不恢复历史 scaffold，也不为此修改训练代码。

### 3.3 checkpoint 纪律

正式 Verified 500 前由用户指定唯一 run，评测代码不浏览多个 checkpoint 后选最高
结果。smoke 只验证管线，不用于报告能力。将来如需 checkpoint 评测，另行显式扩展，
不在第一阶段预建通用 candidate resolver。

## 4. 实现设计

### 4.1 复用现有 Agent Loop

现有完整状态机位于 `SWEGRPOTrainer._tool_call_loop`，已经处理普通 assistant
message、fixed fake user、tool parse error、bash/editor/finish、termination、长度上限
和 Loop exit reason。evaluator 只构造该方法需要的最小适配对象，并将单轮生成代理到
原生 vLLM OpenAI-compatible endpoint；不调用 `GRPOTrainer.evaluate()`。

实施前使用 scripted generation 做 characterization test，比较 eval 与现有 Loop 的：

- 完整 messages；
- tool call 与 observation 顺序；
- plain-message、parse-error、finish、iteration/context 上限分支；
- termination 和最终 patch。

训练专用的 logprob、reward、advantage、completion token mask 不属于 eval 成果，不为
它们建设第二套记录逻辑。只有直接复用被实测证明不可行时才重新评估共享 Loop 抽取；
第一阶段不重构 `trainer.py`。

### 4.2 eval-only environment 与容器收束

训练环境 `_finalize()` 会调用 SWE-Gym verifier，不适用于 Verified。`eval.py` 定义
eval-only environment 适配：复用 `reset()` 和三个工具，但结束时只读取公开的
`terminated`/`frozen_patch` 以及 Loop exit reason，关闭 rollout container，然后把
patch 交给 official harness。不得使用 dummy verifier，也不得把 official grading
塞进训练环境。

每个任务无论成功、模型终止、超长或异常都必须进入 `finally` 调用现有
`DockerSandbox.close()`；eval 最外层 `finally` 和 `atexit` 直接复用现有
`sweep_run_containers(client, eval_run_id)`，按独立 `swe_agent.run_id` label 清扫残留，
不实现第二套 sweep。清理失败报告为 infrastructure failure 并保留可追踪容器 ID，
不伪装成模型 unresolved。

### 4.3 Verified official image 适配

官方 TestSpec 镜像初始 `HEAD` 是额外 prep commit，其父提交才是任务
`base_commit`；现有训练 sandbox 要求 `HEAD == base_commit`，不能直接使用。

适配只在 `eval.py` 的 `DockerSandbox` 子类中完成：

1. 要求镜像初始 `HEAD == base_commit` 或 `HEAD^ == base_commit`；其他形态 fail closed；
2. 若为 prep commit，进入 Agent Loop 前 checkout 到准确 `base_commit` 并再次断言
   worktree clean；
3. `inspect_image` 仍精确核对本地 image ID、linux/amd64 和已记录 digest；
4. 不修改 `docker.py`，不放宽训练 sandbox 合同。

模型输入只能包含 public problem statement，不得包含 gold patch、test patch、
`FAIL_TO_PASS`、`PASS_TO_PASS`、official eval script 或 grader 结果。

### 4.4 vLLM、LoRA 与 local base

依据 [vLLM LoRA 官方文档](https://docs.vllm.ai/en/v0.22.1/features/lora/)，同一个
原生 vLLM 0.22.1 服务可以通过请求的 `model` 字段分别路由 base 和 runtime
LoRA adapter；它是两组独立请求，不是一次请求双输出。共享服务避免第二次加载 base
权重，但 base 仍增加一整套 Agent inference、Docker rollout 和 grading 成本，不能再
表述为“边际成本为零”。

启动必须使用项目环境并保证子进程能找到项目中已锁定的 `ninja`：

```bash
uv run --no-sync vllm serve <base_model_path> \
  --enable-lora \
  --lora-modules candidate=<adapter_path> \
  --max-lora-rank <adapter 实际 rank> \
  --gpu-memory-utilization <按空闲卡余量> \
  --port <port>
```

`scripts/eval.sh` 同样通过 `uv run --no-sync` 启动 evaluator。evaluator 若直接创建
vLLM 子进程，必须把 `Path(sys.executable).parent` 前置到子进程 `PATH`，并在分配 GPU
前执行：

```text
vllm --version
ninja --version
torch.utils.cpp_extension.is_ninja_available() == True
```

当前 `uv.lock` 和 `.venv` 已包含兼容的 `ninja==1.13.0`，不新增依赖、不运行
`pip install`、不运行 `uv add ninja`、不安装系统级 ninja。默认保留 FlashInfer sampler；
不得用 `VLLM_USE_FLASHINFER_SAMPLER=0` 静默绕过启动问题。若未来必须更换 sampler，
它构成新的 eval 协议；运行 base 时必须同时作用于 base/final，并重新执行下述
adapter 资格检查和记录。

正式 inference 前执行固定 prompt、`temperature=0` 的 adapter 门禁：

**adapter parity**：runtime LoRA 与 HF+PEFT/merged 参考输出一致。

adapter parity 通过后，final 使用 runtime LoRA；`EVAL_BASE=True` 时，base 和 final
使用同一 LoRA-enabled server，请求分别选择 base model 和 `candidate`。local base 的
准确名称是“shared-LoRA-engine base”，不是独立部署的 canonical base。

若 adapter parity 失败，final 改用临时 merged model + 原生 vLLM；
`EVAL_BASE=True` 时另启 base-only server。fallback 必须使用相同 greedy 协议和 engine
参数并写入 metadata，临时 merged 产物只能位于该次评测的 `candidate/` 目录，不修改
训练 run 的任何既有产物。fallback 的单卡显存可行性与协议后果由实测决定，
见 §8.1 第 2 条。

## 5. 数据与 official harness

固定数据：

```text
dataset: princeton-nlp/SWE-bench_Verified
split: test
instances: 500
HF revision: 91aa3ed51b709be6457e12d00300a6a596d4c6a3
parquet sha256: 43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21
harness revision: f7bbbb2ccdf479001d6467c9e34af59e44a840f9
```

本地 parquet：

```text
data/swegym/SWE-Bench__SWE-bench_Verified/
  91aa3ed51b709be6457e12d00300a6a596d4c6a3/
  data/test-00000-of-00001.parquet
```

official harness 必须：

- 使用本地 parquet，禁止在线漂移到其他 dataset revision；
- 使用 `DOCKER_HOST=unix:///run/docker-swegym/docker.sock`；
- 复用本地 official instance tags，`--cache_level instance --clean false`；
- predictions 每行只写 `instance_id`、`model_name_or_path`、`model_patch`；
- 每个任务都写 prediction；空 patch 显式计数，不能因 harness 静默过滤而改变分母；
- 直接保存 pinned harness 产生的 report，不实现自定义 Verified verifier；
- grading 可复用 harness 的 completed-instance 跳过能力，不建设另一套复杂调度平台。

进入模型 inference 前，先用 gold patch 完成一个离线 official grading smoke；这验证
数据、镜像、harness 和 daemon，而不是模型能力。

当前可见 SWE-Gym、SkyRL 训练任务和 491 条本地 SFT trajectory 与 Verified 未发现
精确任务/文本交集；这只排除当前可见数据直接泄漏，不声称排除 base model 预训练污染。

## 6. baseline 对比输出

### 6.1 主 baseline

`EVAL_BASE=True` 时，主 baseline 是同一次正式 evaluator 得到的 local base。只有
该结果和 final 都完成 500/500 official grading，才输出正式百分点差。
`EVAL_BASE=False` 时不声称存在本地同协议 baseline，只输出 final，并将此前记录的
baseline 作为带协议差异说明的外部参考。

### 6.2 已记录的外部参考

`comparison.json` 保留以下参考，但必须附带协议标签：

| 来源 | 模型/系统 | Verified | 对比性质 |
|---|---|---:|---|
| [SWE-Gym 论文](https://arxiv.org/html/2412.21139v2) | Qwen2.5-Coder-7B-Instruct zero-shot | 1.8% | 外部系统参考 |
| [SWE-Gym 论文](https://arxiv.org/html/2412.21139v2) | SWE-Gym OpenHands-7B-Agent SFT | 10.6% | 外部系统参考 |
| [SkyRL README](https://raw.githubusercontent.com/NovaSky-AI/SkyRL/a0d50c482436af7fac8caffa4533616a78431d66/README.md) | OpenHands-7B-Agent base | 11.0% | 发布内 baseline；外部参考 |
| [SkyRL README](https://raw.githubusercontent.com/NovaSky-AI/SkyRL/a0d50c482436af7fac8caffa4533616a78431d66/README.md) | SkyRL-Agent-7B-v0 | 14.6% | 发布结果；外部参考 |

SWE-Gym 的 7B 结果使用 OpenHands CodeActAgent 2.1、temperature=0、最多 100 turns/
32k context；SkyRL 发布记录没有完整保存所有 decoding 和精确模型 revision。本项目
必须在输出中列出自身 scaffold commit、turn/context 预算和 rollout 数，不能只把
百分比并排后声称严格优于这些模型。

不纳入主比较：SWE-Gym 32B + learned verifier 的 32.0%、排行榜中的 multi-rollout、
RAG、review 或 best-of-N 系统，因为它们不是本项目单 rollout final/local-base 的同类
结果。

## 7. 实施顺序与验收门禁

实施顺序：

1. official gold-patch 单任务 smoke；
2. 新增 eval 入口、evaluator 和测试，不修改训练文件；
3. 单任务 final Agent Loop smoke；`EVAL_BASE=True` 时同次增加 local base smoke；
4. adapter parity 资格检查；
5. 预先指定唯一 run，执行 final 的 Verified 500 inference；`EVAL_BASE=True` 时先执行
   local base；
6. official grading，生成 comparison.json。

进入完整 500 前必须全部满足：

- gold prediction 被 pinned official harness 判为 resolved；
- eval-only image/base-commit 适配成功且没有放宽训练 sandbox；
- Agent Loop characterization 覆盖消息、工具、终止、长度和 patch；
- adapter parity 通过，或按 §4.4 明确进入 fallback；
- vLLM/ninja preflight 通过，sampler backend 已记录；
- `EVAL_BASE` 只接受 `True`/`False`，默认 `False`；
- `EVAL_BASE=True` 时 base/final 使用同一 greedy 协议、任务顺序、并发度和预算；
- 模型输入不含 private grader 字段；
- 空 patch 和 infrastructure failure 不被静默丢弃；
- harness 使用 pinned 本地数据、专用 daemon 和本地镜像；
- 只使用一张确认空闲 GPU，不抢占训练；
- smoke 后没有 eval label 的残留容器或 vLLM 子进程；
- 新增 eval 测试和全部既有训练测试通过；
- `git diff` 除三个新增 eval 文件和本计划外不包含其他实现变化。

## 8. 已具备资产与剩余阻塞

已具备：

- Verified parquet 已固定并校验 SHA256；
- 500/500 official image tags 已存在于专用 daemon；
- image manifest：`data/swegym/verified_pull/images-x86_64.txt`，500 行，
  SHA256 `b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a`；
- daemon Root Dir 为 `/home/2025user/zyp/.docker-swegym`，镜像与训练容器通过 run
  label 隔离；
- vLLM 0.22.1 的 base/runtime-LoRA 路由已做小规模 GPU smoke；
- `ninja==1.13.0` 已在 lock 和项目环境中，依赖检查通过，无需环境变更。

剩余阻塞：

- official gold-patch grading 尚未成功闭环；
- eval-only base-commit checkout 尚未实现；
- 正式 SWE prompt 的 adapter parity 尚未执行；
- evaluator 尚未实现；
- 正式目标 run 尚需在执行前由用户唯一指定；
- 评测需等待一张不被训练占用的 GPU。

### 8.1 必须由实测决定、不得预先写死的三处缺口

以下三点计划在实施到对应步骤时用实测结果回填本文件；回填前不得进入下一步，
也不得以猜测值实现：

1. **turn/context 上限的读取来源与缺失行为**（§3.1）。实施步骤 2 时，对目标 run
   目录做实际盘点：确认 turn/context 上限能从 run 内哪个文件（如 `config.yaml`、
   `run.json`）以哪个字段解析；文件缺失或字段缺失时是 fail closed 还是报错退出，
   以真实 run 目录的盘点结果为准写入本计划。不允许实现时临时猜测路径或静默回退
   默认值。
2. **fallback 的资源与协议后果**（§4.4）。实施步骤 4 的 adapter parity 实测
   同时回答：parity 是否通过；若失败，merged model server 与 base-only server 在
   一张空闲 GPU 上是否能同时放下 7B 权重（以实测显存为准），放不下时串行执行
   还是 abort；fallback 下 base server 不带 `--enable-lora` 与 §3.1"同一 engine
   参数"的偏差是否仍允许输出正式百分点差，还是降级为参考比较。结论按实测写入
   §4.4 与 metadata 模板。
3. **Verified 500 的吞吐与时长预算**（§7 步骤 5）。单任务 smoke 与 adapter
   parity 完成后，用实测单任务 rollout + grading 时长推算 500 任务在固定并发度
   下的 wall-time 和磁盘占用，写入本计划；若推算超出可接受窗口，先与用户确认
   再进入完整 500。

## 9. 明确删除或不实施的冗余逻辑

以下内容不属于“保证 eval 正常”的优秀设计，第一阶段不实施：

- checkpoint-N 通用解析、自动发现、自动选优和多 checkpoint sweep；
- canonical base registry；
- 动态 LoRA 加载 API 和未来 RL 迭代 serving 预留；
- 通用 eval 平台、插件化 verifier、独立 Docker asset manager；
- 自定义 Verified grader；
- 为 eval 复制训练 logprob、reward、advantage、tool mask 记录；
- 为恢复历史训练 scaffold 修改或回滚工作区；
- 镜像下载器、镜像站重试、tmux 下载流程、mirror tag 清理等已完成资产阶段逻辑；
- 自动 Docker prune、镜像所有权数据库；
- 复杂任务调度、自动重跑选优、排行榜抓取或发布流水线；
- 把外部 10.6%/11.0%/14.6% 当成严格同协议 local baseline；
- 为规避 PATH 问题新增/重装 ninja，或静默切换 sampler backend。

以下设计虽然增加检查，但直接保障结果有效，不视为冗余：gold-patch smoke、固定
dataset/harness revision、private grader 隔离、base-commit 适配、Agent Loop
characterization、adapter parity、空 patch/infra 分类、资源清理、
metadata 协议记录以及全部训练回归测试。
