# TRL-based SWE Agent 架构、迁移与实施纠偏计划

> 调研日期：2026-07-18
>
> 旧项目：`/home/2025user/zyp/work/2607_swe_agent`（只读）
>
> 新项目 filesystem path：`/home/2025user/zyp/work/2607_trl_swe_agent`（只是当前物理目录名，不是 Python package identity；当前报告文件为`docs/plan.md`）
>
> 初始调研基线：TRL 1.8.0（2026-07-18时尚未在本机安装或执行；当前实施事实与已冻结版本见第21节）
>
> 实施前复核补充：2026-07-19（重新核验本地模型资产、Docker container 生命周期、工具合同、运行输出布局、模型训练方式、Python 包命名与领域 schema）
>
> 实施中整体纠偏：2026-07-20（依据第21节事实记录，删除正式设计中的多run抽样层级、跨run汇总与非零参数更新硬目标；统一为一次CLI调用创建一个run、一套Trainer/vLLM和一条run内policy状态；2026-07-20第二次审查进一步把进程内CUDA计数降为诊断观察，将正式run的native policy path与Trainer group consumption改为并列验收事实）

> **当前冻结边界：项目已经产生第21节记录的实现与运行产物，但当前只授权修订`docs/plan.md`。在用户审查并明确授权继续前，不得修改任何源码、配置、测试、依赖或运行产物，不得执行import/CUDA/vLLM/Docker资格、加载模型或启动训练。下文修订后的“实施”“资格”“测试”均是后续获批后的控制合同，不是当前操作授权。**[用户决策]

## 0. 结论先行

主方案总体合理，但不能按“给旧 `AgentLoop` 套一层 TRL”实施。建议采用 **TRL 1.8.0 的 `GRPOTrainer + environment_factory + 原生 tool calling` 作为 rollout/训练控制面**，把旧项目中经过验证的 SWE 领域能力拆成无模型依赖的 `SWEEnvironment`、`DockerSandbox`、工具执行、任务加载、patch/verifier 和精简轨迹记录。旧项目的 raw Qwen XML、OpenAI vLLM client、`PolicyTrace`、`GRPOBatch`、自研 objective/trainer、独立 vLLM 生命周期和 adapter 激活链不迁移。

最重要的六个结论如下。

1. **旧项目已有可复用的真实 SWE 执行底座，但当前 HEAD 尚未完成真实在线闭环。** 固定任务 `getmoto__moto-7023`、锁定数据、Docker 镜像、base/gold verifier 资格、隔离容器、工具和二值 verifier 都有代码与运行证据；2026-07-17 的运行到达了两条真实 rollout，随后在 Trainer GPU 资源预检失败；2026-07-16 的历史运行确实调用了一次 optimizer step 并保存 adapter，但 group 为 `[0, 0]`、梯度为零，且后续 adapter 激活失败。当前配置下最新运行又阻塞在 30B vLLM 启动/显存。因此不能把任一次描述为当前 HEAD 的完整、有效 GRPO 闭环。
2. **旧项目当前 reward 不是严格的“仅 verifier 二值结果”。** `SingleInstanceWorkflow._score_rollout()` 对 `invalid_tool_call` 直接产生 `reward_source="model_protocol", reward=0`，不调用 verifier；正常 patch 才经 `SWEGymVerifier.verify()`。新项目不增加 format reward：策略终止直接映射 reward 0；只有实际完成 verifier 才构造 `Verification`，其 `resolved/unresolved` 分别映射 1/0；基础设施故障不构造假的 `Verification`，也不进入 reward group，而是中止当前 batch 并进入 run 级失败处理。
3. **TRL 1.8.0 已覆盖目标控制面，但 `environment_factory` 是实验接口且有一个关键语义缺口。** 它会池化并复用 environment、每次 rollout 调 `reset`、把公开方法暴露为 tools、多轮生成并屏蔽 tool-result tokens、调 reward、计算 group advantage/GRPO、执行优化和 checkpoint；但没有 environment `done`/`terminate` 回调。`submit` 无法像旧 `AgentLoop` 一样立即终止。最小适配是 submit 冻结 patch、返回“已提交，请输出最终无工具回复”，submit 后再调用任何工具记策略失败，下一次 assistant 无 tool call 才由 TRL 结束。该差异必须用集成测试验证，不能 patch/fork TRL。
4. **第一阶段改为 `Qwen2.5-Coder-7B-Instruct + LoRA`，后续目标改为 `Qwen3-Coder-30B-A3B-Instruct + QLoRA`。** 7B 不传 4-bit `quantization_config`，只使用 BF16 base + PEFT LoRA；30B 才启用 BNB 4-bit + PEFT。两份完整、独立、无继承的配置入口分别是`configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml`和`configs/grpo_swegym_qwen3_coder_30b_a3b_qlora.yaml`；字段完整只表示可独立解析并表达方案，不表示运行资格已通过。第一阶段不运行30B，也不提前锁定其TP/DP/FSDP资源拓扑；只冻结模型、QLoRA、PEFT和共享SWE代码边界。30B资源方案必须在7B闭环后基于4×A100实测另行资格。[官方接口事实+用户决策+待实施验证]
5. **真实运行使用单层 `outputs/<run-id>/`，该目录直接作为 Trainer `output_dir`。** 不再配置 run name，也不建立模型/算法系列目录。Trainer checkpoint 原样保留为 `checkpoint-<global_step>/`；一次 rollout 与一个 checkpoint 没有一一对应关系。根级 `run.json` 改为本 run 唯一、动态、综合的结构化记录；`batch.json`/`group.json`表达 generation batch、group 和 optimizer step 的最小关联，不保存 TRL 私有 advantage/logprob tensor。[官方接口事实+用户决策]
6. **仓库路径与 Python 包身份没有冲突。** 当前 filesystem path 可以继续叫 `2607_trl_swe_agent`；唯一 Python 包和内部 import 永远是 `src/swe_agent/` 与 `from swe_agent...`。核心领域对象统一在 `src/swe_agent/models.py`，保留 `Task/Environment/Evaluation/Sample/Action/Observation/Step/Trajectory/Verification`，不创建 `core/`、`EnvironmentSpec/QualifiedTask/RunContext` 等层级或中间对象。[用户决策+建议]

主方案没有已证实的架构致命缺口，但第21节已经证明当前实现和真实run尚未通过核心闭环，后续实施现已暂停等待本修订版审查。获批继续后，Agent按第13节从实际未通过的门禁继续，不重做已有且仍有效的环境/领域基础。7B生成后端固定为TRL原生vLLM colocate+sleep、TP1；若wheel、ABI、显存、sleep/wake或PEFT权重同步任一gate失败，立即停止对应阶段并忠实记录证据，不切Transformers generate、独立server、其他vLLM拓扑或自研生成循环。30B QLoRA只实现配置和共享边界，不在第一阶段运行。**最终运行边界是一条单run控制流：一次CLI调用只创建一个run，只构造一次Trainer/vLLM并维持该run唯一policy状态；execution failure或interruption直接终止并记录本run，不在同一次CLI内创建另一run；正常policy验收未达标则正常结束该run并记录failed验收事实。**[用户决策+审计结论]

### 0.1 强制非目标：永久排除网络安全、认证和通用并发治理

这是后续实施的硬边界，优先级高于本报告中任何可能被误读为“安全平台”或“并发平台”的旧项目事实描述：[用户决策]

- 新项目是**单用户、单机、命令行启动的离线研究项目**，不是网络服务、远程执行平台、多租户系统或多人并发接入系统。项目生命周期内不设计 Web/API 服务、远程 worker、任务队列、网关或常驻控制面；
- 永久不实现网络安全、身份认证、授权/RBAC、用户/租户/session、API key、TLS/证书、签名/证明、secret 管理、配额、限流、安全审计、策略引擎、威胁模型、漏洞扫描或安全合规模块；
- 永久不实现通用并发接入与资源所有权协议，包括 admission control、分布式锁、租约、fencing token、heartbeat、leader election、worker registry、GPU owner/claim、跨进程容器 owner receipt 或并发运行认证。运行合同直接规定：**一个项目工作目录同一时刻只启动一个逻辑训练作业**；该作业内由 TRL/Accelerate/vLLM 启动的必要 worker 进程仍属于同一作业。项目不保证多个独立训练作业共享同一目录、GPU 或 Docker daemon 时仍能正确工作；
- TRL 在**同一个逻辑训练作业、同一个 GRPO group 内**并行推进多个 rollout 是算法所需的有限并行，不是“并发接入”。对此只保证每个 active environment 持有自己的 repository/container handle、没有共享可变 repo，并能在 `finally` 中清理；不得为此建设通用锁、调度器、owner 服务或并发认证层；
- `--network none`、无 host mount、`/testbed` 路径限制、`.git` 拒绝、command denylist 和 private evaluator/oracle 分离只用于固定 SWE-Gym 任务边界、避免外部状态影响结果及防止 gold 泄漏。它们不是网络安全方案，不做安全等级认证，也不继续扩展 Docker hardening；
- `run_id`、`episode_id`、container label、model/image revision、文件 hash/digest 只用于关联运行输出、定位资源和复现实验，不是身份凭证、授权依据或安全 digest chain。cleanup 优先使用 environment 已持有的明确 container ID/对象引用；label 只作诊断和崩溃后人工定位，不做“owner 认证”；
- 如果未来真的需要联网服务、多用户或多个独立训练作业共享资源，应另立项目或由用户明确重定范围；本项目不预留 manager/controller/coordinator、安全接口或并发治理抽象。

因此，后文出现“隔离”“private”“identity”“owner”“digest”“并行”时，只能按上述实验正确性含义解释。实施评审一旦发现新增模块主要服务认证、网络安全、并发接入或资源所有权协议，应直接删除，而不是继续完善。

### 0.2 强制任务与 Docker 下载边界：只运行固定 `getmoto__moto-7023`

第一阶段及未获用户新授权前，Docker/SWE-Gym 测试只允许旧项目已经资格化的单一任务 `getmoto__moto-7023`，以及其现存本地镜像：[用户决策]

```text
image tag: docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7023:latest
image ID:  sha256:8ce447e420f0511fe21b50bc5406b937411b4d829829e82b9b9c1619eeace9de
platform:  linux/amd64
Docker inspect Size: 2,849,787,451 bytes（约 2.85 GB / 2.65 GiB）
```

- Dataset/loader 即使持有包含其他行的既有 Parquet，也只能选择上述 exact instance；存在其他元数据不构成运行其他任务的授权。配置、资格 fixture 和正式闭环都必须 fail-closed 拒绝其他 `instance_id` 或 image；不要为此设计通用 allowlist 服务，一个固定常量/配置断言即可；
- 禁止代码、脚本、测试或依赖安装过程隐式执行 `docker pull/build/load`，也禁止在固定镜像缺失或 identity 不匹配时自动修复。必须直接失败并报告；
- “增加第二个任务”“扩大样本集”或“换一个更容易获得特定 reward 分布的任务”均不属于本计划，**没有用户主动修改范围并逐次明确批准，不得下载、拉取、启动或测试**；即使机器上偶然已有其他任务镜像，也不能据此直接运行；
- 请求用户批准新增任务前，必须先提供空间说明：任务 ID/数量、数据来源与预计下载量、每个 image tag/digest/platform、registry 传输量估计、镜像本地 `Size` 估计、共享 layer 导致的增量不确定性、临时下载/解包峰值、`data/`、`assets/`与`outputs/`预计增长、目标磁盘当前可用空间、最终预计新增占用，以及保留/人工删除方案。估计无法从 registry metadata 确认时必须标注区间和不确定性，不能先下载再补报；
- 未得到批准时只能运行固定任务，并如实记录该run实际产生的reward分布；不能扩大采样预算、自动启动新run或静默换任务。

这里不建设通用 Docker image manager。固定镜像的目标生命周期是：**只读 inspect → exact image ID/platform 检查 → `--pull=never` 复用 → 长期保留**。项目不自动 `rmi/prune`，镜像删除只能由用户在查看占用和影响后手动决定。容器则是另一条生命周期：每条 rollout/verifier 创建 fresh container，结束后按明确 container ID 在 `finally` 删除。

### 0.3 一次实施授权与连续执行边界

当前是实施后纠偏计划审查阶段，继续实现未获授权；首页冻结边界保持有效。用户一旦明确确认修订计划可以继续实施，实施Agent即获得**从当前事实状态推进到第一阶段7B真实闭环所需的一次性连续授权**，无需在已批准范围的门禁之间反复询问。[用户决策]

该授权包含：创建`pyproject.toml/uv.lock/.venv`，执行`uv lock/uv sync --locked`，编写源码、两份配置、`swe_agent` CLI、`grpo.sh`和测试；运行静态/CPU/fixture测试；执行CUDA与7B模型资格；启动7B vLLM colocate/sleep资格；使用固定`getmoto__moto-7023`镜像执行真实Docker工具/verifier集成；最后由一次正式CLI调用创建一个run，在同一Trainer/vLLM/policy状态下运行固定4条rollout的真实GRPO group、完成一次基于该在线group的GRPO optimizer step并保存checkpoint/final adapter。该step是否产生非零parameter update只在运行后观察。

连续授权仍有明确范围：

- 不运行30B模型、30B vLLM或30B训练；第一阶段只实现其完整配置与共享代码边界，运行资格留到7B闭环后另立阶段；
- 不下载或运行第二个SWE-Gym任务，不pull/build/load/rmi/prune Docker image；第0.2节边界保持不变；
- 不自动修改系统driver、安装系统级CUDA、fork/patch第三方库、扩展到多用户/多作业平台或改变binary reward；
- 每个耗GPU/Docker的动作执行前仍须由Agent自己打印即将运行的命令、GPU/显存、固定image/container数量、预计时长、磁盘增长与cleanup范围，作为可审计preflight，**但这不是再次请求用户授权，也不得因此暂停等待回复**；
- 普通`pytest`仍不应意外触发大型运行；真实GPU、vLLM、Docker和系统闭环使用显式marker或正式CLI调用，由第13节顺序主动执行；正式CLI无论execution failure、interruption还是正常policy验收未达标，结束后都不得自动重新执行。

因此，真实7B vLLM不是“实现后私自追加”的无关大型实验，而是固定生成后端资格；一次真实GRPO也不是可省略测试，而是项目系统验收。两者都在用户给出一次开始指令后由Agent按门禁连续推进。重新执行正式CLI、运行30B、增加第二任务或进行系统级环境变更都必须由用户另行明确启动或授权。[用户决策]

## 1. 调研范围、方法与证据标记

### 1.1 实际阅读范围

本次不是只读 README。旧项目已从 CLI 入口沿实际调用链阅读到 optimizer/checkpoint，并交叉检查了配置、测试与真实 artifacts：

- 入口/配置：`pyproject.toml`、`scripts/grpo_once.sh`、`src/swe_agent/cli.py`、`src/swe_agent/config.py`、`configs/grpo_single_instance.yaml`；
- 数据/schema：`src/swe_agent/swegym/loader.py`、旧项目领域模型文件、固定 Parquet revision 和 `artifacts/SWE-Gym/qualification/getmoto__moto-7023/`；
- Agent/工具：`runtime/agent.py`、`model.py`、`chat_template.py`、`tool_protocol.py`、`tool_spec.py`、`tools.py`、`recorder.py`；
- Docker/verifier：`runtime/docker.py`、`verifier.py`；
- GRPO/概率/训练：`training/policy_trace.py`、`batch.py`、`objective.py`、`trainer.py`、`worker.py`、`synchronization.py`；
- vLLM/资源：`runtime/resources.py`、`runtime/vllm_context_plugin.py`、相关 tests；
- 真实证据：`artifacts/test_evidence/20260717_cpu_pytest.xml` 及 2026-07-16 至 2026-07-18 的多个 `grpo_runs`。
- 本机环境：旧`pyproject.toml/uv.lock/.venv`包元数据、uv 0.11.8命令能力/缓存、uv-managed Python、OS/glibc、CUDA toolkit/driver、4张A100的只读占用快照和本地模型目录完整性；未安装、未下载、未启动GPU任务。

TRL 侧优先核对了建议锁定的 [TRL v1.8.0 release](https://github.com/huggingface/trl/releases/tag/v1.8.0)、[`GRPOTrainer` 源码](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_trainer.py)、[`GRPOConfig` 源码](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_config.py)、[`chat_template_utils.py`](https://github.com/huggingface/trl/blob/v1.8.0/trl/chat_template_utils.py)、[`VLLMGeneration`](https://github.com/huggingface/trl/blob/v1.8.0/trl/generation/vllm_generation.py)、官方 tests 目录与 GRPO 文档。依赖关系同时对照 TRL、Transformers、PEFT、vLLM 的官方声明。

### 1.2 证据等级

下文使用以下标记，避免把事实、推断和设计混在一起：

- **[代码事实]**：本地旧项目或指定版本官方源码直接可见；
- **[运行事实]**：本地日志/artifact 或测试报告直接证明；
- **[官方接口事实]**：指定版本官方源码、测试、release 或文档；
- **[推断]**：由前述事实推出，但本机没有执行目标组合；
- **[建议]**：迁移设计选择；
- **[待实施验证]**：受本轮禁止安装/GPU/Docker 限制，不能在本机确认。

### 1.3 本机、旧环境与目标栈约束

| 层面 | 只读观察 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `uv` | `/home/2025user/zyp/.local/bin/uv` 为 `0.11.8`；未发现用户级或 `/etc` 级 uv 配置 | 当前CLI支持project `.venv`、`uv lock`与`uv sync --locked` | 不能证明目标依赖能解析/安装 |
| Python | 系统 Python 3.8.10；uv 已管理 3.12.13、3.13.13 | 可在新项目明确选择 3.12.13 | 不能复用系统 3.8 运行目标栈 |
| 旧项目 `.venv` | uv 0.11.8 创建的隔离 Python 3.12.13 环境 | 旧项目现有环境可复现旧锁 | 不能作为新项目环境；`uv sync` 还可能精确移除非锁定包 |
| 旧 ML 栈 | Torch `2.9.0+cu128`、Transformers 4.57.6、vLLM 0.12.0、PEFT 0.18.0、Accelerate 1.12.0、PyArrow 24.0.0 | 旧 artifacts 的运行背景 | 不满足 TRL 1.8 environment 的 Transformers `>=5.2` |
| 旧环境缺项 | 没有 TRL、bitsandbytes、datasets；`jmespath 1.1.0` 仅作为传递依赖存在，未在 `pyproject.toml` 声明 | 新项目必须自行声明所有运行时直接依赖 | 不能依赖旧环境中偶然存在的传递包 |
| 主机/GPU | Ubuntu 20.04、glibc 2.31、x86_64；4×A100-SXM4-80GB；driver 570.133.07，`nvidia-smi` 报 CUDA 12.8 | 平台/driver 资格输入 | 审计时 GPU 已被其他进程不同程度占用，不能假定四卡随时空闲 |
| CUDA toolkit | `/usr/local/cuda -> cuda-12.0`，`nvcc 12.0.76` 可用但不在默认 PATH | 主机有 CUDA 12.0 编译工具 | 不能据此编译要求 CUDA 12.8/13 的目标扩展；源码构建不是现成后备 |
| 模型缓存 | Qwen3-Coder-30B 本地目录完整；Qwen2.5-Coder-7B 的可读别名 `/home/2025user/zyp/.cache/modelscope/hub/Qwen/Qwen2.5-Coder-7B-Instruct` 有效指向 `Qwen2___5-Coder-7B-Instruct` 完整目录 | 7B 可用于获批后的离线 tokenizer/model 资格，不需要再次下载 | 仅凭文件齐全不能证明目标 Transformers/PEFT/vLLM 栈可加载，更不能证明仅属于30B路径的BNB/MoE QLoRA组合；ModelScope 元数据只写 `Revision:master`，尚未固定不可变远端 revision |
| uv cache | 可见部分 Torch 2.11+cu128、BNB 0.49.2及其他历史 wheel；未见完整 TRL1.8/vLLM目标组合 | 缓存可能减少下载量 | cache 命中不等于 resolver 完整、wheel ABI 可加载或可离线安装 |

2026-07-19 对 7B 资产进行了索引级只读复核：[运行事实]

- 上述可读路径是符号链接，实际目录为 `/home/2025user/zyp/.cache/modelscope/hub/Qwen/Qwen2___5-Coder-7B-Instruct`；此前报告只检查了 `._____temp/Qwen/Qwen2.5-Coder-7B-Instruct` 空临时目录，因而得出“本机缺少完整 7B”的结论是错误的；
- `model.safetensors.index.json` 声明 4 个分片，4 个文件全部存在，无缺失；分片文件合计 `15,231,271,864` bytes，索引中的 tensor payload `total_size` 为 `15,231,233,024` bytes，二者差额来自 safetensors 文件头等容器开销；
- `config.json`、`generation_config.json`、`tokenizer_config.json`、`tokenizer.json`、`vocab.json`、`merges.txt` 和权重索引均存在，目录大小约 15 GB，未发现 `.part`、`.partial`、`.incomplete` 或 lock 残留；
- 本地 `config.json` 标识 `Qwen2ForCausalLM/qwen2`、28 层、28 attention heads、4 KV heads、32768 context、BF16；本地 `tokenizer_config.json` 标识 `Qwen2Tokenizer`、32768 context，且静态包含 `<tools>` 与 `<tool_call>` 模板片段；
- `.mv` 只记录 `Revision:master`，不能作为不可变 revision 证明。实施资格 B 应在终端打印符号链接解析后的路径、索引/配置及分片清单；正式run再把实际路径、revision可观测值和必要文件摘要写入`run.json`。当前无法从本地元数据恢复不可变远端revision，应如实记录`Revision:master`，不为此另建model manifest，也不要求重新下载模型。

旧项目 `pyproject.toml` 固定 Python `>=3.12,<3.13`，并通过显式 PyTorch cu128 index 锁 `torch/torchaudio/torchvision`；`uv.lock` 是旧栈的真实 resolution，不能复制到新项目后修改。新项目当前没有 `pyproject.toml`/`uv.lock`。[代码事实]

TRL 1.8.0 仍是本报告分析的**目标源码基线**，不是已经解析的 package lock。其官方依赖只给出下界（例如 Transformers `>=4.56.2`、datasets `>=4.7`），environment 路径在运行时另要求 Transformers `>=5.2` 和 `jmespath`；TRL 的 `jmespath` 只列在 dev extra，故新项目必须将其作为直接依赖声明。TRL vLLM extra 声明 `vllm>=0.16,<=0.23`。[官方接口事实]

原报告的精确版本组应改为以下**解析搜索空间**，不再称“候选 lock”：

| 包 | 搜索约束/起点 | 当前判断 |
|---|---|---|
| Python | `3.12.*`，实施时固定 uv-managed patch version | 合理且与旧项目隔离 |
| TRL | `1.8.0` | 报告接口基线；必须精确锁 |
| Transformers | `>=5.2`；优先验证 5.13 response-template 路径 | 5.13 是待解析起点，不是已证兼容版本 |
| vLLM | TRL 声明的 `>=0.16,<=0.23` 内选择有官方 A100/Python3.12/glibc2.31、driver570 可用 wheel 的版本 | **首要 blocker/gate**；不得只取范围最高版本 |
| Torch/torchaudio/vision | 必须跟所选 vLLM 官方 wheel/requirements 一致 | 不可独立先锁 Torch 2.11+cu128 再强配 vLLM 0.23 |
| PEFT/Accelerate/datasets/bitsandbytes | 由同一 resolver 产生精确版本；bitsandbytes作为后续30B QLoRA所需的项目直接依赖保留在同一普通依赖集合中 | 0.19.1/1.12.0/0.49.2仅是起点；BNB wheel解析/安装属于统一环境gate，BNB运行功能资格不属于7B gate |
| `jmespath`、PyArrow、Pydantic、PyYAML | 作为项目直接依赖精确锁定 | 分别服务 TRL tool parsing 和旧领域资产迁移 |

特别地，vLLM 0.23 的官方 CUDA requirements 固定 Torch 2.11.0，并依赖 `nvidia-cutlass-dsl[cu13]` 与 `humming-kernels[cu13]`；主机 driver 报 CUDA 12.8。Transformers 5.13 未被其 common requirements 排除，只能说明 Python 依赖元数据没有这一冲突，**不能推出 CUDA wheel 可加载**。[官方接口事实+运行事实] 在找到官方兼容 wheel 前，不应通过自编 vLLM、CUDA forward-compat 包或忽略依赖来“凑出”环境。

### 1.4 uv 环境方案与分层资格

在报告定稿并获实施授权后，建议新项目使用自身根目录下由 uv 管理的 `.venv`，Python 固定为 3.12.13 或实施时可获得的同一 3.12 patch；不得激活/同步旧项目 `.venv`，也不得用 system Python。uv 官方 project 模式默认把环境放在项目 `.venv`，`uv lock` 负责解析、`uv sync` 负责从 lock 安装；`uv sync` 默认精确同步，会删除不在 lock 的包。这是保持新旧项目隔离的最简单方案，但本轮不创建或解析该环境。[官方接口事实+建议+实施冻结]

项目采用一个`pyproject.toml`、一个`uv.lock`、一个项目`.venv`和一个普通依赖集合，不建立train/serve/gpu/dev/7b/30b依赖组。bitsandbytes作为后续30B QLoRA所需的直接依赖安装在同一环境中；因此它的依赖解析、wheel取得和安装成功是**统一环境建立的硬门槛**：这三步任一失败，唯一环境尚未建立，7B也不能越过安装门禁启动。这个安装门槛不表示7B进程使用BNB，也不把BNB import、CUDA扩展、4-bit load或MoE QLoRA能力变成7B模型/运行资格。7B路径直接依赖Torch、Transformers、TRL、PEFT、Accelerate、datasets、jmespath、PyArrow、Pydantic、PyYAML和vLLM；vLLM是计划内唯一生成后端，因此其wheel、import、CUDA extension、colocate、sleep/wake和权重同步都是7B系统闭环硬gate。[用户决策]

资格阶段允许**冻结前的有界resolver迭代**。Agent先在TRL声明的vLLM范围、Python3.12、linux x86_64、glibc2.31、driver570和官方wheel/requirements边界内生成至多3组不重复的Torch/vLLM候选；排序先看CUDA/wheel/平台明确匹配度，再以较新vLLM版本作为同等证据下的tie-breaker，不能盲取范围最高版本。每组候选只运行一次`uv lock`并记录完整resolver反馈；失败后可修改尚未冻结的精确Torch/vLLM版本再试下一组，但不得删除vLLM、扩大声明范围、忽略依赖或源码构建。首个成功lock的`pyproject.toml/uv.lock`立即以内容hash冻结，随后只对该lock执行一次`uv sync --locked`；sync、wheel获取或安装失败时保留冻结文件并停止，不再换版本、后端或环境。[用户决策+建议]

`uv --torch-backend=auto` 只适用于 `uv pip` 接口且属于 preview；`uv lock`/`uv sync` 的 project 路径不能把它当作自动解决方案。project lock 应像旧项目一样通过受控 index/source 与精确版本表达 Torch wheel 来源。获批实施后不得先运行普通 `uv run`：官方行为会隐式 lock/sync，容易把“执行脚本”与“修改依赖状态”混在一起。应先显式审查`pyproject.toml`，再按层执行；下表是未来资格合同，不是当前执行清单：[官方接口事实+建议+实施冻结]

| 资格层 | 获批后的操作与产物 | 通过标准 | 未通过时含义 |
|---|---|---|---|
| A. 能解析并冻结 | 在上述边界内最多评估3组不重复Torch/vLLM精确候选；每组执行一次`uv lock` | 首个成功解包含全部直接依赖（包括vLLM、bitsandbytes），来源可解释且Torch/vLLM wheel/CUDA variant共同选择；记录失败候选与原因，成功后冻结manifest/lock hash | 仅是metadata/索引结论；三组均失败即保留最后provisional manifest和候选评估摘要并停止，不删除依赖或改后端 |
| B. 能安装 | `uv sync --locked`到新`.venv` | lock中的wheel全部取得并安装，无意外sdist/CUDA source build；bitsandbytes wheel也必须安装成功 | 解析成功不等于wheel可取得；BNB wheel失败会阻断唯一环境及7B启动，但wheel已安装只证明安装门禁，不证明BNB可import、CUDA扩展可用或4-bit能力成立 |
| C. 基础 import | Agent直接调用新`.venv/bin/python`导入并记录版本/路径 | Python、Torch、Transformers、TRL、PEFT、Accelerate、datasets、jmespath、PyArrow、Pydantic、PyYAML和vLLM均来自新`.venv` | 证明计划内Python模块装载；**不要求bitsandbytes import**，但vLLM import失败即停止7B路径 |
| D. 7B CUDA/生成后端 | 先验证Torch CUDA，再做vLLM extension/ABI探针 | A100、Torch与vLLM CUDA extension可用，才能进入colocate资格 | 不测试BNB CUDA；vLLM失败即停止，不改driver、依赖、配置、后端或拓扑 |
| E. 接口能用 | 固定 7B tokenizer；TRL response schema、native tool render/parse、最小 environment 多轮 fixture | 可使用最小第三方接口 fixture，但不得把 fixture 的 completion/reward 当项目闭环 | 证明 tool/environment contract，不证明真实模型或 vLLM |
| F. 7B BF16 LoRA | 真实7B BF16 base + PEFT LoRA `GRPOTrainer`构造、精确target/trainable set、最小forward/backward、原生save/reload | base frozen、仅预期LoRA参数可训练、无device-map错误；`quantization_config is None`且`load_in_4bit=false` | 证明第一阶段训练模型路径，不证明rollout backend；不要求Params4bit或任何BNB 4-bit能力 |
| G. 7B vLLM | 验证单卡colocate TP1 + sleep/wake + PEFT merged/full-weight sync | 必须由同一个`GRPOTrainer/environment_factory`驱动真实policy，且配置始终`use_vllm=true` | 任一项失败即停止并记录；不切Transformers、独立server、自研loop或其他拓扑 |
| 完整系统组合（对应第16.3节两项并列事实） | 真实 SWE-Gym/Docker/verifier/group/one step | 第15节系统闭环标准 | 这是唯一“环境和项目闭环可用”的结论；其前还必须通过第16.2节I/J记录与Docker门禁 |

E层允许小型第三方接口fixture，但不得使用人工completion、mock reward或mock Docker宣称项目闭环。A–G任一单层或若干层通过都只能使用对应层名称，不能合并写成“环境可用”或“项目闭环可用”；只有第16.3节`native_policy_path_reached`与`trainer_group_consumed`在同一正式run中都为true，才能给出系统闭环结论。状态还必须逐级区分：bitsandbytes写入依赖声明、wheel安装、BNB import、BNB CUDA extension、4-bit模型加载、30B MoE QLoRA训练、30B与vLLM同步、30B完整配置通过，八者互不等价。前两项属于单一环境A/B硬门；后六项只属于未来30B运行资格，不得反向要求7B执行BNB能力探针。[用户决策+建议]

30B QLoRA未来仍使用同一个环境，但其运行资格链不在第一阶段执行：`bitsandbytes import → BNB CUDA/4-bit能力 → BitsAndBytesConfig与真实Params4bit load → Qwen3 MoE模块枚举与PEFT target精确命中 → 量化base冻结与最小forward/backward → vLLM BNB realization → PEFT同步 → sleep/wake → 基于4×A100选择资源拓扑`。第一阶段只保证30B YAML能够完整表达模型/QLoRA/PEFT/vLLM/SWE共享边界，并以`runtime_qualified=false`阻止误启动；不从7B结果外推30B资源结论。[用户决策+待实施验证]

### 1.5 TRL、Transformers、veRL 与 SkyRL 的输出目录惯例

本轮额外核对了与目录直接相关的源码和实际配置，而不是照抄各项目 README 目录树：[官方源码事实+建议]

| 项目/版本证据 | 实际输出合同 | 对本项目的含义 |
|---|---|---|
| TRL 目标1.8及本地main `7c1fa85`（`1.9.0.dev0`） | `GRPOConfig → _BaseConfig → transformers.TrainingArguments`，因此继承`output_dir`、save/logging和resume语义；`GRPOTrainer._save_checkpoint()`最终调用Trainer基类。main中`log_completions=True`会额外创建`output_dir/completions/completions_<step>.parquet` | 直接把一次run目录传给`output_dir`；已有SWE rollout recorder时显式保持`log_completions=False`，避免第二份completion dump |
| Transformers main `b70d02f`（`5.14.0.dev0`）及目标Trainer基线 | `Trainer._save_checkpoint()`在`output_dir/checkpoint-<global_step>/`保存model、optimizer/scheduler/RNG和`trainer_state.json`；`resume_from_checkpoint=True`在同一`output_dir`查找最后checkpoint；只有超参搜索才在其下再建`run-<trial>` | 不插入`training/checkpoints`；第一闭环显式`save_strategy="steps", save_steps=1`得到`checkpoint-1`；不启用超参搜索，故无额外trial目录 |
| veRL 0.8.0本地包 | Hydra配置用`trainer.project_name/experiment_name`给tracker命名；checkpoint默认独立为`checkpoints/${project}/${experiment}`并使用`global_step_<n>`；`rollout_data_dir`和`validation_data_dir`默认`null`、只在显式配置时dump | 说明大型分布式框架会把tracker、checkpoint和可选dump分开，但这是veRL自有Trainer合同，不适用于Transformers checkpoint命名；“默认不dump rollout/validation”值得吸收 |
| SkyRL main `0270053` | `TrainerConfig`有`project_name/run_name`、独立`ckpt_path`、`export_path`、`log_path`；checkpoint为`global_step_<n>`，debug batch/eval dump分别进export路径；mini-SWE示例另配trajectory目录 | 这些路径服务Ray/多节点、自研checkpoint/export和独立infra日志；本项目不复制。只吸收输入数据与训练输出分离、debug dump应显式开启的原则 |

共同惯例是：**数据输入与训练生成物分离；checkpoint由所用 Trainer 自己的原生保存合同管理；rollout/validation 大文件只有明确消费者时才保存。** 重要差异是 veRL/SkyRL 并不使用 Transformers Trainer，所以它们的 `global_step_<n>`、顶层 ckpt/export/log 根不能覆盖 TRL 的 `checkpoint-<global_step>` 语义。本项目按已拍板边界采用更小的交集：`data/ + assets/` 固定输入、`outputs/<run-id>` 单一 run 根、Trainer 原生 checkpoint、项目自有的 batch/group/rollout 复盘目录。[推断+用户决策]

第一阶段不启用 W&B、MLflow、Trackio 或 TensorBoard，也不使用本地实验系列名分层。run identity 只由 UTC `run_id` 表达，指标写入Trainer原生产物、`run.json`和`train.log`；实施Agent不得自行接入远端tracker。`train.log`是一份主进程文本日志，不是分布式日志汇聚服务。[用户决策+建议]

## 2. 旧项目真实架构

### 2.1 入口、配置与运行命令

`pyproject.toml` 将 `swe_agent` 映射到 `swe_agent.cli:main`；`scripts/grpo_once.sh` 实际调用 `uv run --no-sync swe_agent grpo --config configs/grpo_single_instance.yaml`；CLI 构造 `SingleInstanceWorkflow` 并调用 `run()`（`src/swe_agent/workflow.py:115,147`）。[代码事实]

当前正式 YAML 的关键合同是：

- 单一实例 `getmoto__moto-7023`、`group_size=2`；
- 两份 Parquet 固定 revision `bb94ed…`/`3f22e6…`；
- Docker 镜像及 local image ID/expected registry digest、4 CPU、16 GiB、512 pids、exec 300 秒、verifier 3600 秒；当前本地 `RepoDigests` 为空，实际强校验事实以 image ID 为准；
- agent 20 steps、context 32768、observation 12000 chars；
- sampling temperature 1、top-p 1、top-k -1、每 assistant turn 最多 2048 tokens；
- 当前模型是本地 `Qwen3-Coder-30B-A3B-Instruct` BF16，GPU `[0,3]`，vLLM TP2；
- LoRA `r=16, alpha=32, dropout=0, target=q/k/v/o_proj`；
- 自研 objective `epsilon=0.2, beta=0.01, reference=base_model`。

配置由 `config.load_config()`（`src/swe_agent/config.py:146`）做严格 Pydantic 校验，并只允许有限部署覆盖。`instance_id` 甚至是 `Literal["getmoto__moto-7023"]`（`config.py:104`），说明旧项目是刻意收窄的资格闭环，不是通用 SWE 平台。run directory 是 `artifacts/grpo_runs/grpo-<UTC>-<id>/`；历史 Trainer 将 adapter 保存到 `<run>/learner/adapter`，并把 `optimizer_step.json`、policy publication、resource transition等证据放在同一 run root。[代码事实]

### 2.2 从 task 到 optimizer 的实际数据流

```text
scripts/grpo_once.sh
  → cli.main()
  → SingleInstanceWorkflow.run() / _run_once()
  → load_qualified_instance(config)
      → 锁定两份 Parquet 的同一唯一行
      → Task + Environment（公开）/ Evaluation（私有 oracle）
  → build PolicyVersion + image/tokenizer/model manifest
  → ManagedVLLMService.start(policy_t)
  → 对 group 中每个 member 顺序调用 _run_rollout()
      → 新 DockerSandbox（/testbed、base commit、clean tree）
      → AgentLoop.run()
          → render prompt/messages
          → VLLMOpenAIModelClient.generate()
          → exact-one Qwen XML parser/provider 交叉校验
          → Action → ToolExecutor.execute()
          → Observation → TrajectoryRecorder → 下一轮 history
          → submit / invalid / max_steps 终止
      → 提取 git diff、保存 Trajectory + PolicyTrace + container evidence
  → _score_rollout()
      ├─ invalid_tool_call → synthetic model_protocol reward=0
      └─ 其他 → _verify_rollout() → 新 verifier DockerSandbox
            → git apply check/apply → offline eval → pytest marker
            → resolved=1 / unresolved=0 / infrastructure_failure=None
  → build_batch() → group reward/advantage + GRPOBatch
  → 完整停止 vLLM，执行 GPU/FSDP preflight
  → accelerate launch swe_agent.training.worker
      → build_lora_policy()（BF16 LoRA，不是 QLoRA）
      → 自研 Transformers Trainer / compute_grpo_loss
      → current/ref logprob、clip、KL、backward、optimizer.step
      → adapter-only checkpoint + optimizer_step.json
  → 重启 vLLM → AdapterPublisher.activate/verify policy_t+1
  → 历史设计中的step后real rollout + verifier
  → run_report.json
```

这条历史链的核心 orchestrator 是 `SingleInstanceWorkflow`，不是 `AgentLoop`。`AgentLoop` 只控制一条 rollout；`Workflow` 还控制 group、verifier、batch、GPU 交接、Trainer、policy publication、当时的step后probe和总报告。[代码事实]

### 2.3 Task、数据和已验证样例

`load_qualified_instance()`（`src/swe_agent/swegym/loader.py:46`）要求两个数据 revision 与代码常量一致，从每份 Parquet 精确读取一行并逐字段比对；再核对 qualification JSON、`eval_script.sh`、offline 单点替换、gold/test patch 及 SHA-256。公开 `Sample` 只含 `Task`/`Environment`，私有 `Evaluation` 持有 gold patch、test patch、F2P/P2P 和 offline evaluator，防止 oracle 进入 prompt。[代码事实]

固定任务：

- repo `getmoto/moto`，base commit `447710c6a68e7d5ea7ad6d7df93c663de32ac7f1`；
- 问题是 Lake Formation `deregister_resource` 对未知 ARN 抛 `KeyError` 而非 `EntityNotFoundException`；
- F2P 1 项，P2P 8 项；gold patch 仅增加存在性检查和 `EntityNotFound`；
- 镜像 `docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7023:latest`；配置同时保存 expected image ID 与 expected registry digest。当前本地 `docker image inspect` 的 image ID 精确匹配，但 `RepoDigests=[]`，所以这次只重新验证了 ID/platform/size，不能声称本地 Docker 元数据重新证明了 registry digest。[代码事实+运行事实]

qualification 目录的 `base_eval_offline.log`/`gold_eval_offline.log` 证明 base 为 1 fail + 8 pass、gold 为 9 pass。[运行事实] 但 `selected_instance.json` 的历史字段仍写 `qualification_status="selected_not_runtime_verified"`。这是一个 **旧资产元数据陈旧问题**，不是实际 evaluator 未运行：loader当前也不消费该status，只校验内容与hash。[代码事实+运行事实] 新项目不把该字符串迁入Task/Evaluation或控制流；资产资格由测试是否通过表达，不新增或维护第二个资格status字段。

### 2.4 AgentLoop 与工具协议

`AgentLoop`（`runtime/agent.py:86`）同时承担五类职责：

1. 用 `SYSTEM_PROMPT` 和 task 构造 messages；
2. 调模型生成并保存 token/logprob；
3. 解析 exact-one raw Qwen XML，和 provider `tool_calls` 交叉核验；
4. 构造领域 `Action`、调用 `ToolExecutor`、记录 `Observation/Step`；
5. 判断 submit、invalid、max_steps、异常并完成 `Trajectory`。

`_render_messages()`（`agent.py:278`）会从历史 Step 重新构造 OpenAI messages，甚至重新解析 raw XML。这是旧协议/概率一致性设计的一部分，不应原样进入 TRL。

工具的 **能力合同** 与 **传输协议** 在旧项目中已经部分分离：

- `runtime/tool_spec.py:19,97,113` 定义 `ToolSpec`、原生 JSON schema、工具表和参数/跨字段校验；
- `runtime/tool_protocol.py:26` 的 `AcceptedCall` 和 XML parser 属于 Qwen XML 传输；
- `runtime/tools.py:71` 的 `ToolExecutor` 才是实际领域执行器。

`ToolExecutor` 的领域价值包括：所有路径限定 `/testbed`、拒绝 `.git`、UTF-8、输出截断；`edit_file` 支持 create 或 exact-once replacement 且必须产生 diff；`run_command` 有超时并拒绝 Docker/Podman、apt/pip 和敏感路径；`submit` 要求非空 diff。[代码事实] 这些不能因改用 native tool calling 而丢失。

### 2.5 Docker 与 repository 生命周期

`DockerSandbox`（`runtime/docker.py:97`）每条 rollout 和每次 verifier 都创建独占容器。创建命令使用锁定 image，`--pull=never`、`--network none`、`--cap-drop ALL`、CPU/内存/pids 限制和 ownership labels；没有 host mount。镜像内 `/testbed` 已准备 repo，`start()` 后校验 `HEAD == base_commit` 和 clean tree；命令用 `docker exec` 执行；patch 用 `git diff --binary --no-ext-diff` 导出；finally 只按匹配 labels/owner receipt 清理。[代码事实]

create command 也没有 `--user`，因此容器继承资格镜像的默认用户；这不是经过项目显式收窄的 non-root 合同。没有设置 `no-new-privileges` 是资格镜像兼容的既有选择，不宜在迁移时顺手改变。网络关闭、无 mount、fresh verifier container、base/clean 校验是第一阶段可信度的关键，应基本保留，但只作为固定任务与复现合同，不开展进一步安全 hardening。当前 rollout 是 workflow 顺序创建，尚未证明同一逻辑训练作业内多个 TRL environment 同时调用 Docker CLI 的行为；新项目只测试这一种 group 内有限并行的 repo/container 隔离，不测试或支持多个独立训练作业的并发接入。[代码事实+推断+用户决策]

#### 2.5.1 Docker image 与 container 生命周期完成度

必须区分两类生命周期：[代码事实+运行事实+建议]

| 对象/阶段 | 旧项目当前事实 | 新项目第一阶段决策 | 当前判断 |
|---|---|---|---|
| image 获取 | `SubprocessDockerClient` 明确声明不会 `pull/build/load/prune`；create 使用 `--pull=never` | 原样保留“绝不自动获取”；只用已存在的固定镜像 | 对本项目是正确边界，不是缺功能 |
| image identity | `inspect_image()` 强校验 image ID 与 linux/amd64；仅当 `RepoDigests` 非空时校验 expected registry digest | 记录 tag、expected/observed ID、platform、`Size`、RepoDigests；ID/missing fail-closed | 已有成熟代码可适配；registry digest 当前不可重新观察 |
| image retention/removal | 不执行 `rmi/prune`，没有配额或自动回收 | 固定资格镜像长期保留；删除必须由用户手动决定 | 足够，不建设通用 image lifecycle manager |
| container create/use | rollout/verifier 各自 fresh create/start，校验 base/clean，执行工具/测试 | 保留；只允许 fixed task/image | 已有成熟底座 |
| container cleanup | context manager/finally 已覆盖正常退出、start/base 校验失败和 `KeyboardInterrupt`；旧实现另有 owner receipt/stale recovery | 按 environment 持有的明确 container ID 幂等删除；失败后保留 handle 供顶层 `finally` 重试；不迁移 owner 认证体系 | 旧底座可适配，但不能原样复制 cleanup 状态机 |
| 当前机器状态 | 2026-07-19 只读 inspect 确认固定 image ID、linux/amd64 和 `2,849,787,451` bytes；按旧项目 label 查询无遗留 container | 实施前重复同样只读 preflight | 固定镜像当前可用；没有观察到旧项目遗留容器 |

所以，“Docker image 生命周期是否做好”的准确答案是：**旧项目已做好固定镜像的存在性/身份检查和禁止隐式拉取，且有容器创建/清理实现；它没有通用镜像下载、配额、删除和 prune 管理。新项目目前只有经本报告锁定的最小合同，尚未实现业务代码。** 在本项目只允许一个固定任务的前提下，不应补通用 image manager；实施只需迁移 inspect/fail-closed、固定 image 复用和简单 container cleanup。

旧实现有两个必须在迁移时修正、不能写成“已经直接满足”的细节：[代码事实]

1. `DockerSandbox.__enter__()` 在 create 成功后只设置 `created=True`，启动和后续 `docker exec/rm` 仍使用预生成名称，没有保存 `docker create` stdout 返回的 container ID；新实现必须在 create 成功后、start 和任何后续初始化前立即解析并保存 ID。
2. `_cleanup()` 会缓存第一次 `docker rm -f` 的 `cleanup_result`，并在检查结果前把 `created/started` 清零；第一次 cleanup 失败后，后续调用不会再次执行 Docker 删除。新实现的 `close()` 必须相反：删除成功或明确 `No such container` 才清除 handle；失败时保留 ID 和错误，使 environment reset 或 runner `finally` 能再次尝试。

旧 `__enter__/__exit__` 已用 `BaseException` 保留 base/clean 校验的原异常并附加 cleanup note，`SingleInstanceWorkflow._run_with_final_cleanup()` 也实现了“原异常优先、cleanup 异常另记”的正确雏形。应复用这个异常优先级，而不是复用 owner receipt、stale recovery 或旧 cleanup 缓存语义。[代码事实+建议]

verifier **不复用 rollout 容器**。`SWEGymVerifier.verify()`（`verifier.py:84,96`）在 fresh sandbox 内：

1. 空 patch → infrastructure failure / 不可计分；
2. `git apply --check`/apply：正常 apply 失败为 unresolved 0，超时为 infra；
3. 注入 offline evaluator；
4. 必须看到真实 pytest marker；
5. exit 0 + marker → resolved 1；非零 + marker → unresolved 0；setup/timeout/无 marker/container failure → infra。

这一分类比“进程退出码转 reward”可靠，直接复制后适配。

### 2.6 Schema、Trajectory、recorder、PolicyTrace

旧项目领域模型使用 `extra="forbid"` 的严格 Pydantic schema：[代码事实]

| 对象 | 当前字段/职责 | 主要消费者 |
|---|---|---|
| `Task` (`:19`) | task/source/repo/base/problem/hints/prompt_policy/metadata | prompt主要只读problem；workflow/Docker读task/base；其余多为snapshot |
| `Environment` (`:32`) | environment/task/repo/base/runtime/image/workdir/timeout/metadata | 主要由Sample校验和snapshot读取；Docker仍消费独立DockerConfig |
| `Evaluation` (`:44`) | IDs、gold/test、F2P/P2P、original/offline script、metadata | verifier正式只读offline script；其他字段服务loader资格/隔离测试 |
| `Action` (`:59`) | action ID/timestamp/type/tool/args/metadata | executor、Step、XML history重建、PolicyTrace |
| `Observation` (`:77`) | observation/action ID、timestamp、exit/text/error/metadata | history、Step、PolicyTrace |
| `Step` (`:87`) | index/timestamp、完整model input/response、单元素actions/observations、diff/usage/status/metadata | trajectory、history重建、PolicyTrace |
| `Trajectory` (`:144`) | IDs、model/agent/timestamps、steps、final status/patch/failure/metadata | verifier、report、batch/PolicyTrace |
| `Sample` (`:166`) | task/environment，加可选evaluation/trajectory及跨ID校验 | loader/runtime/canonical rollout |

`TrajectoryRecorder`（`runtime/recorder.py:13`）很轻，只保证连续 Step 并完成 Trajectory；可保留思路。

`PolicyTrace`（`training/policy_trace.py:59,104,135`）则是自研 Trainer 的概率 sidecar：每个 assistant segment 保存完整 rendered messages、prompt/completion token IDs、old logprobs、action mask、sampling config、policy/model/template/tokenizer/tool-contract digest，并重放/检查 Qwen tokenization。它解决了旧“外部 vLLM 采样 + 自研 Trainer 重算”的身份与概率合同。TRL 自己已经持有 `completion_ids`、old/current/ref/sampling logprob、completion/tool mask、advantages；完整迁移 `PolicyTrace` 会制造第二个 Trainer 内部状态源，**不迁移**。

`GRPOBatch`（`training/batch.py:105`）又复制 reward、advantage、trace digest 和 artifact path，是自研 worker 的输入；新项目由 TRL sampler/buffer 负责，不迁移。训练侧只把TRL公开指标、本项目的核心闭环判定、非门禁数值观察和checkpoint引用写入动态`run.json`；generation batch/group的自然统计分别写入`batch.json`/`group.json`，不建立第二套训练张量证据，见第18节。

### 2.7 旧自研 GRPO 的契约

旧实现值得理解，但不复现：

- `compute_group_advantages()`（`training/objective.py:22`）使用 group 均值和 **population std**，退化组全 0；
- `PolicyTrace` 提供 vLLM sampling old logprob；Trainer 对每个 assistant segment 构造独立训练 row，同一 trajectory advantage 复制给各 segment；tool observation 不作为 action token；
- `compute_grpo_loss()`（`objective.py:48`）计算 current/old ratio，按 `[1-epsilon,1+epsilon_high]` clip，`beta=0.01` 时通过 disable adapter 得 base reference，并用 reverse-KL K3 estimator；
- 自研聚合按有效 action-token mass，分布式 padding/scale 由 worker 处理；
- `build_lora_policy()`（`training/trainer.py:253`）加载 **BF16 base + LoRA**，不是 4-bit QLoRA；
- Transformers Trainer 执行一次 optimizer step，保存 adapter-only；
- `ManagedVLLMService` 完全停止后启动 Accelerate/FSDP Trainer，再重启服务；`synchronization.py` 通过 vLLM LoRA HTTP endpoints 激活新 adapter。旧测试明确当前统一配置**不使用 sleep endpoints**。

旧数学与生命周期均有明确目的，不应评价为错误；但新项目锁定 TRL 标准后，它们属于被替代的 contract。

## 3. 旧项目已经跑到哪里，具体未跑通什么

### 3.1 测试覆盖不等于端到端

`artifacts/test_evidence/20260717_cpu_pytest.xml` 记录 111 项、0 error/0 failure、约 25 秒。[运行事实] 覆盖 schema/loader、XML parser、Docker command/cleanup mock、工具、verifier fake sandbox、资源 owner、fake full workflow、objective、tiny LoRA/gradient/FSDP contract 等。它证明模块合同较成熟，但没有启动真实 Docker daemon、30B vLLM、A100 FSDP 或当前 HEAD 完整链路。

### 3.2 三组关键真实运行证据

| run | 已真实到达 | 阻塞/含义 |
|---|---|---|
| `grpo-20260717T105121Z-c51237f1` | vLLM 启动；两条 fresh Docker rollout；模型实际生成 `list_files(moto/lakeformation)`；group/artifacts | 历史 client 把末尾 `<\|im_end\|>` 当语义文本，故两条被判 invalid，reward `[0,0]`、advantage 0；当前 `runtime/model.py` 已新增只移除一个已校验 EOS boundary 的修复。随后 vLLM 已停止，但 GPU 3 仅 46,436 MiB free，低于 70,000 MiB Trainer 门槛，未进入当前 Trainer。 |
| `grpo-20260716T143239Z-531b87be` | 历史配置下真的 `optimizer_steps=1`、adapter-only save；报告称 LoRA hash 变化 | group 退化，`train_loss=0`、`gradient_norm=0`；hash 变化不能证明由非零学习信号造成。之后重启 vLLM 的 adapter activation 返回非 JSON，未完成当时设计的post-step rollout/report。它不是当前 HEAD/当前 GPU 配置闭环。 |
| `grpo-20260718T024136Z-e714f845` / `...T122433Z-26370ae7` | 尝试当前 30B vLLM TP2 | 前者 model load 后 KV cache 余量约 0.02 GiB、无法满足 32K；后者 worker 各约 30,008 MiB，但 1800 秒启动资格超时，清理成功。最新进展甚至未到 rollout。 |

所有检查到的正式 run 中没有最终 `run_report.json`。[运行事实] 当前真实状态应表述为：**任务/Docker/verifier/rollout 的关键资产已分别或部分真实验证；历史一次零梯度 optimizer step 已发生；当前 HEAD 的 system closed loop 未完成，也没有非零parameter update证据。**

### 3.3 可直接承接的已验证资产

- 固定任务/数据 revision/qualification hash 与私有 oracle 隔离；
- 镜像身份、base commit、offline evaluator、base/gold 结果；
- 无网络、无 mount 的 fresh Docker sandbox 与 cleanup 思路；
- 工具文件边界、编辑/运行/submit 语义；
- fresh verifier 和 resolved/unresolved/infra 分类；
- `Task/Environment/Evaluation/Action/Observation/Trajectory` 的语义基础；
- 真实失败 artifacts 所揭示的 EOS boundary、资源门槛、启动/激活问题，作为迁移回归样例。

## 4. TRL 1.8.0 的真实接口与 GRPO 语义

### 4.1 environment_factory 和多轮 tool loop

[`GRPOTrainer.__init__`](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_trainer.py) 的直接合同是：[官方接口事实]

- `tools` 需要 Transformers `>=5.0`，`environment_factory` 需要 `>=5.2`，两者需要 `jmespath`，tokenizer/processor 必须通过 `supports_tool_calling()`；
- `environment_factory` 可为一个 factory 或按数据行 `environment` 字段选择的 dict，明确标为 **experimental**；
- environment 的正式类型合同是 `reset(**dataset_columns) -> str | None`（实现内部还兼容multimodal content list，本项目不依赖该扩展）；单 factory 模式会把当前数据行（包括 `prompt`）传给 `reset`，dict 模式只剔除选择器字段 `environment`。因此第一版 `reset` 应显式接收需要的task ID并容忍其余只读列，不能假设只收到task fields。返回字符串会追加到最后一条user message；除`reset`/`get_reward`和私有方法外的public bound methods被暴露为tools；
- Trainer 初始化时 probe 一个实例验证方法/schema，按 batch 需要扩充 pool，后续 **construct once, reset often**；每个同时 rollout 使用独立 instance，但跨 batch 会复用；
- environment 可有无参 `get_reward()`，Trainer 每条完成调用一次，也可与 `reward_funcs` 并存；环境实例列表也会传给 custom reward function；
- `_tool_call_loop` 解析结构化 tool calls；一次 assistant message 可以含多个 call。同步 SWE tools 按列表顺序调用，调用表达式只有 `tool(**function["arguments"])`，**不会注入 tool-call ID、message index、completion index或 environment context**；这些不能成为 wrapper 的必需参数；
- callable 返回 list 时作为 content blocks 透传，否则直接 `str(result)`；TRL 生成的 tool message只有 `role/name/content`，不含 `tool_call_id`。因此 wrapper 应返回确定性的字符串（建议由唯一 `Observation` 单向序列化），不能直接返回 Pydantic 对象后依赖其 `str()` 形式；
- 未知工具或 callable 异常会被 `_tool_call_loop` 捕获并转为 error tool result，继续下一轮。也就是说，tool 内抛出的 Docker 基础设施异常不会立即穿透 Trainer；environment 必须先保存 infra 状态，finalizer/reward 再 raise，中止 batch；
- tool-result tokens 在 `tool_mask` 中为 0，模型生成 tokens 为 1；completion mask 处理 padding/truncation，loss 使用 `completion_mask * tool_mask`；
- 终止条件是一次 assistant response 没有 tool calls、达到 `max_tool_calling_iterations` 或总 completion 长度上限；超预算的 tool-result/suffix 会回滚。

这意味着提示中“每条 rollout 一个隔离 environment”的方向成立，但需要修正为：**每个当前 batch 的 rollout 独占一个 pooled instance；隔离 repository 必须由每次 `reset` 重建，而不是依赖 environment 构造一次后永不复用。** pool slot 和列表下标都不是持久 episode identity。

官方 `tests/test_grpo_trainer.py` 在 v1.8.0 分支覆盖了 environment-only reward、environment reward 与普通 `reward_funcs` 并存、多 environment中有/无 `get_reward` 的混合，以及完全没有 reward source 时拒绝构造等路径；tool calling/loss/vLLM还有相邻测试。[官方测试事实] 本轮因本机未安装目标栈，**没有执行这些官方测试**；并且官方 tests 中没有发现一项把 QLoRA、multi-turn environment、colocate、sleep 和 A100 TP 全部放在同一用例中。因此本报告对组合能力只给“源码可组合、需本机资格”，不把子功能测试外推为完整闭环。

### 4.2 submit 与 termination 的表达缺口

TRL 1.8.0 environment 没有 `done`、`terminate` 或 tool result 携带 terminal flag 的正式接口。[官方接口事实] 工具方法返回值只会成为 observation；只要模型继续发 tool call，loop 就继续。因此旧 `submit → 立即 break` 无法一比一表达。

推荐最小语义：

1. `submit()` 读取并冻结 non-empty patch，记录 `submitted=True`，返回简短 terminal observation；
2. 提示要求模型下一 turn 只输出最终文本、不要再调用工具；
3. submitted 后任何 public tool 调用返回策略错误并把 episode 标为 unresolved；不得修改 repo；
4. 下一 assistant 无 tool call，TRL 原生停止；reward finalizer 检查 exactly-one successful submit；
5. `max_tool_calling_iterations=20`、`max_prompt_length=8192`、`max_completion_length=22528`和`context_safety_margin=2048`显式固定，满足`8192+22528+2048=32768`。prompt上限必须用目标tokenizer对最终chat-template渲染结果实际计数，包含system/user边界、tools schema和所有special tokens；completion上限由assistant输出、动态tool-result及其消息边界共同消费。资格与正式入口使用同一计数函数，prompt超过8192即停止并缩短prompt文案，不动态放大context/completion或占用2048安全余量。

这会多生成一个 assistant final turn，和旧协议不同，但不改变 SWE 核心目标。[建议] 若模型频繁 submit 后继续 tool call，它是可学习的策略失败，不应靠 grammar 或 TRL patch 隐藏。

### 4.3 Qwen2.5 tool calling

Qwen2.5-Coder-7B 官方 [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/tokenizer_config.json) 包含 `<tools>`/`<tool_call>` 原生模板。[官方接口事实] 本地完整目录的 `tokenizer_config.json` 也静态包含这两个模板片段。[运行事实] TRL 1.8 的 `add_response_schema()` 识别已知 Qwen2.5 template 并映射到 Qwen3 风格 response template/schema；对非 prefix-preserving template，Trainer 会使用 training-safe template。[官方接口事实]

但静态 JSON 存在不等于目标版本 Transformers 能成功实例化 tokenizer，也不等于 TRL 能识别这一份实际模板。本轮没有导入目标栈或加载 tokenizer。获准实施后应直接使用上述本地目录，在不启动训练前验证；探针默认只向终端打印解析路径、revision可观测值和检查结果，正式运行再由`run.json`记录输入身份：

- `supports_tool_calling(tokenizer)` 为真；
- `add_response_schema()` 成功；
- 一个已知 native tool call 可 render → parse → arguments 保真；
- prompt prefix 在追加 assistant/tool messages 后保持前缀；
- `<|im_end|>` 只作为消息边界，不进入 tool arguments/content；
- unknown/invalid args 能进入预期失败路径。

失败合同已确定：先允许一次只使用Transformers/TRL官方格式的最小training-safe chat-template适配，不恢复XML、不增加provider协议；适配后仍不能稳定round-trip即停止7B native-tool路径并保存证据，不自动更换模型或协议。[用户决策]

### 4.4 GRPOConfig：显式值、版本默认和数学语义

以下以 TRL 1.8.0 源码为准。为防未来默认漂移，建议把核心语义参数即使等于默认也 **显式写入配置**。

| 参数 | 1.8 默认 | 第一阶段建议 | 来源 | 运行/数学语义 |
|---|---:|---:|---|---|
| `num_generations` | 8 | 4；仅最小接口fixture可用2 | 显式 | 正式同一prompt group固定4条rollout，与输出目录`0000..0003`一致；group size不因reward结果改变 |
| `num_iterations` | 1 | 1 | 显式 | 同一 rollout batch 更新一次，不复用多轮 PPO epoch |
| `loss_type` | `dapo` | `dapo` | 显式固定默认 | active model tokens 的 loss 总和除以全局 accumulated batch 的 active-token 数；非旧自研 token-mass/segment 聚合 |
| `scale_rewards` | `group` | `group` | 显式固定默认 | 每个 group 先减均值，再除 group nan-aware std（源码加小 epsilon）；退化 group advantage 0 |
| `multi_objective_aggregation` | `sum_then_normalize` | `sum_then_normalize` | 显式固定默认 | 本阶段只有一个 reward，结果无差异 |
| `epsilon` | 0.2 | 0.2 | 显式 | PPO/GRPO ratio 下界 `1-ε` |
| `epsilon_high` | `None→epsilon` | `None` | 显式 | 上界 `1+0.2` |
| `delta` | None | None | 显式 | 不加额外 two-sided upper cap |
| `beta` | 0.0 | 0.0 | 显式 | 不创建 reference model、不加 KL；不同于旧项目 0.01 base ref |
| `importance_sampling_level` | `token` | `token` | 显式固定默认 | current/old PPO ratio 逐 token，而非 sequence ratio |
| `max_completion_length` | 256 | 22528 | 显式 | **多轮总response budget**，含所有assistant turns、动态tool-result和消息边界；与实际tokenized prompt≤8192之间保留2048 tokens安全余量 |
| `max_tool_calling_iterations` | None | 20 | 显式 | 限制tool-calling iterations；一次iteration可有多个call |
| `mask_truncated_completions` | False | False（第一阶段固定） | 显式固定默认 | 截断样本仍参与标准loss；本计划不根据首轮结果自动改值 |
| `use_vllm` | False | True | 显式 | 第一阶段唯一生成后端；资格失败不改为False |
| `vllm_mode` | `colocate` | `colocate` | 显式固定默认 | 同 Trainer 进程/训练 GPU |
| `vllm_enable_sleep_mode` | False | True | 显式 | optimizer 时 offload weights/cache；唤醒有传输开销 |
| `vllm_tensor_parallel_size` | 1 | 1 | 显式 | 仅为第一阶段7B vLLM资格值；30B第一阶段不运行，不由此锁定拓扑 |
| `vllm_gpu_memory_utilization` | 0.3 | 0.3 | 显式固定默认 | 只资格计划值；失败即停止，不临场改成0.4/0.5、独立server或多卡vLLM |
| `vllm_importance_sampling_correction` | True | True | 显式固定默认 | 修正 vLLM sampling logp 与 Trainer 重算 logp 的框架差异 |
| `vllm_importance_sampling_mode` | `sequence_mask` | `sequence_mask` | 显式固定字段默认 | 整条序列 ratio 超界则 mask；文档枚举中“token_truncate default”字样与字段默认矛盾，以 dataclass 为准 |
| `vllm_importance_sampling_clip_max/min` | 3.0/None | 同值 | 显式 | ratio >3 的序列 mask，无下界 |
| `vllm_structured_outputs_regex` | None | None | 显式固定默认 | 不启用grammar/regex constrained decoding；非法调用保留为策略行为 |
| `temperature/top_p/top_k` | 1/1/0 | 1/1/0 | 显式 | vLLM/Transformers 的 top-k 禁用值在 TRL 是 0，不照搬旧 vLLM `-1` |
| `repetition_penalty` | 1.0 | 1.0 | 显式固定默认 | 不对策略分布增加未说明惩罚 |
| `learning_rate` | 1e-6 | 1e-6 | 显式 | 与旧 config 一致 |
| `gradient_checkpointing` | True | True | 显式固定默认 | PEFT 路径由 TRL 处理 input gradients；7B LoRA 和未来 30B QLoRA 均需实测显存/速度 |
| `bf16` | 环境判定/通常 True | True | 显式 | A100 compute/adapter dtype |

`mask_truncated_completions` 第一阶段固定为 TRL 1.8 默认 `False`，避免在尚无真实长度分布时主动改变标准基线。截断仍由finalizer记录为策略失败0；实施Agent不得根据首轮结果自行改为True。任何超出第一阶段的截断策略实验都需要先修改计划。[用户决策+建议]

第一阶段验收run的训练调度值固定为`per_device_train_batch_size=1`、`gradient_accumulation_steps=1`、`num_iterations=1`、`max_steps=1`、`logging_steps=1`、`save_strategy="steps"`、`save_steps=1`、`save_total_limit=2`和`log_completions=False`。`max_steps=1`只限定本阶段验收run执行一次基于真实在线group的GRPO optimizer step，不引入run外抽样层级；一次CLI仍只创建该一个run和一套Trainer/vLLM。optimizer step不等于非零parameter update：若advantage/gradient为零，step后policy可在数值上不变。任何由TRL派生的`generation_batch_size/steps_per_generation`必须通过锁定版本配置校验；校验失败即修正配置关系，不通过增加第二套训练循环解决。[用户决策+建议]

### 4.5 reward、advantage、mask、loss 和 optimizer

TRL `_calculate_rewards` 将各 reward source 对不可用项写 NaN；同组做 nan-aware mean/std，再 `nan_to_num`，全不可计分或退化项 advantage 为 0。[官方接口事实] 第一阶段不应借 `None` 吞基础设施错误：Docker/verifier infra 应 raise，使训练 batch 失败并保存 failure evidence；否则 Trainer 可能在缺失样本上正常 optimizer，造成虚假系统成功。[建议]

完整多轮 completion 由 Trainer tokenize/持有；tool-result token mask=0，assistant 生成 token mask=1。`dapo` 对有效模型 token 在全局 accumulated generation batch 上归一化。`num_iterations=1` 时，若无需 vLLM correction，old logp 可直接取 current detached；启用 vLLM correction 时仍重算训练 logp，并用 sampling logp 计算额外 ratio。它与 PPO 的 current/old token ratio是两层不同 correction。[官方接口事实]

`beta=0` 时没有 reference/KL。若以后显式设非零，PEFT 通常通过临时 disable adapter 得 base reference；`sync_ref_model=True` 与 PEFT 不支持。[官方接口事实] 第一阶段遵循标准默认，不为复现旧 `beta=0.01` 增加内存和语义复杂度。

optimizer、gradient accumulation、scheduler 和 checkpoint 由 Transformers Trainer 基类控制。PEFT checkpoint 保存 adapter config/weights；完整 checkpoint 还需保存 optimizer/scheduler/`trainer_state.json` 才能 resume。[官方接口事实] 新项目运行证据必须区分“LoRA adapter 可加载”和“Trainer 状态可恢复”。

与旧自研实现的关键契约差异集中如下；这些是版本契约差异，不评价孰优孰劣：

| 维度 | 旧项目当前实现 | TRL 1.8目标基线 |
|---|---|---|
| reward来源 | verifier 0/1；`invalid_tool_call`另有 `model_protocol=0`；其他rollout错误多为fail-closed | 一个episode binary finalizer；所有策略失败0，infra raise |
| group std | `objective.compute_group_advantages` 的 population std | `trainer.utils.nanstd` 明确做 Bessel correction，即 sample std，并加 `1e-4`；group2 `[0,1]` 的advantage幅度因此不同 |
| training unit | 每个assistant segment一行，trajectory advantage复制到各segment | 整个多轮completion一条序列；assistant生成tokens参与，tool-result tokens由tool mask排除 |
| old policy | 外部vLLM逐token raw old logprob写入PolicyTrace | `num_iterations=1`时通常current detached；启用vLLM correction时另持sampling logp并重算Trainer logp |
| vLLM mismatch | 无单独框架IS correction，依赖旧trace一致性 | 默认sequence-mask correction，ratio上限3；再叠加token级current/old ratio |
| clipping | 自研token ratio `[0.8,1.2]` | DAPO surrogate同样epsilon0.2，但聚合/IS/mask不同 |
| reference/KL | `beta=0.01`，disable adapter得base ref，K3 reverse-KL | `beta=0`无reference/KL；若非零才使用相同形式的per-token estimator |
| loss aggregation | 自研effective action-token mass和分布式padding scaling | DAPO按全局accumulated batch active model tokens归一化 |
| quantization | BF16 base + LoRA | 第一阶段仍为 BF16 base + LoRA；未来 30B 单独资格 BNB 4-bit base + LoRA |
| lifecycle | vLLM完整stop→FSDP Trainer→restart→HTTP adapter activate | colocate engine由TRL sleep/wake并推送合并后权重 |

因此迁移后的数值不能和旧 `group_advantages.json`/`optimizer_step.json` 做逐值等价断言；应针对新锁定版本保存配置和公共metrics，并用行为级验收判断。

### 4.6 LoRA、未来 30B QLoRA、colocate、sleep 与权重同步

TRL 1.8.0 `GRPOTrainer` 分开接受 `peft_config` 与可选 `quantization_config: BitsAndBytesConfig`。因此两份配置的模型构造合同必须不同：[官方接口事实+用户决策]

- 7B LoRA：`peft_config` 非空、`quantization_config=None`、BF16 base；资格必须证明 base frozen、LoRA target 精确且只有预期参数可训练。
- 30B-A3B QLoRA：`peft_config` 与 4-bit BNB `quantization_config` 同时存在；TRL 检测到量化 base 后会准备 k-bit training，并为 gradient checkpointing 处理 input gradients。该路径的“接口存在”不能代替 Qwen3 MoE 实机可用性。

`VLLMGeneration` colocate 在 PEFT 模型更新时会 wake weights，临时 `merge_adapter()`，把合并后的 named weights 推给 vLLM，再 `unmerge_adapter()`并清 prefix cache；这不是“只发送低秩矩阵”。对 4-bit module，源码还会让 vLLM 使用 BNB quantization realization。sleep 模式管理 weights/KV 的 wake/sleep，但会把压力转移到 host RAM、pinned memory 和同步延迟。[官方接口事实]

因此旧项目的“保存 adapter → HTTP 激活”链不迁移。7B LoRA 与未来30B QLoRA共享TRL PEFT构造边界，但不能共用资格结论：前者只要求统一环境中的BNB wheel已安装，不验证BNB import/CUDA/4-bit运行能力；后者必须额外验证MoE target、BNB 4-bit load、生成后端和实际资源拓扑。第一阶段不执行这些30B运行资格。[建议+待实施验证]

风险边界：

- 本地 TRL main 快照 `7c1fa85` 的 sync 源码仍保留“PEFT + FSDP 是否工作”的TODO；本项目已决定不走FSDP，该事实只说明不能把上游其他并行路径当作TP1失败后的自动后备；
- 没有找到一项官方 test 同时覆盖目标模型、multi-turn environment、colocate、sleep、PEFT sync 与 A100 拓扑；
- 7B 首阶段不同时引入 FSDP；30B 不能从旧项目的 BF16 base 运行证据外推 4-bit BNB/PEFT 结果；
- 两条模型路径都要记录 HBM/RAM峰值、wake 延迟和更新前后生成权重身份；30B 失败时先定位 BNB、MoE target、sync 或资源层，不先写模型专属业务补丁。

## 5. 确定的 TRL 目标架构

### 5.1 修正后的数据流

```text
loader资格化固定SWE-Gym资产 → 直接返回(sample, evaluation)
  → 唯一prompt builder由Task生成prompt
  → TRL Dataset行只含task_id与生成所需prompt（不含完整Sample/Evaluation）
  → environment_factory闭包按task_id取得公开Sample与私有Evaluation
  → TRL GRPOTrainer
      → environment_factory() probe / pool
      → 对每条 completion：SWEEnvironment.reset(**row)
          → 清理上次 episode（如有）
          → fresh DockerSandbox.start()，验证 base commit/clean tree
          → 返回本 episode 的环境说明
      → Transformers/Qwen native tool schema + chat/response template
      → vLLM colocate 生成 assistant tool_calls
      → TRL _tool_call_loop 调 SWEEnvironment public methods
          → Tool adapter → 既有参数校验 → ToolExecutor → DockerSandbox
          → 每个已执行调用追加一个Step(Action, Observation)
          → tool result 作为 observation 回到 TRL conversation
      → submit 冻结 patch；最终无 tool assistant turn 使 TRL 终止
      → 单一 reward_func(prompts, completions, environments, **row)
          → environment.finalize(completion)
          → 策略终止检查；冻结只可从 live rollout container 取得的证据
          → 关闭 rollout container；成功后才启动 fresh SWEGymVerifier container
          → 策略终止→0 / Verification resolved→1、unresolved→0 / infra raise
          → recorder按generation batch/group/rollout写messages、trajectory、final patch和verification证据
      → TRL group normalization + DAPO loss + 7B LoRA backward/optimizer
      → TRL colocate 权重同步 / sleep-wake
      → Trainer原生 `checkpoint-<global_step>/`
      → Trainer/PEFT原生 `save_model(output_dir)`写最终adapter到run根
      → 单写者原子更新动态`run.json`
```

这里建议 **不在 `SWEEnvironment` 暴露 `get_reward()`**，而使用 TRL 正式支持的单一 custom `reward_func`，原因不是绕开 environment reward，而是需要 completion 与 environment state 同时作为最终事实：未知工具由 TRL 捕获，最终无工具 assistant turn也只在 completion 中；无参 `get_reward()` 看不到这些内容。`reward_func` 在一次调用内按 TRL 提供的同位置 `completions`/`environments` 做 `zip(..., strict=True)`，逐一调用私有 `env.finalize(completion)`，只返回 `[0.0/1.0]`。[官方接口事实+建议] episode 的持久关联仍使用 `env.episode_id`，不能把 pool slot、当前列表下标或 GRPO group index当全局身份。

### 5.2 职责边界

| 层 | 唯一职责 | 明确不做 |
|---|---|---|
| TRL/Transformers | messages/template、生成、多轮 tool dispatch、token/logprob/mask、reward 调度、group advantage、loss、optimizer/checkpoint、vLLM 同步 | repo/Docker、patch、SWE verifier、领域 trajectory |
| `SWEEnvironment` | episode 状态、至多一个 rollout sandbox handle、tool method adapter、submit/termination state、正常 finalize、幂等 close | 自己生成模型回复、自己维护训练 token/logprob/advantage、持有 verifier sandbox |
| `ToolExecutor` | 参数通过后的真实文件/命令/编辑/submit 语义 | tool-call parsing、conversation 调度 |
| `SWEGymVerifier` | 在 fresh sandbox 应用 patch、运行真实 tests、返回 `Verification` 或抛出 infra | group reward normalization、训练控制 |
| domain recorder | 连续`Step(Action, Observation)`与Agent termination | 保存完整messages、patch/verification/cleanup、重建Trainer tensors、重复tokenize或计算GRPO |
| run recorder/runner | 创建单一`output_dir`；分配batch/group/rollout目录；作为`run.json`唯一写者接收环境事件并原子更新；保存 factory 创建的 environment 普通引用列表并在顶层 `finally` 逐一 close | 自研 Trainer、通用registry、全局artifact manager或第二套 lifecycle manager |

## 6. 领域模型与schema：保留语言，删除重复所有权

### 6.1 命名和模块合同

新项目的唯一 Python 包名是 `swe_agent`，与当前 filesystem path 无关；二者并不构成冲突，也不是“命名债务”。核心领域对象直接放在 `src/swe_agent/models.py`；只有一个模型文件，不保留无消费者的 `core/` 层级。`Pydantic`只是严格校验和序列化实现，不足以把模块命名为 `schemas.py`。第一阶段不存在平行的 `models.py/schemas.py/dto.py/records.py` 定义，也不在领域类名中加入 `TRL`。[用户决策]

稳定领域名原样保留：`Task`、`Environment`、`Evaluation`、`Sample`、`Action`、`Observation`、`Step`、`Trajectory`、`Verification`。有状态的 TRL adapter 仍叫 `SWEEnvironment` 并放在 `src/swe_agent/environment.py`；模块边界已经足以区分：

```python
from swe_agent.models import Environment
from swe_agent.environment import SWEEnvironment
```

不创建`EnvironmentSpec`、`TaskSpec`、`QualifiedTask`、`TaskBundle`、`TaskContext`、`EvaluationSpec`、`TrajectoryRecord`、`EpisodeContext`、`RunContext`或task registry schema。当前固定实例由loader直接返回`sample, evaluation`；environment factory通过普通闭包或一个局部字典按`task_id`取得它们，无registry类。[建议]

TRL Dataset row不是第十个领域schema，只是训练接口所需的`task_id + rendered prompt`映射；prompt由`Task.problem_statement`和固定prompt builder单向生成，不能反向成为Task的第二事实源，也不能附带完整Task/Environment/Evaluation序列化。[建议]

### 6.2 旧字段与消费者审计

旧项目领域模型文件中的对象名称有价值，但字段不是都应迁移：[代码事实]

| 旧字段组 | 真实消费者/用途 | 新项目结论 |
|---|---|---|
| `Task.repo_name/base_commit/problem_statement` | prompt、Docker base检查、workflow和分组 | 保留，归`Task`唯一所有 |
| `Task.source/hints_text/prompt_policy/metadata.version` | 固定样例中为常量、`None`或只随snapshot落盘；没有核心控制流消费者 | 第一阶段删除；dataset revision/version进固定资产或`run.json` |
| `Environment.task_id/repo_name/base_commit` | `task_id`是Sample关联所需外键；repo/base只被`Sample.validate_links`重复核对，Docker实际从`Task.base_commit`取值 | 保留`task_id`引用；删除repo/base副本 |
| `Environment.image_name/workdir/timeout`及旧`DockerConfig` limits/image identity | Docker create、exec、image fail-closed | 迁入`Environment`并成为唯一运行环境事实源 |
| `Evaluation.gold_patch/test_patch/F2P/P2P/original eval_script` | loader资格/hash/oracle隔离测试；正式`SWEGymVerifier`均不读取 | 只在固定资产资格过程读取，不进入正式运行时`Evaluation` |
| `Evaluation.offline_eval_script` | `SWEGymVerifier._verify_in_sandbox()`唯一正式消费者 | 保留为`Evaluation`私有运行时字段 |
| `Action.action_id/action_type/timestamp/metadata.provider_tool_call_id/tool_protocol` | 旧XML history重建、`PolicyTrace`对账和旧recorder校验 | 这些消费者不迁移；字段全部删除 |
| `Observation.observation_id/action_id/timestamp/metadata` | 旧Step ID/wire对齐；`timed_out/duration/changed_paths`藏在metadata | 删除ID/linkage/任意metadata；正式控制事实显式建模 |
| `Step.model_input/model_response/usage/status/metadata/workspace_diff` | history重建、概率trace、旧失败报告 | conversation、TRL或独立patch已经持有；核心Step全部删除这些字段 |
| `Trajectory.model_name/agent_config/timestamps/final_status/final_patch/failure_type/metadata` | 旧workflow、PolicyTrace、report和patch输出 | 由conversation、patch、run status和输出目录分别承担；不迁移到核心Trajectory |
| `Sample.evaluation/trajectory` | 可选序列化与旧跨ID校验；loader正式返回的public sample已经令两者为空 | 删除；`Sample`只组合公开`Task + Environment` |

关键证据链是：`runtime/agent.py::_render_messages()`依赖`Action.metadata.provider_tool_call_id/tool_protocol`重建旧history，`training/policy_trace.py::build_policy_trace()`再依赖Action/Observation IDs、Step messages和Trajectory failure/metadata做概率对账；这两个消费者都由TRL替代。相反，`verifier.py::SWEGymVerifier._verify_in_sandbox()`只读取`Evaluation.offline_eval_script`，而`swegym/loader.py::load_qualified_instance()`读取gold/test/F2P/P2P/original script只是为了资格比较并构造旧对象。因此删除字段有具体消费者边界，不是因为字段“看起来复杂”。[代码事实]

### 6.3 第一阶段确定的核心字段

以下是实施合同，不再保留“可选保留旧list/ID”的第二方案：[建议]

```text
Task
├── task_id
├── repo_name
├── base_commit
└── problem_statement

Environment
├── environment_id
├── task_id                       # 只引用Task主键，供Sample校验配对
├── image_name
├── expected_image_id
├── expected_registry_digest
├── workdir
├── cpus
├── memory
├── pids_limit
├── exec_timeout_sec
└── verifier_timeout_sec

Evaluation                         # 私有，不可进入Dataset/prompt/reset/tool result
└── offline_eval_script

Sample
├── task: Task
└── environment: Environment

Action
├── tool_name
└── arguments                     # 执行前已通过领域校验的唯一canonical dict

Observation
├── text
├── exit_code
├── error_type
├── timed_out
└── truncated

Step
├── index
├── action: Action
└── observation: Observation

Trajectory
├── task_id
├── environment_id
├── steps: list[Step]
└── termination

Verification
├── result                        # resolved | unresolved
├── patch_apply_status
├── pytest_started
├── exit_code
├── stdout
└── stderr
```

`Environment`是image、资源限制和超时的唯一所有者；`Task`是repo/base和task identity的唯一所有者。`Environment.task_id`只是外键引用，`Sample`用它拒绝Task/Environment错配，不复制repo/base。`expected_registry_digest`在旧`runtime/docker.py::inspect_qualified_image()`中仍有fail-closed消费者，因此是必须保留的例外；它是镜像复现事实，不是安全认证。[代码事实+建议]

`Evaluation`不保存`evaluation_id/task_id/environment_id`。loader返回`sample, evaluation`后建立只存在于当前进程内的不可变`task_id -> (Sample, Evaluation)`映射；Dataset row只带`task_id + prompt`，`SWEEnvironment.reset(task_id, **_)`按key取得私有`Evaluation`，不依赖row位置或pool slot，也不把Evaluation放入公开列/reset参数。gold patch、test patch、F2P/P2P和original eval script继续存在于`assets/`和loader资格测试中，用来证明固定资产一致；loader完成资格后只构造含`offline_eval_script`的运行时`Evaluation`。这避免重复ID和gold进入长期存活的episode对象。[用户决策+建议]

`Verification`只在真实 verifier 形成可归因的 resolved/unresolved 结论后构造；patch apply 正常拒绝和真实 pytest 失败都是 `unresolved`。setup、timeout、无 pytest marker、Docker 或 cleanup 错误不构造假的 `Verification`，而进入 run 级 infrastructure failure。`Verification` 不含 reward、eligible flag、container、cleanup 或泛型 failure category；reward adapter 只做 `resolved→1`、`unresolved→0` 一次映射。类名已经拍板；字段仍保持 `result`，本报告不静默改名。[用户决策+建议]

`Observation.error_type`必须是首阶段工具合同中的受限枚举（例如`tool_error`），不是任意异常字符串或嵌套failure对象；基础设施异常不通过它传递。`timed_out`和`truncated`是正式控制/复盘事实，因此显式存在；其他工具附加值没有消费者时不进入metadata替代通道。[建议]

### 6.4 顺序、关联与wire边界

一个被接受并实际执行的tool call形成一个`Step`。同一assistant message含多个tool call时，按真实执行顺序形成多个连续Step；`Step.index`从0连续递增，是领域执行顺序和Action/Observation配对的唯一机制。因为Action和Observation直接嵌在同一Step中，所以不再需要`action_id`、`observation_id`、`Observation.action_id`、`event_seq`或新`Event`对象。[建议]

assistant 消息分组、tool call wire 格式和 tool-result 消息保存在 `messages.json`。核心模型不加入 `tool_call_id/provider_call_id/assistant_message_index/tool_message_index/completion_index`，即使字段可设为 `None` 也不加入。TRL 1.8 callable 不能稳定提供这些信息，旧消费者又属于 XML history/PolicyTrace；缺失它们不影响执行正确性。离线分析若需要，只能按 messages 和 Step 各自顺序 best-effort 对账，不回写核心 schema。[官方接口事实+代码事实+建议]

invalid/unknown/非法参数在执行前失败时不伪造Action、Observation或Step；conversation保留wire事实，`Trajectory.termination`说明交互终止原因。最终无tool的assistant消息也只在conversation和termination出现。[建议]

### 6.5 termination、verifier、reward与基础设施失败

四者是单向派生关系，不共享一个泛型`status/outcome/failure`字段：[建议]

```text
Trajectory.termination
  → 只解释Agent交互为什么结束
  → submitted / no_tool_call / invalid_tool_call / max_turns /
    truncated / invalid_after_submit / tool_timeout /
    infrastructure_interrupted

Verification.result
  → 只解释真实verifier对已提交patch的结论
  → resolved / unresolved

reward
  → policy termination（未形成可验证提交）= 0
  → Verification.unresolved = 0
  → Verification.resolved = 1

infrastructure failure
  → 无reward、不中性化为unresolved、立即使当前batch/run失败
```

`infrastructure_interrupted`只说明已经记录的 Agent 步骤为何停止，不是 episode outcome，也绝不能据此生成 0 reward；真正异常类型、traceback 和 cleanup 历史进入根级 `run.json`/`train.log`。核心 `Trajectory` 没有 `status/failure/outcome/policy_failure/verifier_status`。若基础设施错误发生在任何 Step 形成前，允许没有 `trajectory.json`；已获得的 messages 和资源清理事实仍由 rollout 文件与 `run.json` 记录。[建议]

### 6.6 metadata和落盘边界

第一阶段九个核心领域对象均不提供任意 `metadata: dict`。旧 metadata 中的正式事实按消费者决定：`Observation.exit_code/timed_out/truncated/error_type`显式建模；dataset revision、模型路径、依赖版本、seed、image inspect、container ID 与cleanup操作历史进入动态 `run.json`。没有控制流消费者的 provider response ID、timestamp、duration 和 changed-path convenience 字段不进入核心模型，必要时只写主日志。[代码事实+用户决策]

`Trajectory`只保存 task/environment 关联、连续 Steps 和 termination；不保存完整 messages、patch 正文/hash/path、`Verification`/ref、cleanup/ref、container ID、reward、run metadata、checkpoint 或 Trainer 状态。同一 rollout 目录天然表达 `messages.json/trajectory.json/final_patch.diff/verifier.json` 的归属；资源清理历史统一在根级 `run.json`，无需把文件关系编码回领域对象。[建议+用户决策]

### 6.7 四类事实的最终归属

| 类别 | 唯一主源 | 不再复制到哪里 |
|---|---|---|
| 领域执行 | `Task/Environment/Sample`与`Trajectory(Step(Action,Observation), termination)` | TRL messages不反向覆盖；Trajectory不装文件引用 |
| wire messages | TRL structured prompt/completion → `messages.json` | Action/Observation无wire ID/message index |
| patch/verification | 同一rollout目录的`final_patch.diff`、成功时`verifier.json` | 不嵌Trajectory；不在`run.json`复制正文 |
| cleanup | 根级动态`run.json.cleanup`，分别保留container、子进程、runtime handle的硬释放证据及全部操作历史；主进程内CUDA计数只作diagnostic | 不创建rollout级清理文件；不嵌Trajectory；不以allocator计数偏离单独判定泄漏 |
| reward/训练 | finalizer单次映射 + TRL内部batch/metrics | 不进入领域模型；不复制PolicyTrace/GRPOBatch |
| run/checkpoint | 根级config/run与Trainer原生checkpoint/final adapter | 不进入Task/Trajectory/metadata |

特别注意：LoRA hash变化只证明bytes改变。reward是否退化、advantage/gradient是否非零以及可训练参数是否变化都只作为本run观察值记录，不构成第一阶段系统闭环的通过条件。

### 6.8 专项复审的十三项结论

| 问题 | 结论 |
|---|---|
| 1. `Environment`是否保持原名 | **是。** 它是稳定领域名，模块已足够区分 |
| 2. 是否删除`EnvironmentSpec`建议 | **是。** 没有独立语义或消费者 |
| 3. 有状态运行类是否仍叫`SWEEnvironment` | **是。** 位于`environment.py`，与领域`Environment`不冲突 |
| 4. 核心模型模块名 | **`src/swe_agent/models.py`。** 不保留`core/`，不建立平行schemas/dto/records |
| 5. `Step.index`是否足够表达领域顺序 | **是。** 连续index同时表达顺序和Action/Observation配对 |
| 6. Action/Observation是否需要event sequence | **否。** 两者嵌在同一Step，重复序号没有消费者 |
| 7. 是否删除未证明的wire引用 | **是。** 核心schema没有call ID或message/completion index |
| 8. Trajectory最终字段 | **`task_id/environment_id/steps/termination`**，没有其他字段 |
| 9. Evaluation是否持有gold patch | **否。** gold仅供固定资产资格；运行时Evaluation只含offline evaluator script |
| 10. metadata允许在哪些领域对象 | **一个也不允许。** run附加事实进入run/output文件，不回流核心模型 |
| 11. 是否保留Sample | **是。** 它是loader/factory间唯一公开Task+Environment组合及配对校验边界，不是TRL Dataset row |
| 12. 是否需要新增领域对象 | **否。** 现有九个名称已经覆盖第一闭环；不新增Event/Context/Bundle/Record |
| 13. 是否删除目标设计中的旧包名 | **是。** 唯一源码/导入根是`src/swe_agent`/`swe_agent`；本地仓库和报告文件路径不构成包名 |

## 7. AgentLoop 拆分结果

| 旧 `AgentLoop` 职责 | 新归属 | 决策 | 说明 |
|---|---|---|---|
| system/user prompt 初始构造 | dataset/prompt builder | 拆分后部分保留 | 保留 SWE 指令；改 native tool calling，删除 exact-one XML 文案 |
| assistant generation | TRL | 由TRL替代 | 不保留 `VLLMOpenAIModelClient` 外层循环 |
| history 续接/chat template | TRL/Transformers | 由TRL替代 | 不再 `_render_messages()` 重建/重解析 |
| raw token/logprob 捕获 | TRL | 由TRL替代 | 仅记录公共训练指标 |
| Qwen XML exact-one parse | 无 | 不再保留 | native response parser 取代；参数校验另保留 |
| provider/tool schema 交叉校验 | tokenizer/tool adapter tests | 拆分后部分保留 | 保留生成 schema/arguments/领域 validator 契约，不要求 XML/wire 双源一致 |
| Action 构造 | `SWEEnvironment` adapter | 复制后适配 | canonical validated call 是领域事实 |
| ToolExecutor 调用 | `SWEEnvironment` | 复制并基本保留 | public methods 转到同一 executor |
| Observation 构造 | executor/recorder | 复制后适配 | 同一对象单向转 tool message |
| Step/Trajectory 记录 | domain recorder | 复制后适配 | 删除训练概率重复字段 |
| git diff/submit | environment + sandbox | 复制后适配 | submit terminal gap 采用两阶段结束 |
| invalid/max_steps/termination | TRL loop + finalizer | 拆分后部分保留 | 统一策略失败 0；max iterations 显式 |
| finally cleanup | `SWEEnvironment.reset/finalize/close` + runner 普通引用列表 | 复制后适配 | 适配 pooled reuse；必须有独立于 reward 调用的顶层兜底 close |

TRL 原生路径无法完全表达的旧语义包括 **submit 立即终止**、“首次 invalid call 立即结束”和 exact-one call transport约束。前两者按第 4.2 节适配；invalid call 不再强制立即 break，TRL 会返回 tool error并允许模型继续，但 finalizer仍把 episode判0。exact-one是旧 XML parser的传输合同，不是必须迁移的SWE领域能力：第一版接受 TRL一次 assistant message中的多个**同步** tool calls并按顺序记录；submit 后的后续调用由环境拒绝并判0。不要为恢复 exact-one另写外层 parser。

## 8. Docker 与 environment 详细边界

### 8.1 生命周期

定向复核结论：**原报告的“environment拥有episode资源、fresh verifier、pooled reset、runner finally”方向足够，修正下述部分失败和中断合同后，可以支持单作业内的正常结束、策略失败、基础设施异常、Trainer/生成后端异常及best-effort用户中断。无需新增生命周期管理类；实现增量仅是 `DockerSandbox` 的显式ID/可重试close、`SWEEnvironment` 的清晰handle与收束方法、runner的一份普通environment list，以及薄CLI SIGTERM边界。**[代码事实+官方源码事实+用户决策]

```text
factory()              创建无 Docker 副作用的轻量 environment
                       立即追加到 runner 的普通引用列表（含 probe/pool 扩容实例）
  ↓
reset(task fields)     若仍持有上轮 container，先 close
                       close 失败则本次 reset 以 infrastructure failure 结束，禁止创建新容器
                       绑定 Task/Evaluation handle、生成 episode_id
                       创建 fresh rollout DockerSandbox；create 返回后立即保存 container ID
                       start 并校验 image、HEAD、clean tree
                       任一后续初始化失败均尝试 close
  ↓
tool methods           validate → Action → ToolExecutor → Observation → recorder
  ↓
submit()               冻结 patch；标 submitted；环境进入 read-only terminal-pending
  ↓
reward_func/finalize   对齐 completion/tool-call evidence
                       策略终止检查
                       冻结 live container 中的 patch/必要诊断
                       close rollout sandbox
                       fresh verifier sandbox（仅合法 non-empty submitted patch）
                       verifier 自己 close；把verification与cleanup事件留在本地缓冲；返回0/1
  ↓
下一 reset / finally   reset 再次确认无旧 container
                       顶层 finally 独立于 reward/finalize，对所有已构造 env 调幂等 close
```

TRL 1.8 在 `GRPOTrainer.__init__()` 中会为每个 factory 构造一个 probe，并把它作为 pool 首个实例；batch 需要更多并行实例时，`_generate_and_score_completions()` 再调用 factory 扩容，随后对复用实例调用 `reset()`。源码没有 environment `close()` 调度或 Trainer 异常时的 pool cleanup。因此，项目提供给 TRL 的 factory 必须在**每次返回前**把新实例追加到当前 runner 的普通 Python list，runner 的最外层 `finally` 遍历该 list。constructor 禁止创建 Docker，所以仅用于 schema probe 的实例正常情况下 `close()` 是 no-op；若 Trainer 初始化中途失败，仍由同一 finally 收束。[官方源码事实+建议]

这份 list 不是 registry：没有注册协议、锁、租约、心跳、owner 认证、跨进程查询或 stale recovery，只解决“本进程已经构造了哪些 environment”。无需新增 lifecycle manager 类。[用户决策]

#### 8.1.1 唯一所有权

| 资源 | 唯一直接持有者 | 正常清理点 | 最终兜底 |
|---|---|---|---|
| rollout container | 当前 episode 的 `SWEEnvironment.rollout_sandbox`，至多一个 | `finalize()` 冻结证据后调用 `close()`；下一次 `reset()` 前再次检查 | runner `finally` 遍历全部 environment 的 `close()` |
| verifier container | `SWEGymVerifier.verify()` 内的局部 `verifier_sandbox` | verifier 自己的 context/finally | verifier 调用栈的 finally；失败结果传回 environment/run evidence |
| pool 中跨 batch 复用的 environment | TRL pool 复用对象；项目 runner 只保留普通引用用于退出清理 | 每次 `reset()` 先收束旧 episode | runner `finally` |
| Trainer 初始化 probe environment | TRL pool 的首个实例，同时在 runner 普通 list 中 | constructor 无 Docker；若以后被 reset，则与普通 pool 实例相同 | runner `finally`，无 active handle 时 no-op |

verifier handle 绝不能写入 `SWEEnvironment.rollout_sandbox`，也不允许 `DockerSandbox` 存一个可被 rollout/verifier轮流覆盖的“当前 container”。生产顺序建议先冻结 patch、关闭 rollout，再创建 verifier，减少同时存活资源；即使单元测试故意让两者同时存在，两个 handle 也必须互不覆盖。[建议]

#### 8.1.2 create 的部分失败与 `DockerSandbox.close()` 合同

`DockerSandbox` 只需一个小状态机，不需要新的管理组件：[建议]

```text
new（无 ID）
  → docker create
  → acquired（立即保存 create 返回的 container ID）
  → docker start
  → running
  → base/clean 校验通过
  → ready

任一 acquired 之后的失败 → close(container ID)
close 成功或明确 No such container → closed（清除 handle）
close 失败 → cleanup_failed（保留 ID，可再次 close）
```

- `docker create` exit 0 后必须先解析 stdout container ID 并写入 sandbox，再执行 start、base/clean 校验、environment状态绑定或rollout目录/recorder初始化。若输出初始化发生在ID已获得之后，其异常路径同样必须close；能够在create前完成的目录与纯内存初始化则应前移。
- create 超时/错误、或 exit 0 但 stdout ID 缺失/不可解析时，只允许在**同一次 create 尝试内**按该 sandbox 唯一生成的精确名称做一次 best-effort inspect：若查到 ID，立即纳入上述 close；若 Docker daemon 不可用或仍无法确认，则记录名称、`container_id_unknown` 和 infrastructure failure，不能宣称已清理。这不是启动时 stale recovery，也不引入 owner receipt。
- `close()` 只负责释放资源，不计算 reward、不运行 verifier、不写主 trajectory。无 ID 或已经成功 closed 时为 no-op；有 ID 时执行 `docker rm -f <ID>`。成功或明确不存在才清除 handle；超时/非零/daemon 错误保留 ID 和错误并抛出 cleanup infrastructure failure。
- `close()` 必须可重复调用：成功后的重复调用不再访问 Docker；失败后的重复调用必须真正重试，而不是返回缓存的第一次失败。每次尝试均写入内存中的 cleanup evidence。
- context manager 可以保留为 `open()/close()` 的语法糖，但不能成为唯一清理入口；部分构造失败和 runner 顶层 finally 必须调用同一个 `close()`。

#### 8.1.3 `reset()`、`finalize()` 与 runner `finally`

三个操作的职责必须分开：[建议]

- `reset()`：如果 environment 仍有上个 episode 的 container handle，先调用 `close()`。本次 close 一旦失败，当前 reset 立即以 infrastructure failure 结束；即使稍后 runner 重试成功，本次 reset 也不得继续创建新 container。只有确认旧 handle 已清除，才能建立新的 episode。
- `finalize()`：正常 episode 收束。它只做一次 completion/termination 对账，冻结 patch 和必须从 live rollout container 读取的证据，关闭 rollout，随后才运行 fresh verifier并产生唯一 binary outcome。若 rollout cleanup 失败，不启动 verifier、不返回 0；若已经完成 verifier而最终清理失败，也不返回普通 reward。为避免重复计分，重复 finalize 应复用已冻结的 patch/verifier事实；若此前只是 cleanup 失败，可重试 cleanup，但不得重跑 verifier制造第二个 reward。
- `close()`/`cleanup()`：只释放当前 environment 的 rollout 资源，允许无 episode/no handle，且可重复调用；它不能假设 reward function 曾经或将会执行。
- runner 顶层 `finally`：覆盖 Trainer 构造失败、生成后端异常、tool loop 异常、reward/finalize 从未调用、Python 异常和用户中断。它以run为唯一资源所有权边界，对factory已构造的**全部**environment逐一做best-effort退出证据冻结和`close()`，随后收束Trainer、vLLM、已知worker和本run资源句柄；不能因一个资源cleanup失败而跳过其余资源。environment/verifier每次cleanup操作只追加本地结构化事件；main process在每个`close()`返回或抛错后立即drain该缓冲并原子更新`run.json`，最后再写汇总状态。不创建第二种清理产物，也不是新的lifecycle manager。Trainer/vLLM只在run开始时构造一次，在run结束或失败时统一释放，不在group之间销毁重建。

最外层控制流应保留一个 `primary_error`，并使用 `except BaseException` 只为记录后再原样传播；`finally` 总会执行。若没有原始错误而 run级cleanup出现未恢复失败，首个cleanup error成为`failure.category="cleanup"`的primary failure，其余cleanup errors结构化记录。不能把“TRL 会调用 custom reward”作为容器清理前提。[建议]

#### 8.1.4 verifier 生命周期

旧 `SWEGymVerifier.verify()` 已经每次调用 factory 构造 fresh sandbox，并用 `with sandbox` 包住 patch check/apply和测试；这是正确边界，应复制后适配。新实现需要强化以下合同：[代码事实+建议]

1. verifier factory 构造、container create/start、patch apply、测试执行、marker/结果解析的任一步异常，都进入 verifier 自己的 `finally`；只要已取得 verifier container ID，就尝试 `close()`。
2. verifier sandbox 是局部变量；rollout sandbox 仍属于 environment。两者的ID、cleanup操作历史和错误分别记录，永不互相覆盖。
3. verifier业务结果与 cleanup 结果分开。测试 failed 是合法 `unresolved=0`；verifier setup/parse/cleanup failure 是 infrastructure failure。cleanup failure不得降格为 reward 0。
4. 若 verifier执行异常和 verifier cleanup异常同时存在，执行异常是 primary，cleanup错误和可能残留的 verifier container ID另记；若只有 cleanup失败，则以`failure.category="cleanup"`记录primary，不归入policy、verifier或trainer失败。

#### 8.1.5 异常优先级、输出写入顺序与“已清理”声明

最小异常合同如下，不引入异常框架：[建议]

- 已有业务/基础设施异常时，保留同一个异常为primary failure并原样抛出；cleanup failure作为独立事件写入对应的`run.json.cleanup.containers[].operations[]`、`processes[].operations[]`或`runtime_handles[].operations[]`，进程内CUDA计数仅写入`gpu_diagnostics[]`，`run.json.failure`只保存primary failure摘要，同时可用`BaseException.add_note()`补充终端诊断，但不能用cleanup异常覆盖原异常。
- 没有原异常时，未恢复的cleanup failure本身成为`category="cleanup"`的primary failure。runner仍继续清理其余environment与run级资源。
- `run.json.cleanup`严格使用第18.2节冻结的`state/clean_release/residuals/containers/processes/runtime_handles/gpu_diagnostics`字段。初始为`state="pending"/clean_release=null`；所有已知资源均完成硬释放检查后才可写终态。`clean_release=true`只要求：本run container均已删除或确认不存在；本run明确启动的worker PID均已退出或确认不存在；vLLM engine shutdown已成功返回；environment、Trainer和vLLM等本run持有的资源句柄已关闭或释放；不存在未恢复的cleanup error或已确认残留。任一硬条件不满足时写`state="failed"/clean_release=false`并记录residual。
- `torch.cuda.memory_allocated()`、`torch.cuda.memory_reserved()`及其与run前数值的差异只能作为主进程退出前的诊断观察，不能单独把GPU标为residual、把`cleanup`写成failed、把`clean_release`写成false或改变run lifecycle。CUDA context与caching allocator可在主进程退出前合法保留显存；本进程不能在`run.json`中证明自身退出后的GPU状态。若未来确需主进程退出后的GPU占用验收，只能由父launcher在训练子进程退出后另行观察，本阶段不把它加入run内硬门禁。
- 某次cleanup operation失败后仍须继续best-effort收束；若后续重试使资源通过最终硬检查，可将cleanup写为completed并保留完整失败/恢复历史。只有未恢复失败或确认残留才令cleanup failed。核心系统闭环是否通过仍由第15节并列验收事实独立判定；cleanup不得启动新的训练run，也不得覆盖先前primary failure。不得另造未在第18.2节schema中的同义字段。
- colocate vLLM若在主进程内持有engine而没有独立worker PID，不制造虚假process记录；runner显式调用engine shutdown、释放模型/Trainer引用并在`runtime_handles[]`记录返回结果，同时可在`gpu_diagnostics[]`记录cleanup前后进程内CUDA计数。真实子进程只记录本run启动并能明确识别的PID，逐个terminate/join/verify-exit；不扫描或治理其他作业。主CLI自身退出由终端退出码证明，进程内`run.json`不能承诺在自身退出后再写一次状态。[建议]

正常 finalize 和异常退出都遵守最短的证据/清理顺序：[建议]

```text
先在内存中冻结 episode/task/container 标识和 recorder 状态
→ 对仍可响应的 rollout container，用有界操作读取 patch/changed paths/必要命令证据
→ 尽快 close rollout container
→ 正常路径再以 fresh verifier验证已冻结 patch，并由 verifier 自己 close
→ container删除后写最终trajectory/verifier文件并由单写者原子更新run.json
```

只有patch、changed paths和需要在container内读取的诊断必须在删除前取得；messages、Action/Observation、异常、`Verification`、cleanup operations和最终run状态都可从内存于删除后落盘。异常退出时读取diff只能best-effort且有界，失败要记录但不能阻止cleanup；不能为了写输出长期保留container。run的`output_dir`应在创建Docker前建立；batch/group/rollout目录由recorder按已资格化的TRL generation顺序分配。若落盘本身失败，仍按相同finally清理资源，并至少向终端/`train.log`报告无法更新`run.json`。[建议]

#### 8.1.6 用户中断和不可恢复终止边界

旧 CLI 的 `_SignalBoundary` 同时安装 `SIGINT/SIGTERM` handler：首次 SIGINT抛 `KeyboardInterrupt`，首次 SIGTERM抛 `WorkflowTermination(BaseException)`；`SingleInstanceWorkflow._run_with_final_cleanup()` 捕获 `BaseException`并在 finally清理，CLI返回130/143。相应 subprocess测试已验证 SIGTERM路径。这证明新入口适合保留一个**很薄的 CLI级**终止边界，但不应恢复旧资源 owner/stale体系。[代码事实+建议]

- 普通 Python异常和 `KeyboardInterrupt`：顶层 `BaseException`/`finally` 自动 best-effort冻结证据并清理。
- `SIGTERM`：首次信号在Python主线程抛一个专用`BaseException`，让同一finally展开，并最终返回143；cleanup期间的重复信号不重复抛出。Python handler可能被长时间C/CUDA调用或阻塞的子进程等待延迟，因此只能承诺best-effort，不能承诺固定时间内完成。[用户决策]
- Trainer/生成后端异常：只要异常回到主 runner并触发 Python展开，就由同一 finally清理；若外部 launcher直接杀死进程，则落入下一条边界。
- `SIGKILL`、主机崩溃、`os._exit`、进程被运行时强制终止，以及Docker daemon崩溃期间无法执行remove：进程内代码无法保证清理。只依靠已知container ID/唯一名称、run/episode/purpose labels和已落盘的rollout/status信息供人工`docker inspect`/确认后清理；不实现自动stale recovery。daemon恢复后也不自动扫描或删除。

### 8.2 rollout 隔离和 ID

- 每个同时 active completion 使用 pool 中不同 environment instance；每次 `reset` 必须创建新容器；
- 不依赖 Dataset 人工注入 generation index，也不为environment增加持久化pool-slot字段。每次 reset生成新的`episode_id`；reward function只在当前调用内按`environments`与`completions`同位置处理，并以`episode_id`落盘。runner普通引用列表的内部位置只用于遍历cleanup，不能成为group member identity；
- environment 保存 Docker create 返回的明确 container ID，并只清理该 ID；container name/label 可包含 run/episode/task/purpose (`rollout`/`verifier`) 以便日志关联、create结果含糊时的同次定向 inspect和不可恢复崩溃后的人工定位，但不实现 owner label/receipt 认证；
- verifier 一律 fresh container，不复用模型修改过的 rollout repo；正常 finalize 建议先冻结 patch并关闭 rollout，再启动 verifier；
- 第一阶段固定串行运行 verifier，避免同组CPU-heavy evaluator争用；这会增加 rollout→reward 延迟，但不改变训练语义。

### 8.3 callable 适配

六个工具固定为 `list_files/read_file/search_code/edit_file/run_command/submit`，能力和失败语义以旧 `runtime/tool_spec.py::TOOL_SPECS`、`validate_tool_arguments()`及`runtime/tools.py::ToolExecutor`为行为基线，不重新设计另一套工具集。[代码事实+用户决策]

TRL/Transformers 从 Python signature、type hints 与 docstring 生成 native schema，并把已解析的 `arguments` 直接作为关键字参数调用 bound method。wrapper 不解析 XML/JSON 字符串，不接收 provider call ID，也不重建 wire call。最小调用链是：[官方接口事实+建议]

```text
SWEEnvironment.read_file(path, start_line, end_line)
  → 由typed kwargs构造一次canonical arguments dict
  → validate_tool_arguments("read_file", args)
  → Action(tool_name, arguments)
  → ToolExecutor.execute(Action)
  → record Step(index, Action, Observation)
  → return Observation的稳定字符串payload
```

函数 schema 与执行校验不是重复解析：前者指导模型并完成基本参数绑定，后者保留 required/unknown/type/range、`edit_file`按operation的互斥字段以及`read_file`行范围等领域合同。需要比较 wrapper 生成 schema 与旧 native schema 的语义等价性；不迁移 XML protocol digest，也不为了输出再造第三份 schema。[代码事实+建议]

`ToolExecutor.execute(action: Action) -> Observation` 直接消费 `action.arguments`；旧实现中的 `action.tool_args`、`Action.metadata`、action/observation ID、timestamp和 provider wire 字段全部删除。executor 内部局部 tuple/命令结果不是新的领域对象，最终只构造一次 `Observation(text, exit_code, error_type, timed_out, truncated)`：[用户决策]

- 所有文件操作仍在 `DockerSandbox` 的 `/testbed` 内执行，resolve 后拒绝越界和任一路径段 `.git`；UTF-8读取、带行号窗口、文件/匹配数量上限原样保留。
- `edit_file`保留 `replace/insert/create`；replace 必须 exact-once，create 必须目标不存在；任何成功返回前必须再次取得 `git diff` 且非空，否则形成 `tool_error`。diff可作为工具 observation 的有界文本，但最终patch唯一落盘到`final_patch.diff`。
- `run_command`保留旧 denylist（Docker/Podman、apt、pip install、触碰Docker socket/系统路径）、固定`/testbed`与镜像内conda环境、`timeout_sec`向`max_timeout_sec`取上限。该 denylist 是固定实验执行边界，不扩张为网络安全模块。
- 所有工具输出统一经过字符上限；发生截断时 `Observation.truncated=true`，不能只在文本后缀中暗示。命令超时写 `timed_out=true`；容器健康且命令自身超时属于策略失败，Docker plumbing故障仍走 infrastructure failure。
- `submit`仍不接受参数，读取当前 git diff 并拒绝空patch。executor成功返回后，environment冻结同一patch并进入terminal-pending；不另写submit parser，也不允许 wrapper 绕过executor的non-empty校验。

每个 wrapper 只做“typed kwargs → 一次validate → Action → execute → Step →稳定字符串”这一条路。unknown tool由TRL记录为wire失败；能够进入wrapper的非法参数由validator形成策略错误。执行前被TRL拒绝的call不伪造Step；真正进入executor后，即使得到工具领域错误，也按一次实际执行尝试记录Action/Observation。该边界须用锁定TRL版本测试确认。[官方接口事实+建议+待实施验证]

环境方法内部必须区分 `ToolError` 与 `DockerRuntimeError`。前者形成显式 error `Observation`并在 finalize 时得到相应 termination/reward 0；后者在 environment 内保存 infra 标记和资源事件，随后抛出。TRL 可能先把 tool 异常转成 tool result，因此 finalizer 必须再次拒绝计分；若更早的 `reset`或 generation 失败，则 run-scoped `finally`兜底并由 `run.json`单写者记录。这是最小必要生命周期逻辑，不需要再建通用 manager。

### 8.4 异常分类

| 情况 | episode 结果 | verifier | Trainer 行为 |
|---|---:|---|---|
| unknown tool / schema 不合法 / ToolExecutor 领域边界拒绝 | 0；termination=`invalid_tool_call` | 不构造结果 | 保留 completion、进入 group |
| tool timeout（命令本身在允许范围、容器健康） | 0；termination=`tool_timeout` | 不构造结果 | 保留 completion |
| max tool turns/总长度、无 submit、submit 空 patch、submit 后继续调用 | 0，具体 termination | 不构造结果 | 保留；截断 mask 按配置 |
| patch apply 正常失败、真实 pytest failed | 0，`unresolved` | 已运行 | 保留 |
| pytest passed | 1，`resolved` | 已运行 | 保留 |
| Docker daemon/image/base mismatch、exec plumbing、verifier setup/timeout/无 marker、cleanup 失败 | 无 reward；前者按真实infrastructure category，cleanup-only按`category="cleanup"` | 不构造`Verification` | raise，中止本 generation batch；不得写成 0 |

命令 timeout 是否策略归因需看容器健康与命令本身：旧 verifier timeout 属 infra；agent 主动运行超时命令属于策略失败。该区分应有显式枚举而非异常字符串猜测。

## 9. 模型、量化、tool template 与资源配置边界

### 9.1 集中配置，而非插件框架

`configs/`根目录只放两份完整、独立、无继承的 YAML 配置入口；不使用配置引用、include、extends、overlay、继承或模型子目录：[用户决策]

```text
configs/
├── grpo_swegym_qwen2_5_coder_7b_lora.yaml
└── grpo_swegym_qwen3_coder_30b_a3b_qlora.yaml
```

每份 YAML 都完整包含 dataset/task、Docker、model/tokenizer、dtype/quantization、PEFT、chat/generation、GRPO、vLLM/runtime、output root、logging/save 参数。重复少量稳定字段是有意选择：每个文件都可独立解析并完整表达对应运行方案，不会因隐式基类或覆盖顺序改变语义；在获批实施后可作为`grpo.sh`的单一配置输入。**配置字段完整不等于目标组合已经可执行或已经通过资格**：7B LoRA是第一阶段主路径，30B-A3B QLoRA是后续目标，二者分别受第1.4、9.2、9.3节资格约束，7B结果不能外推到30B。config loader只做严格解析和跨字段校验，不拼装配置继承。environment、tools、verifier不读取模型名和vLLM参数；训练入口不按模型名写业务分支，也不为两份配置预建插件框架。[用户决策+建议+待实施验证]

### 9.2 第一阶段 7B LoRA 完整配置合同

| 配置面 | 固定值/合同 | 说明 |
|---|---|---|
| model | provenance ID `Qwen/Qwen2.5-Coder-7B-Instruct`；本地入口固定`/home/2025user/zyp/.cache/modelscope/hub/Qwen/Qwen2.5-Coder-7B-Instruct`，run记录其解析后的实际路径和可观测revision | 7.61B dense，28 Q heads/4 KV heads；本地文件齐全不替代目标栈加载资格 |
| context | 32768 | 不改RoPE；实际tokenized prompt≤8192，`max_completion_length=22528`，另保留2048 tokens margin；tool输出、assistant输出和消息边界共享completion预算 |
| base precision | BF16 | A100原生支持；这是未量化base |
| quantization | disabled；`load_in_4bit=false`且不传`BitsAndBytesConfig` | 配置测试必须拒绝“文件名LoRA但实际启用4-bit” |
| LoRA | r16/alpha32/dropout0；target固定`q_proj,k_proj,v_proj,o_proj` | 启动前要求四类target全部精确存在且只有预期LoRA参数可训练；不使用`all-linear`或自动扩展 |
| GRPO | binary reward、DAPO、beta0、`num_generations=4`、`num_iterations=1` | group固定4；reward分布和更新数值只如实观察 |
| generation | temp1/top-p1/top-k0；`max_tool_calling_iterations=20`；`max_completion_length=22528`；margin 2048 | 禁用grammar约束；动态工具输出与消息边界计入总completion预算，安全余量不可被运行时借用 |
| generation backend | colocate vLLM、TP1、sleep true、`use_vllm=true` | 以非量化BF16推理并验证PEFT merge/full sync；任一gate失败停止，配置保持不变 |
| Trainer | 单GPU、第一阶段验收run使用`max_steps=1`、batch/accumulation均为1、checkpoint每step保存且最多保留2个 | 不引入FSDP/DeepSpeed；一个run只构造一次Trainer/vLLM并执行一次基于在线group的GRPO optimizer step；不保证非零parameter update |

本机有4×A100 80GB。第一阶段7B固定以单GPU TP1资格vLLM；wheel/ABI、HBM、sleep/wake或同步失败时按第12.2节停止并记录，不改`use_vllm`、memory utilization或GPU拓扑，也不建立GPU owner/claim/lock。该7B选择不推导30B拓扑。[运行事实+用户决策+待实施验证]

7B完整YAML固定一个`seed`作为本run复现输入，不包含run数量、循环索引、自动重启或跨run选择字段。一次正式CLI调用只生成一个UTC-Z `output_dir`，只构造一次`GRPOTrainer`与colocate vLLM，并在该run内维持唯一policy、optimizer和scheduler状态。第一阶段验收run使用一个真实group与`max_steps=1`；reward全0、advantage为0、gradient为0或LoRA参数不变都按实际结果记录，不触发另一个run。[用户决策]

正式CLI只打印本run的终端摘要，字段限于`run_id`、`lifecycle`、`native_policy_path_reached`、`trainer_group_consumed`、`system_closed_loop`、`failure`引用、`final_model_ref`和`cleanup`摘要；不创建跨run summary、跨run目录、软链接或adapter副本。任何execution failure或interruption都结束当前CLI并保留当前run；正常policy未达到native path也结束当前CLI，但不写failure。第一阶段没有自动重试。用户随后显式再次执行CLI时创建的是新的独立run，该动作不由前一个run的reward、gradient或cleanup状态自动触发。[用户决策]

正式层级关系固定为：[用户决策]

```text
一次CLI调用
└── 一个run / 一个output_dir
    └── 一套Trainer + colocate vLLM + optimizer/scheduler + 持续policy状态
        └── run内部generation batch
            └── 同一prompt的GRPO group
                └── 4条rollout
```

group及其多条rollout是GRPO在run内部的采样与训练结构，不是独立作业；Trainer/vLLM和policy状态的构造、持有与释放均以run为边界。

`generation_batch_size`、`steps_per_generation`、`gradient_accumulation_steps`和process数由锁定`GRPOConfig`校验。完整7B YAML必须显式记录这些值、`num_generations`、`max_steps`和save策略；测试证明group整除、generation batch边界以及预期optimizer step。第一阶段验收run固定只有`batch-0000`、`global_step_at_generation=0`和一次成功后的`trainer.state.global_step=1`；train runner在成功返回后把公开state写成`consumed_by_global_steps=[1]`，失败前保持空数组。不为该记录关系添加callback或Trainer subclass。batch/group索引属于run内部结构，不表示新的run层级。[官方接口事实+用户决策]

### 9.3 后续 30B-A3B QLoRA 完整配置合同

旧项目使用同一本地 Qwen3-Coder-30B-A3B-Instruct，加载BF16 base并对attention projection加PEFT，曾以TP2/FSDP尝试运行；这些日志只提供模型路径、模块名和资源失败基线，**不能证明目标4-bit QLoRA路径**。[代码事实]

| 配置面 | 30B目标 | 与7B关系/资格要求 |
|---|---|---|
| model | provenance ID `Qwen/Qwen3-Coder-30B-A3B-Instruct`；本地路径固定`/home/2025user/zyp/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct`，run记录可观测revision | 只换完整YAML，不改SWE业务模块；文件存在不等于目标组合通过 |
| architecture | 48层MoE、128 experts/top-8、32 Q heads/4 KV heads | 旧本地`config.json`可证明结构；不证明运行组合 |
| quantization | BNB 4-bit NF4、BF16 compute、`bnb_4bit_use_double_quant=true` | 与7B明确不同；必须看到真实`Params4bit`和冻结base；不在资格时改变量化方案 |
| LoRA/PEFT | r16/alpha32/dropout0；target固定`q_proj,k_proj,v_proj,o_proj` | 必须枚举MoE模块并要求四类target精确命中；不启用expert层或`all-linear`，失败即停止 |
| chat/tool | Qwen3模板走同一TRL response/template资格 | 只换fixture/config，不写模型名业务分支 |
| generation backend | 固定表达colocate vLLM+sleep、`use_vllm=true`，同时保持`runtime_qualified=false` | 后续30B阶段必须实测BNB realization、PEFT同步与资源拓扑；第一阶段不运行，也不预选TP/DP/FSDP |
| runtime/topology | `runtime_qualified=false`；TP/DP/FSDP字段在第一阶段保持未激活，不传给Trainer | 30B不在第一阶段运行；后续必须基于真实QLoRA显存、vLLM能力和4×A100形成已资格的运行配置 |
| MoE loss | 不额外启用router auxiliary loss | 记录模型原始config，但训练入口不增加router-aux专属业务逻辑 |
| context | 32768 | 不因模型支持262K放大KV/activation压力 |

第一阶段对30B只做静态合同：目标Transformers架构名、BNB NF4 double-quant字段、固定q/k/v/o PEFT target、context、vLLM字段和`runtime_qualified=false`必须可严格解析，CLI必须拒绝用这份尚未资格的配置启动模型。后续30B阶段才实测MoE/BNB/forward/backward、vLLM realization与PEFT同步以及4×A100资源布局；本报告不凭7B或旧BF16日志预选TP1/TP2/TP4、DP或FSDP。[用户决策+待实施验证]

未来30B资格完成后仍只选择30B完整配置文件；`SWEEnvironment`、六工具、Docker、verifier、reward、recording和Trainer主构造流程保持不变。第一阶段不预埋模型专属补丁或资源状态机，也不把尚无证据的拓扑写成既成事实。

## 10. 模块迁移矩阵

每个关键模块只使用用户指定的六类决策之一。

| 旧模块/职责 | 现状 | 新项目决策 | 原因 | TRL是否替代 | 适配工作 |
|---|---|---|---|---|---|
| 旧领域模型中的`Task` | 严格公开任务schema，但含无消费者metadata/prompt策略字段 | 复制后适配 | repo/base/problem是SWE领域事实 | 否 | 迁入`src/swe_agent/models.py`；保留`task_id/repo_name/base_commit/problem_statement`，删除source/hints/prompt_policy/metadata |
| `Environment`/`Evaluation`/`Sample` | 名称正确，但Environment重复Task字段，Evaluation混合资格oracle与运行输入，Sample含无消费者可选对象 | 复制后适配 | public/private边界有价值，字段所有权需收口 | 否 | 保持`Environment`原名；Environment只持image/limits/timeouts；Evaluation只持offline script；Sample只持Task+Environment；loader直接返回sample/evaluation |
| `Action/Observation` | canonical工具语义与旧XML/provider linkage混合 | 复制后适配 | 执行动作与实际结果仍是领域事实 | 否 | Action只保留tool+validated arguments；Observation显式exit/error/timeout/truncated；删除所有ID、event_seq、wire refs和metadata |
| `Step/Trajectory` | 同时含messages、概率旁路、patch/status/failure和领域事实 | 拆分后部分保留 | 保留复盘，删除conversation/输出/run重复 | 部分 | Step固定一对Action/Observation且只有index排序；Trajectory只留task/env、steps、termination |
| `swegym/loader.py` | 双表/revision/hash fail-closed | 复制后适配 | 已验证task资产 | 否 | 去除`WorkflowConfig`耦合；只读新项目`data/swegym`和`assets/swegym/<instance>`；资格后直接返回sample/evaluation，不迁移历史`qualification_status` |
| 固定 Parquet/qualification patch/scripts | 固定真实样例 | 复制并基本保留 | 第一阶段指定事实依据且新项目不得运行时读旧仓 | 否 | 精确复制两份Parquet（约43.6MB/0.54MB）到`data/`；必需selected/eval/gold/test和小型manifest进`assets/swegym/getmoto__moto-7023/`；private由接口保证，不用目录名模拟安全边界 |
| `tool_spec.py` schema/validation | 六工具native schema + required/type/range/跨字段校验 | 拆分后部分保留 | schema价值与XML digest混合 | 部分 | 固定六工具；signature只生成wire schema，执行前只做一次canonical dict领域校验；删除XML digest |
| `tool_protocol.py` raw XML parser | exact-one Qwen XML | 不再保留 | native response parser 取代 | 是 | 历史 invalid fixtures仅可转测试数据，不进生产 |
| `ToolExecutor` | 六个真实工具；`/testbed`、`.git`拒绝、exact-once edit、diff检查、command denylist、timeout/output cap、non-empty submit | 复制后适配 | SWE核心成熟实现且与资格镜像共同定义行为 | 否 | `execute(Action)->Observation`改用`action.arguments`和新字段；env thin wrappers调用；不保留IDs/metadata/provider字段，不重复解析native call |
| `SYSTEM_PROMPT` | 强制 XML/exact-one | 拆分后部分保留 | SWE修复指令有用，协议文案过时 | 部分 | 在`prompts.py::build_prompt(task)`中只保留领域指令，改为native tools、多轮和submit terminal-final-turn说明 |
| `AgentLoop` | 自己 generation→tool→history | 拆分后部分保留 | 领域记录/终止需保留，通用 loop被替代 | 是（控制面） | 按第7节分散到 prompt/env/recorder |
| `model.py` OpenAI vLLM client | 自采 token/logprob/EOS | 由TRL替代 | TRL持有生成与概率 | 是 | EOS历史回归测试迁到 template adapter测试 |
| `chat_template.py` | Qwen token对齐 | 由TRL替代 | TRL/Transformers负责 | 是 | 只保留资格测试，不保留运行转换器 |
| `recorder.py` | 依赖旧Step状态机、AgentLoop逐轮exact-one行为 | 拆分后部分保留 | 领域事实有独立消费者，但类不能原样独立复制 | 否 | 保留连续追加/一次finalize意图；Step只用index且固定一对Action/Observation；无wire refs |
| `DockerSandbox` | fresh/no-net/no-mount/limits及owner receipt混合；依赖`DockerConfig`和`atomic_write_json`；旧 cleanup失败后不可重试且按名称删除 | 复制后适配 | 成熟 SWE隔离有价值，但旧owner认证/receipt和cleanup缓存状态机不进入新边界 | 否 | 迁移最小config；create成功立即保存ID；按ID删除；失败保留handle供重试；label仅关联日志；支持pooled reset和幂等close |
| image qualification helpers | image ID fail-closed；`RepoDigests`非空时才校验expected digest | 复制后小幅适配 | 镜像身份与复现证据，不是供应链安全认证；当前本地digest不可观察 | 否 | 只允许固定image；记录ID/platform/Size/RepoDigests；缺失/不匹配直接失败，不pull、不建立签名/receipt协议 |
| `SWEGymVerifier` | fresh sandbox、真实 pytest marker；依赖私有`Evaluation`和Docker类型 | 复制后适配 | reward事实基础 | 否 | 返回`Verification`；保留fresh/check/apply/offline-script/marker合同；空patch在调用前判policy failure；infra由finalizer raise |
| `PolicyTrace` | 自研概率完整 sidecar | 不再保留 | 与TRL token/logprob/mask重复 | 是 | 只在`run.json.training`与batch/group文件保存公开聚合指标，不保存私有tensor |
| `GRPOBatch`/`TrainingTrajectory` | 自研 worker批输入 | 由TRL替代 | sampler/buffer/reward/advantage由TRL持有 | 是 | Dataset row + environment list映射即可 |
| `objective.py` | 自研 clip/K3/token-mass | 由TRL替代 | 锁定TRL标准语义 | 是 | 仅做离线契约对照测试，不复制函数 |
| `trainer.py`/`worker.py` | BF16 base + PEFT custom Trainer/FSDP | 由TRL替代 | 7B LoRA、30B QLoRA、optimizer/checkpoint交给TRL | 是 | 薄train entry按完整YAML构造model/quantization/PEFT/reward/env；不复用旧worker |
| `resources.py::ManagedVLLMService` | 外部服务停/重启/owner控制 | 不再保留 | colocate由TRL构造与同步；owner协议属于明确非目标 | 是 | 只做一次只读GPU空闲检查和运行finally，不保留claim/lock/服务状态机 |
| `synchronization.py` | adapter publish/HTTP activate | 由TRL替代 | colocate同步机制不同 | 是 | 直接引用Trainer原生checkpoint相对路径；不另存checkpoint manifest，不自行激活 |
| `vllm_context_plugin.py` | vLLM0.12 Qwen3MoE LoRA patch | 不再保留 | 新版本且7B主路径不需要；第一阶段不运行30B | 是/上游 | 不迁移或重建plugin；未来30B先验证官方路径，再单独修订计划 |
| CLI/run report/output IO | 旧`pyproject`已有`swe_agent`console script，但命令落到旧Workflow | 复制后适配 | 稳定本地CLI与证据入口有价值，旧Workflow控制面不保留 | 部分 | `pyproject.toml`固定`swe_agent = swe_agent.cli:main`；`scripts/grpo.sh`调用`.venv/bin/swe_agent grpo --config <完整YAML路径>`；CLI只解析命令并进入薄`train.run()`；一个`outputs/<run-id>`；删除旧多份report与并发/owner协议 |
| 旧 tests | 111项模块合同混合历史架构 | 拆分后部分保留 | 领域回归价值高 | 部分 | 迁移schema/tools/Docker/verifier fixtures；删除XML/custom objective/lifecycle tests |
| 旧 `artifacts/grpo_runs` | 历史事实与失败样例 | 不再保留 | 只作为本报告事实依据，不是新项目运行输入或生产资产 | 否 | 不复制历史run；若具体领域测试需要输入，按对应旧测试的最小fixture重新建模，不承接artifact目录 |

### 10.1 复用边界复核

这些模块不是“复制单文件即可”。会影响实施正确性的依赖如下：[代码事实+建议]

| 资产 | 必须一起承接的最小依赖/行为 | 明确不承接 |
|---|---|---|
| models | Pydantic strict/extra-forbid；`Task/Environment/Evaluation/Sample`的public/private分界；Step连续index | 重复repo/base、泛型metadata、PolicyTrace字段、provider/XML ID、逐Step messages/概率、patch/output refs |
| loader | 两份revision路径中的Parquet、PyArrow、qualification JSON、original/offline eval script、gold/test patch/hash | `WorkflowConfig`整类、旧模型/GPU配置、`qualification_status`历史字符串作为truth |
| DockerSandbox | `DockerConfig`最小字段、fixed task/image断言、image inspect、明确container ID handle、失败后可重试的幂等close、`/testbed` base/clean检查、no-network/no-mount/limits；label仅用于本次output/episode关联和崩溃后人工定位 | pull/build/load/rmi/prune、通用image manager、atomic owner receipt、owner-label认证、自动stale recovery、`ManagedVLLMService`、全局GPU owner/claim状态机、旧项目label命名 |
| tool schema/executor | `ToolSpec.validate`的required/unknown/type/range/enum校验、路径逃逸/`.git`限制、exact-once edit、command denylist、输出截断、镜像内conda语义 | XML digest/AcceptedCall parser、provider call交叉校验、通用shell平台抽象 |
| verifier | private `Evaluation.offline_eval_script`、fresh sandbox factory、patch check/apply、offline eval、pytest-start marker和resolved/unresolved证据 | infra伪装成`Verification`、reward字段/Trainer eligibility批处理、旧Workflow调用壳 |
| recorder | Step.index、Action/Observation、termination与幂等finalize思路 | 时间/ID/wire linkage、完整messages、patch/verifier/cleanup refs、token/logprob、policy digest链、Trainer resume状态 |
| Workflow/AgentLoop | prompt中的SWE任务约束、policy/infra分类、失败也落证据、finally cleanup的**设计意图** | 类本身、rollout/group控制、model client、XML循环、GPU/vLLM生命周期 |
| tests/历史输出 | `test_loader_and_schema.py`、`test_agent_and_docker.py`、`test_verifier.py`中领域合同；旧`artifacts/`内真实任务资格和失败run只作事实依据 | 111项整包、fake full workflow当E2E、历史run/资格日志复制进新项目、长期`test-results/` |

固定 loader 当前不仅读取“样例元数据”，还逐字段比对官方表与subset表，并读取 gold来验证资格资产。新项目完全独立意味着后续必须复制这两份精确 Parquet或由独立、不可变的数据根提供同一内容；**指向旧仓路径不符合独立原则**。为了保持现有fail-closed合同，第一阶段推荐复制精确文件和hash，不提取一个手写JSON替代。gold/selected JSON虽需loader/evaluator读取，但必须留在private evaluator资产边界，不能进入Dataset row、`reset` kwargs、environment public attributes或容器可读路径。

## 11. 目标目录结构

```text
2607_trl_swe_agent/                   # filesystem目录名，不是Python包名
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── grpo_swegym_qwen2_5_coder_7b_lora.yaml
│   └── grpo_swegym_qwen3_coder_30b_a3b_qlora.yaml
├── scripts/
│   └── grpo.sh                       # 只启动一次真实GRPO作业
├── src/swe_agent/                    # 唯一Python包；物理仓库目录名不进入import identity
│   ├── __init__.py                   # 显式普通Python包
│   ├── cli.py                        # swe_agent console script；仅本地grpo命令
│   ├── config.py
│   ├── models.py                      # 九个核心领域对象
│   ├── swegym.py                      # 固定数据与任务资产加载
│   ├── prompts.py                     # 唯一build_prompt(task)纯函数
│   ├── docker.py                      # DockerSandbox
│   ├── tools.py                       # 六工具spec/validator/ToolExecutor
│   ├── verifier.py
│   ├── environment.py                 # SWEEnvironment reset/finalize/close + thin wrappers
│   ├── rewards.py                     # 唯一binary reward adapter
│   ├── recording.py                   # messages/trajectory/batch/group + run.json单写者
│   └── train.py                       # 薄GRPOTrainer构造、preflight与run-scoped finally
├── data/
│   └── swegym/                        # 锁定revision的数据集文件
│       ├── SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb/data/train-00000-of-00001.parquet
│       └── SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd/data/train-00000-of-00001.parquet
├── assets/
│   └── swegym/                        # 随项目版本化的小型固定任务资产
│       └── getmoto__moto-7023/
│           ├── selected_instance.json
│           ├── eval_script.sh
│           ├── eval_script.offline.sh
│           ├── gold.patch
│           ├── test.patch
│           └── manifest.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   └── plan.md
└── outputs/                           # 真实run；每个直接子目录就是Trainer output_dir
    └── <run-id>/
```

prompt builder已确定为`src/swe_agent/prompts.py::build_prompt(task)`纯函数，唯一读取`Task.problem_statement`并加入native-tool/submit说明；Dataset只调用它，environment不得再拼第二版。`src/swe_agent/__init__.py`显式存在，项目不使用namespace package。[用户决策]

新项目不建立笼统artifact根、资格结果根、长期测试结果根、全局日志根或自定义checkpoint根。`data/`和`assets/`都是固定输入，不允许一次运行写入；`outputs/`只含真实训练/闭环run。普通测试和资格命令默认只输出终端或使用pytest `tmp_path`。[用户决策+建议]

`pyproject.toml`固定console script：`swe_agent = swe_agent.cli:main`。`src/swe_agent/cli.py`只提供本地`grpo --config <完整YAML路径>`命令、SIGINT/SIGTERM边界和退出码，不引入网络/API/通用任务管理。`scripts/grpo.sh`只调用`.venv/bin/swe_agent grpo --config "$1"`；`cli.py`再调用`train.run(config_path)`。所有内部导入使用`from swe_agent...`，当前filesystem目录名不进入package identity。[用户决策]

`assets/private_eval/`没有独立必要性，改为自然的`assets/swegym/<instance_id>/`。`gold.patch`只由loader/资产资格测试读取，永不构造进运行时`Evaluation`；`test.patch`和original script也只用于资格一致性，正式verifier只从私有`Evaluation.offline_eval_script`取得已核验脚本。上述内容都不能进入Dataset row、prompt、`reset`公开参数、environment public attributes或tool-readable mount；目录名本身不提供安全边界，也不因此增加权限系统。[建议+用户决策]

### 11.1 单一run根与命名

采用`outputs/<run-id>/`。`run-id`使用UTC（格林尼治零时区）`YYYYMMDDTHHMMSSZ-<4 hex>`，例如`20260719T153012Z-a13f`；未来实现必须以`datetime.now(UTC)`取得时间，`Z`明确表示UTC+00:00。该目录直接作为`GRPOConfig.output_dir`；不配置本地run name，不增加模型、算法、日期或实验系列中间层。[用户决策]

新run若目标目录已存在则fail closed，禁止静默覆盖。第一阶段正式CLI不提供自动resume或失败后自动续跑；checkpoint保持Trainer原生可恢复格式，但实际resume属于后续单独授权范围。不复制、移动或改名checkpoint。第一阶段不建设run registry、目录扫描服务或实验数据库。[建议]

### 11.2 目录相关配置的最小集合

```text
output_root: outputs
run_id: null                 # 第一阶段固定由入口按UTC-Z+4位随机码生成
```

程序从`output_root + run_id`派生唯一`output_dir`和全部rollout相对路径，不提供trajectory/patch/verifier/log/checkpoint/report等独立路径配置。`GRPOConfig.output_dir`由入口最终注入，不能在model/environment配置中另写一份。[建议]

第一阶段验收run显式配置Trainer原生项`logging_steps=1`、`save_strategy="steps"`、`save_steps=1`和`save_total_limit=2`，因为`max_steps=1`时默认save间隔不会产生所需checkpoint。这些是训练语义，不是路径抽象。真实SWE闭环总是保存第18节的最小rollout事实；TRL的`log_completions`显式为`False`，避免框架completion dump与`rollouts/`重复。[官方接口事实+用户决策]

`config.yaml`是项目解析后的完整配置主源。checkpoint内Transformers原生生成的`training_args.bin`、adapter config、tokenizer文件和`trainer_state.json`是Trainer可加载合同，不视为项目重复配置，也不得为了“目录整齐”搬出checkpoint。[官方接口事实]

### 11.3 资格与测试不建立第二套脚本或输出系统

- `scripts/`只有`grpo.sh`，它通过`.venv/bin/swe_agent grpo --config ...`启动真实作业。依赖解析/安装/import由实施Agent直接运行uv与`.venv/bin/python -c`命令；可复现的第三方接口、模型构造和vLLM gate进入`tests/integration/`并用pytest marker显式选择；正式CLI再做便宜、确定、无副作用的preflight。三者职责互补，不新增资格脚本。[用户决策]
- 稳定版本进入`pyproject.toml/uv.lock`，架构边界留在本报告，正式run实际版本/模型/image进入`run.json`。当前无需另建环境说明文档；lock无法表达的稳定主机前置条件直接写入本报告的资格合同，失败即停止，不临场新增文档或脚本体系。
- 普通pytest只依赖退出码、终端输出和`.pytest_cache/`；文件型测试使用`tmp_path`。JUnit、coverage等仅由具体命令或CI写到临时/外部路径。
- Docker集成测试的container、patch和日志在`tmp_path`内产生并在测试finally清理；默认不长期保存。真正执行“固定任务真实rollout→verifier→Trainer step”的验收必须通过`grpo.sh`/正式train入口创建普通`outputs/<run-id>`，不能伪装成pytest证据。

## 12. TRL/vLLM 组合可行性结论与失败合同

### 12.1 分项判断

| 组合/能力 | 判断 | 证据 | 实施门槛 |
|---|---|---|---|
| `GRPOTrainer` 标准 GRPO/optimizer/checkpoint | 可直接实现 | 1.8源码/官方Trainer基类 | 固定核心默认并做一次真实step |
| native multi-turn `tools` | 需要小型适配 | `_tool_call_loop`/tool mask已有 | Qwen template round-trip；工具wrapper |
| `environment_factory` 隔离 SWE repo | 需要小型适配 | 1.8 pool/reset/每rollout instance | reset重建容器、runner兜底close |
| submit立即终止 | 目标组合不成立（按旧精确语义） | 无 done/terminal hook | 采用submit后最终无工具turn的最小替代 |
| env/verifier binary reward | 可直接实现 | custom reward拿到 completions+environments | 单一 finalizer；infra raise |
| Qwen2.5-Coder native tools | 存在版本限制/待本机确认 | 官方tokenizer模板 + TRL已知模板映射 | 锁revision、`supports_tool_calling`与prefix测试 |
| 7B LoRA trainer | 需要本机资格（接口清楚） | `peft_config`与无量化model load源码 | BF16 base、target/trainable set、one-step gate |
| 7B LoRA + colocate vLLM sync | 存在版本限制 | 1.8 PEFT merge/full sync源码及release fix | 单卡TP1参数/输出变化实测；重新测BF16显存 |
| colocate + sleep | 库中有路径，本机当前无法确认 | config/VLLMGeneration生命周期；vLLM0.23 CUDA13风险 | 先过wheel/extension，再做HBM/RAM/wake/cleanup资格 |
| 上述全部 + multi-turn + 7B单卡TP1 vLLM | 当前无法确认 | 无单一官方/本机完整组合测试；GPU资源需实施时预检 | 一次实施授权后资格固定TP1；失败按第12.2节停止并保留证据与配置 |
| 30B-A3B QLoRA 同一SWE主流程 | 软件边界可保持，运行组合当前无法确认 | TRL有BNB+PEFT接口；旧运行不是4-bit路径；本地模型是MoE | 第一阶段只冻结NF4 double-quant、q/k/v/o target和`runtime_qualified=false`；资源拓扑留到后续30B实测 |

总体结论：目标主路径不是“不可实现”，但也不能写“TRL 自动完成”。准确等级是：**接口层需要小型适配，版本和资源层有明确 qualification gate；没有证据要求自造大型补丁。**

该可行性判断按模型分层：7B BF16 LoRA的vLLM gate覆盖Torch/vLLM wheel与CUDA ABI、BF16模型/PEFT、colocate、sleep/wake和受控adapter权重同步机制；全部都是计划内硬gate。bitsandbytes作为唯一依赖集合成员必须通过resolver/wheel安装硬门，但其import、CUDA扩展、4-bit与MoE功能不属于7B通过条件。30B-A3B QLoRA运行资格全部推迟到7B闭环后，第一阶段不锁定或验证其GPU拓扑。[用户决策+待实施验证]

### 12.2 7B固定vLLM路径与失败证据合同

1. 7B固定采用`TRL 1.8.0 + 经资格的vLLM + colocate TP1 + sleep`，完整配置始终`use_vllm=true`、`vllm_gpu_memory_utilization=0.3`；精确Torch/vLLM版本按第1.4节在冻结前最多迭代3组有官方依据的候选，不默认取范围最高版本。
2. 首个成功lock后依赖manifest/lock冻结。此后的wheel安装/import/extension/ABI失败、单卡显存不足、sleep/wake或PEFT同步失败，均停止对应阶段。不得删除vLLM依赖、改冻结版本、改`use_vllm`、改memory utilization、换独立server、TP2/TP4、多卡、Transformers生成或自研循环。
3. 失败时保留冻结后的`pyproject.toml/uv.lock`和两份计划配置；若三次resolver候选评估均失败，则保留最后provisional manifest及三次候选评估摘要，不以“修复”为名继续试探。若尚未创建正式`output_dir`，终端和实施汇报必须给出失败stage、精确命令、候选版本/wheel来源、原始错误及资源状态；若run已经初始化，则同样的primary failure写入`run.json.failure`，原始输出进入`train.log`，cleanup独立写入`run.json.cleanup`。
4. failure记录只证明该组合在本次环境/参数下未通过，不得被描述为TRL/vLLM普遍不可行；实施Agent停止并等待用户根据证据修改计划。除第1.4节明确允许的冻结前最多3组resolver候选外，不得继续循环试版本、参数或拓扑，也不得fork/patch第三方库。
5. 30B不适用本节的TP1资源结论：第一阶段不运行30B；后续资源拓扑在7B闭环后基于4×A100和30B QLoRA实测单独设计，但生成框架仍以vLLM为目标。[用户决策]

## 13. 第一阶段实施计划

> **执行授权冻结：当前只制定计划。只有用户明确确认“计划修改完毕，可以开始实施”后，以下阶段才可执行；该一次指令授权Agent按顺序完成A–J前置资格，并推进到7B单run真实在线group与GRPO optimizer step验收，不再为CUDA、7B模型、vLLM、固定Docker或该次GRPO验收逐项询问。正式CLI只执行一次；失败直接结束并记录本run，不自动创建下一run。**[用户决策]

### 阶段 0：环境manifest、lock与独立`.venv`

- 输入：第1.3节依赖约束、本机driver/platform、空的新项目依赖状态；
- 动作：Agent先完成vLLM wheel元数据检查，按第1.4节在声明边界内最多尝试3组不重复Torch/vLLM候选；每组亲自更新provisional `pyproject.toml`并执行一次`uv lock`。首个成功解立即冻结manifest/lock hash，再只执行一次`uv sync --locked`，最后用新`.venv/bin/python`完成基础import；不得把命令交给用户手工运行；
- 输出：唯一且明确包含vLLM的`pyproject.toml/uv.lock/.venv`；Torch/vLLM版本和wheel来源共同锁定；
- 风险/停止：三组候选均无法解析、冻结后任一声明wheel无法安装，或第1.4节C层列出的基础import任一失败时，按第12.2节记录并停止；bitsandbytes必须通过resolver/wheel安装，但本阶段不要求import；只允许冻结前的三次有界resolver反馈，冻结后保留manifest/lock，不删除vLLM、不改后端或临场重选版本；不改旧项目或旧`.venv`。

### 阶段 1：先建立项目骨架、完整配置和正式入口

- 输入：第9、11节固定目录、两份配置合同与CLI合同；
- 动作：建立`src/swe_agent/__init__.py/config.py/cli.py/train.py`、两份平铺完整YAML、唯一`scripts/grpo.sh`和最小测试目录；`pyproject.toml`注册`swe_agent = swe_agent.cli:main`；
- 输出：两份YAML可独立严格解析；`.venv/bin/swe_agent grpo --config ...`可进入preflight但在领域模块尚未完成时明确拒绝真实训练；`train.py`提供模型/PEFT/GRPOConfig/environment_factory/reward的薄构造边界；
- 验收：7B与30B配置字段完整、无继承；7B LoRA/30B QLoRA互斥正确；CLI、config和preflight单元测试通过。该阶段只建立后续资格所依赖的真实文件，解决“先资格却没有配置/入口”的循环依赖。

### 阶段 2：第三方接口与7B模型/vLLM资格

- 输入：阶段0环境、阶段1完整7B YAML/入口、本机完整7B模型、TRL官方最小environment合同；
- 动作：依次运行tokenizer/native-tool round-trip、最小多轮environment fixture、真实7B BF16+LoRA构造/冻结/forward/backward/save/reload，随后资格固定的vLLM colocate TP1/sleep/sync；
- 输出：可复现pytest/终端证据；两份源配置和依赖声明保持计划值，不根据资格结果改写；不建立qualification目录；
- 风险/停止：native tool template按第4.3节只允许一次官方training-safe适配；7B模型或vLLM任一硬gate失败都按第12.2节停止并记录。fixture必须标注“第三方接口资格，不是项目闭环验收”。

### 阶段 3：独立迁移领域 schema 与固定任务资产

- 输入：旧领域模型文件、loader、两份Parquet、固定任务selected/eval/gold/test资产及对应领域tests；
- 前置：阶段0–2通过，逐文件来源审计完成；
- 输出：`src/swe_agent/models.py`中的九个精简领域对象；显式`src/swe_agent/__init__.py`；`src/swe_agent/prompts.py::build_prompt(task)`作为唯一prompt builder；loader直接返回`sample, evaluation`并构造进程内只读`task_id -> (Sample, Evaluation)`映射；新项目自有`data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`、`data/swegym/SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd`和`assets/swegym/getmoto__moto-7023`；
- 风险：gold/test oracle泄漏；仍通过旧仓绝对路径读取；Task/Environment重复repo/base；为wire审计增加无消费者字段；
- 验收：第6.3节字段集合精确成立；双表唯一行/hash/field mismatch fail-closed；gold只在资产资格过程读取且不进入Evaluation；公开Dataset/prompt/reset kwargs递归检查无Evaluation/gold/test；Dataset只含`task_id+prompt`，shuffle、同task四rollout与pool reset后仍由`task_id`取回同一私有Evaluation；项目在旧仓不可见时仍可静态加载任务资产。

### 阶段 4：独立迁移 Docker、tools 与 verifier

- 输入：旧`docker.py/tool_spec.py/tools.py/verifier.py`及第10.1节最小依赖；
- 前置：schema稳定；Agent先运行fake-client/CPU合同测试，再打印固定image、container数量、时长、磁盘和cleanup preflight并直接继续真实Docker集成；无需再次询问；
- 输出：独立`DockerSandbox`、固定六工具的ToolExecutor/native wrappers、返回`Verification`的fresh binary verifier；
- 风险：无意改变image/no-network/base/timeout/conda语义；把empty patch或infra错误混入普通0；
- 验收：迁移领域CPU tests；只接受`getmoto__moto-7023`及固定image ID；missing/mismatch直接失败且无pull；真实fixed image的base/gold资格与旧证据一致；create成功后在start前保存ID；start/base/recorder初始化失败均尝试按ID清理；cleanup失败保留handle供再次close；每次verifier为fresh container；固定image运行后仍保留。

### 阶段 5：实现最小 SWEEnvironment、reward 与 recorder

- 输入：阶段4执行器和已通过的TRL最小environment合同；
- 前置：不再新增旧Trainer/XML/vLLM lifecycle模块；
- 输出：无副作用constructor、fresh `reset`、6个sync tool methods、submit terminal-pending、single custom binary reward/finalizer、以`Step.index`排序的精简trajectory、动态`run.json`单写者和run-scoped finally；run recorder按TRL generation顺序在同一run内分配batch/group/rollout目录；第一阶段验收run分配`batch-0000/group-0000/0000..0003`，environment reset按已资格的TRL输入顺序领取rollout槽，训练返回后读取公开`trainer.state.global_step`完成batch消费记录；
- 风险：pool复用串repo；误以为callable能拿wire ID；tool内infra被TRL吞成普通error；generation早退遗留容器；
- 验收：factory创建的probe和pool扩容实例均进入runner普通引用列表；连续reset和**同一逻辑训练作业/group内**同时active的instance隔离；旧container清理失败时reset不创建新container；固定四条rollout与四个目录一一对应；多call有序；Observation稳定字符串；completion只读对账；policy/infra分类；即使reward/finalize未调用，run级finally仍遍历close并继续收束Trainer/vLLM/GPU。每个environment/verifier只追加自己的内存事件缓冲，main process在generation/reward/finally边界归并并原子写`run.json`，不增加callback、Trainer subclass、writer线程、queue、共享文件写或多作业并发治理。Trainer和vLLM在一个run中只构造、释放各一次，不随group重建。此阶段可用fake Docker client做单元合同，但不算真实闭环。

### 阶段 6：真实 SWE group、GRPO step 与并列验收

- 输入：固定task、真实7B LoRA policy、真实environment/Docker/verifier和经资格的vLLM colocate后端；
- 前置：第16.2节A–J门禁全部通过；group/batch派生值已由锁定`GRPOConfig`测试；Agent打印精确命令、固定vLLM配置、GPU、固定task/image、预计时长、最大输出增长与cleanup preflight后直接执行；
- 输出：一次CLI调用创建一个新的`outputs/<run-id>`和一套Trainer/vLLM/policy状态；其中含一个真实group的4条rollout、二值rewards、一次由`GRPOTrainer`消费该在线group而执行的GRPO optimizer step、原生`checkpoint-<global_step>/`及`save_model(output_dir)`生成的根级final adapter；
- 风险：时长/上下文、全0/全1、vLLM IS全mask、infra混入0、checkpoint与episode错配；
- 验收：第15节并列验收。`native_policy_path_reached`与`trainer_group_consumed`都在同一次`trainer.train()`结束后按真实证据独立判定，二者都为true时`system_closed_loop=passed`。不得在正式Trainer前额外生成rollout，也不得因四条rollout没有tool call而在reward阶段抛出基础设施异常；`no_tool_call`按策略结果reward 0并继续进入group消费。reward全0、advantage为0、gradient为0或LoRA参数不变均按观察事实记录，不触发新的run。

### 阶段 7：单run收束与第一阶段结束

- `trainer.train()`正常返回、失败或中断后都只进入当前run的统一收束路径：先保留既有primary failure，再关闭所有environment与仍存container，随后释放Trainer、vLLM、已知worker与本run资源句柄，最后完成checkpoint/final adapter引用、cleanup事实、CUDA诊断观察和`run.json`终态写入。
- 当前run的reward分布、advantage、gradient和参数变化只写为观察结果；任何零值或参数不变都不启动另一run。当前第一阶段没有自动重试、自动resume或CLI内部再次执行训练。
- CLI打印本run摘要后结束。若用户以后显式再次执行CLI，入口创建新的run；前一个run不决定、触发或汇总后一个run。
- 完整optimizer resume、长期吞吐和post-step真实SWE rollout不属于第一阶段验收，不自动执行。vLLM同步机制在正式run前以受控adapter权重探针资格；正式run只记录step前后参数identity、是否发生数值变化，以及变化发生时可获得的同步证据。参数未变时明确记为`post-step policy is numerically unchanged`，不虚构“新policy”或以数值相同证明同步。
- 第一阶段到此结束，不运行30B。30B YAML和共享代码必须已完成，但其vLLM TP、训练并行和4×A100资源布局不在本阶段拍板；后续30B实施需基于当时QLoRA显存/吞吐证据单独制定运行资格，不重写SWEEnvironment/tools/verifier。

## 14. 测试计划

测试范围同样服从第0.1节：不增加网络安全、认证授权、漏洞扫描、并发 admission、分布式锁/租约或多个独立训练作业共享资源测试。唯一的并行正确性测试是 TRL 可能同时激活的同一 group environment 之间 repo/container 不串扰；Docker `no-network` 只断言固定任务配置未回归，不做网络安全认证。[用户决策]

普通测试默认只产生终端输出和pytest自己的`.pytest_cache/`；任何文件写入使用`tmp_path`并由测试清理。JUnit/coverage只在具体命令或CI显式要求时写到临时/外部路径。只有`full system`通过`scripts/grpo.sh`与正式train入口运行时才创建标准`outputs/<run-id>`。[建议]

普通`pytest`只运行CPU、纯函数、fake-client和小fixture测试。所有会加载真实模型、使用CUDA/GPU、启动vLLM、访问Docker或执行真实GRPO的测试都必须带显式marker并默认skip，避免开发过程或无参数`pytest`意外占用大型资源。用户发出第0.3节的一次实施授权后，实施Agent才可按第13节显式选择这些marker或正式CLI，并在每次大型动作前打印资源preflight后连续执行；preflight不是新的授权点。[用户决策]

| 层级 | 必测内容 | 关键断言 |
|---|---|---|
| package/models unit | 唯一包路径、严格字段、字段所有权、序列化 | 项目模块一律`from swe_agent...`导入且无第二个项目包；`src/swe_agent/models.py`是唯一领域模型文件且无`core`包；extra字段拒绝；Task/Environment无repo/base重复；九个核心模型无任意metadata、output path和长期null wire字段 |
| Sample/Evaluation unit | public组合、loader返回边界、private oracle | Sample仅含Task+Environment；Dataset/prompt/reset/tool结果无Evaluation；正式Evaluation仅含offline script；gold只在资格代码读取且不进入运行时对象 |
| loader unit | revisions、exact fixed instance、unique row、hash、offline transform | 任一漂移或非`getmoto__moto-7023`请求fail closed |
| tool adapter unit | 六个wrapper schema vs旧ToolSpec语义；参数/跨字段；多行edit | 名称集合精确为六工具；canonical args唯一；无XML/二次JSON解析；`.git`/越界拒绝；使用`Action.arguments`且无metadata/wire ID |
| executor unit | list/read/search/edit/run/submit/error/timeout | Action/Observation一一对应；exact-once edit成功必须有diff；command denylist、timeout cap、输出限长/`truncated`、non-empty submit均与旧行为一致 |
| environment unit | constructor、reset kwargs、multiple calls、submit、finalize、close | constructor无Docker；public/private字段隔离；reset新episode；finalize不重复验证/计分；close只释放资源且幂等；submit后只读 |
| pool/cleanup integration | probe/pool扩容、同实例多reset、同一group两个同时active env、异常/KeyboardInterrupt/SIGTERM best-effort | factory实例全部纳入runner list；repo/container无串扰；reward未调用也cleanup；不声称支持多个独立训练作业并发 |
| Docker integration | fixed image inspect/base/clean/no-network/no-mount/limits/diff | image ID/platform一致；`RepoDigests`状态如实记录；缺失/mismatch不pull；真实patch可导出；不自动删除image |
| verifier integration | empty/apply fail/test fail/pass/setup/timeout/no marker；中途异常/cleanup失败 | policy unresolved与infra不混；fresh独立handle；原异常与cleanup错误分别保留 |
| TRL template unit | Qwen2.5 supports/prefix/render/parse | 原生模板通过；若失败，只允许一次基于官方Transformers/TRL合同的最小training-safe模板适配并重测；仍失败即停止，不引入XML或其他provider协议 |
| TRL multi-turn qualification | reset列、callable上下文、`str(result)`、多个call、tool mask、unknown、max turns、submit final turn | 不依赖wire ID；稳定content；assistant mask1/tool mask0；终止原因准确 |
| reward/verifier unit | policy termination、resolved、unresolved、infra | policy→0；真实`Verification`才映射0/1；infra raise且不构造`Verification`/不进入group；每episode只计一次；`result`字段保持不变 |
| trajectory unit | 单call、多call、连续Step.index、termination、messages去重 | 一个实际执行调用恰好一个Step；多call按执行顺序；Action/Observation无ID/event_seq/wire refs仍可复盘；termination与`Verification`不混合；messages与Trajectory不复制完整相同内容；Trajectory无patch/path/ref/cleanup/run metadata |
| output layout unit | run-id、拒绝覆盖、batch/group层级、原子run记录、checkpoint路径 | 唯一`outputs/<run-id>`；目录名满足UTC-Z+4hex；`batch.json/group.json`关联顺序/rewards/consumed steps且均支持interrupted；无扁平索引；Trainer checkpoint原样在根下；resume才复用目录 |
| run record unit | 单写者、原子replace、生命周期/训练/cleanup正交 | 多active env只提交事件不直接写文件；run/batch/group的running/completed/failed/interrupted映射正确；`native_policy_path_reached`与`trainer_group_consumed`并列记录；正常policy未达标允许`lifecycle=completed/failure=null/system_closed_loop=failed`；cleanup以pending/completed/failed和nullable clean_release区分未完成；primary failure不被cleanup覆盖；container/process/runtime handle硬释放事实准确，进程内CUDA数值仅作诊断 |
| dependency qualification | lock、wheel source、基础import、Torch/vLLM CUDA extension | 按A–D以退出码/终端分层判断；bitsandbytes wheel解析/安装失败会阻断唯一环境，wheel已安装仍不等于BNB可import或4-bit可用；7B gate不执行BNB CUDA/4-bit探针；失败测试证明冻结后failure handler不会改写`pyproject.toml/uv.lock/configs/*.yaml`；大型层默认skip，只由获一次实施授权的Agent显式运行；默认不落长期目录 |
| config unit | 两份完整YAML、无继承、模型路径、context、tools、group/batch/save | 两份均为`num_generations=4`、`max_tool_calling_iterations=20`、`max_prompt_length=8192`、`max_completion_length=22528`、`context_safety_margin=2048`、第一阶段验收run的`max_steps=1`、batch/accumulation均1、每step保存且`save_total_limit=2`；测试用真实tokenizer计最终模板并断言三者之和≤32768；7B另固定单个run seed、`use_vllm=true`、TP1、sleep true、memory utilization 0.3，且不存在run循环或跨run汇总字段；7B `load_in_4bit=false/quantization_config=None`，30B NF4+BF16 compute+double quant且`use_vllm=true/runtime_qualified=false`；配置完整不等于资格通过 |
| 7B LoRA qualification | BF16 base、无quant config、LoRA trainable set、forward/backward、save/reload | r16/alpha32/dropout0；target精确为`q_proj,k_proj,v_proj,o_proj`且全部存在；base frozen；仅预期LoRA参数可训练；非零tiny grad；不要求BNB import、Params4bit或4-bit load；真实模型/GPU测试默认skip，由一次实施授权后的阶段2显式运行 |
| 30B QLoRA static contract | 完整YAML、BNB字段、Qwen3架构声明、PEFT target、`runtime_qualified=false` | 第一阶段只做解析、交叉字段校验和CLI拒绝真实启动；不加载30B、不验证Params4bit/forward/backward/vLLM/TP拓扑，也不以7B证据替代未来30B资格 |
| 7B vLLM qualification | 固定vLLM colocate/TP1/sleep/wake/sync/cache | 以受控adapter权重验证BF16+PEFT同步机制且不要求BNB realization；该资格与正式run是否发生非零parameter update分离；`use_vllm=true`、memory utilization 0.3保持不变；注入每类gate失败时都停止、记录且配置hash不变；默认skip且不得由普通pytest触发 |
| batch/recording contract | run内batch/group/rollout关联、公开global step、run单写者 | recorder支持按generation顺序分配run内索引；第一阶段验收run产生`batch-0000/group-0000/0000..0003`；reset顺序经fixture资格；训练成功后只读`trainer.state.global_step==1`写消费关系；per-environment buffer由main process归并；无callback/subclass，不读私有tensor |
| full system | 同一次`trainer.train()`内并行观察fixed task真实group的native tools/repo/patch/verifier/reward路径与TRL group consumption | `native_policy_path_reached`和`trainer_group_consumed`分别据实记录，二者都为true才满足第15节系统闭环；不得在Trainer前另采样或因无tool call中止reward/消费；一次CLI只创建一个run与一套Trainer/vLLM；全0 reward、零advantage、零gradient和参数不变均允许；只由一次实施授权后的正式入口执行，普通测试不得触发 |
| single-run control | CLI、Trainer/vLLM构造次数、失败终止、终端摘要 | 一次CLI恰好一个run；Trainer/vLLM各构造一次；不存在内部run循环、自动重启、跨run选择或汇总；任一failure/interruption结束CLI并保留当前run；cleanup不覆盖更早primary failure |

### 14.1 Container 生命周期定向合同测试

这些测试只覆盖同一个逻辑训练作业自身创建的资源。除最后一项真实集成检查外，优先使用记录 Docker CLI 调用的 fake client验证状态机，不启动容器；它们是实现合同测试，不是最终 Docker/训练闭环验收。[建议]

| 场景 | 必须断言 |
|---|---|
| create成功、start失败 | create stdout ID在start前已保存；按该ID调用remove；start错误为primary，cleanup错误若有则另记 |
| start成功、image/base/clean校验失败 | 每个失败点都进入close；ID不因校验失败而提前清空 |
| 取得ID后rollout目录或recorder初始化失败 | 初始化错误为primary；仍按ID清理；落盘失败不跳过close |
| 正常finalize | patch/必要证据先冻结；rollout先关闭；fresh verifier独立创建/关闭；只计算一次reward；所有cleanup operations提交给run recorder |
| reward/finalize完全未调用 | 模拟Trainer构造或生成后端提前抛错；runner finally仍关闭probe/pool list中所有active environment |
| 同一environment连续多次reset | 每次得到新episode/ID；第二次create只发生在前一ID确认删除之后 |
| reset前旧container cleanup失败 | reset直接报infrastructure failure；本次不执行任何新create；旧ID仍在handle中供runner重试 |
| cleanup失败后再次close、成功后再次close | 第一次失败保留ID且第二次真正重试；成功后清除ID；其后重复close为no-op |
| rollout与verifier handle隔离 | 即使测试中两者同时存在，verifier构造/close不改变environment的rollout ID，反之亦然 |
| verifier构造、create/start、apply、test、parse任一步异常 | 已取得verifier ID时总会close；cleanup失败不转reward0；rollout evidence不被覆盖 |
| 业务/基础设施异常与cleanup异常同时发生 | 原异常对象/类型是primary；container/process/runtime handle cleanup failure和残留标识结构化进入`run.json`；进程内CUDA计数仅进diagnostics；存在未恢复cleanup failure时`clean_release=false` |
| `KeyboardInterrupt` | 从reset、tool、generation或finalize任一阶段中断，runner finally均遍历其余environment；active batch/group和run均写interrupted；CLI退出语义为130 |
| `SIGTERM` best-effort | 第8.1.6节薄handler必须实现；首次信号触发Python栈展开/finally，active batch/group和run均写interrupted并返回143；重复信号不覆盖原终止原因；不测试SIGKILL自动恢复 |
| runner最终集成检查 | 正常/失败运行后，按`run.json`确认container已删除、已知子PID已退出、vLLM shutdown成功返回且本run的runtime handle已关闭；进程内CUDA allocated/reserved变化只验证被记录为diagnostic且不能单独令cleanup失败；未恢复cleanup failure与训练failure维度互不覆盖；不扫描或治理其他作业 |

不增加SIGKILL、主机崩溃后的自动恢复测试，也不增加多个独立训练作业的锁、租约、冲突或stale cleanup测试。不可恢复终止只测试label和已经原子落入`run.json`的ID/名称足够人工检查，不测试自动删除。[用户决策]

旧 111 项 tests 不能整包复制。优先迁移 loader/schema、ToolExecutor、Docker create/cleanup、verifier分类的测试意图；删除/重写 raw XML、`PolicyTrace`、custom objective/FSDP/service lifecycle tests。TRL自身已覆盖的 loss/logprob/mask不在项目中重测数学实现，只做锁定版本 smoke/contract assertion。

## 15. 第一阶段单run系统闭环验收标准

### 15.1 同一次 `trainer.train()` 的两项并列事实

正式run只调用一次`trainer.train()`。generation、tool loop、reward、group normalization、loss和optimizer step都属于这一次Trainer控制流，因此以下两项只能在同一次调用内收集、在调用返回或异常收束后并列判定，不能实现成“先证明native path，再允许Trainer消费group”的运行时顺序：[用户决策+纠偏结论]

1. `native_policy_path_reached`：固定SWE-Gym task从锁定数据加载且oracle不泄漏；同一task的4条completion来自当前7B LoRA policy；四条rollout使用fresh隔离Docker repo；其中至少一条形成TRL可识别的native tool call、进入实际repo工具执行、产生非空实际diff并submit，随后由fresh verifier container运行真实offline pytest，形成`Verification.resolved`或`Verification.unresolved`及其真实0/1 reward。policy文本中的普通JSON、只创建container、程序退出0、人工completion、gold/mock patch或checkpoint存在均不能替代这些证据。
2. `trainer_group_consumed`：上述同一真实4-rollout group及其reward由同一个`GRPOTrainer`按TRL 1.8标准group normalization/DAPO消费，并执行一次GRPO optimizer step；消费关系由公开`trainer.state.global_step==1`和`batch.json.consumed_by_global_steps=[1]`记录，不能只由checkpoint文件名倒推。

两项没有可执行的先后门禁关系。不得在正式Trainer前额外生成一批rollout来预先证明native path；也不得在reward阶段因`no_tool_call`、未形成patch或策略termination而抛出基础设施异常阻止Trainer消费。此类正常策略结果映射reward 0并留在原group中。Docker、verifier setup/parse、recording或Trainer自身异常仍按真实基础设施类别失败，不能伪装成reward 0。

`run.json.training.system_closed_loop`仅在`native_policy_path_reached=true`且`trainer_group_consumed=true`时写`passed`，否则在可控终态写`failed`。一次CLI只创建一个`outputs/<run-id>`，只构造一次`GRPOTrainer`、colocate vLLM、optimizer/scheduler和该run唯一policy状态；batch/group/rollout证据、Trainer原生checkpoint和`save_model(output_dir)`产物都保存在该run内。

reward可以是`[0,0,0,0]`，group advantage可以全0，gradient norm可以为0，LoRA参数可以不变。必然要求的是一次基于真实在线group的GRPO optimizer step，不是非零parameter update。非零reward方差、非零advantage、非零gradient、parameter update和vLLM IS mask统计只作为非门禁观察字段记录；参数不变时明确记录`post-step policy is numerically unchanged`。正式run前的vLLM同步资格以受控adapter权重验证同步机制；正式run参数不变时不以相同数值虚构“更新后同步”或“新policy”证据。

### 15.2 Run生命周期与资源结果正交记录

run结束、失败或中断时，必须在同一个run级`finally`中收束全部environment、仍存container、Trainer、vLLM、已知worker和本run资源句柄。Trainer/vLLM不在group之间重建或销毁。cleanup状态与两项正式run验收事实分别记录：cleanup失败不能抹去已经完成的真实工具/verifier/GRPO事实，也不能覆盖更早的primary failure；若没有更早failure，未恢复的cleanup failure才以`failure.category="cleanup"`成为本run的primary failure。

`clean_release`的硬证据仅来自本run container终态、明确worker PID终态、vLLM engine shutdown返回和本run资源句柄关闭结果。进程内CUDA allocated/reserved计数及其与run前基线的差异只进入`gpu_diagnostics`，不能单独改变cleanup、lifecycle或failure。主进程退出后的GPU状态不由主进程内`run.json`声称。

第一阶段没有自动重试、自动resume、基于reward结果的新run启动或跨run结果汇总。任何execution failure或interruption都结束当前CLI并记录当前run；正常策略未达到native path也结束当前CLI，但它是验收结果而不是execution failure。重新执行只能由用户显式再次调用CLI，并创建另一个新run。

### 15.3 状态组合矩阵

`lifecycle`描述CLI执行是否正常完成，`failure`只记录异常/中断根因，`system_closed_loop`描述两项验收是否同时成立，`cleanup`描述run持有资源的收束结果。四者不得互相代替：[用户决策+纠偏结论]

| 事实组合 | `lifecycle.state` | `failure` | `system_closed_loop` | `cleanup` |
|---|---|---|---|---|
| `trainer.train()`正常返回；两项并列事实都为true；硬cleanup通过 | `completed` | `null` | `passed` | `completed / clean_release=true` |
| `trainer.train()`正常返回；正常策略结果使native path为false；group仍被消费；硬cleanup通过 | `completed` | `null` | `failed` | `completed / clean_release=true` |
| execution/infrastructure异常 | `failed` | 对应真实category，非null | 两项都为true才可`passed`，否则可控终态为`failed` | 按独立cleanup事实 |
| 只有cleanup出现未恢复失败 | `failed` | `category="cleanup"` | 仍按两项并列事实判定 | `failed / clean_release=false` |
| 捕获`KeyboardInterrupt`或可展开SIGTERM | `interrupted` | `category="interrupted"` | 两项都为true才可`passed`，否则可控终态为`failed` | 按best-effort cleanup事实 |
| 仅进程内CUDA计数未回到run前值，所有硬cleanup证据通过 | 不受该观察影响 | 不受该观察影响 | 不受该观察影响 | `completed / clean_release=true`，差异只写diagnostic |

`pending`只用于run正在进行或SIGKILL/主机崩溃导致最终状态无法写回；正常策略失败不得写入`failure`，也不得归因为environment、verifier或trainer异常。

本验收必然涉及真实7B、vLLM、Docker和GRPO。它不能由普通`pytest`或“实现完成”隐式触发。用户发出第0.3节的一次实施授权后，Agent须在执行前打印精确命令、固定vLLM配置、GPU/显存范围、固定image和container数量、预计时长、4条rollout的输出增长与run级清理范围，然后按第13节执行一次正式CLI。[用户决策]

## 16. 已确定决策、资格门禁与停止条件

本计划不再保留需要实施Agent临场选择的第一阶段架构未决项。正式run之前尚未获得运行证据的内容称为**前置资格门禁**，不是新的授权点；第16.2节A–J通过则按顺序继续，失败则停止、保存证据并报告。正式run中的native policy path与Trainer group consumption不是前后门禁，而是同一次`trainer.train()`后的并列结果。实施Agent不得临场更换依赖范围、模型、任务、LoRA target、量化方案、vLLM参数/拓扑、生成后端、训练框架或另造生成循环，也不得通过改写计划配置掩盖失败。[用户决策+纠偏结论]

### 16.1 已确定的实现合同

| 事项 | 最终合同 |
|---|---|
| prompt | 唯一由`src/swe_agent/prompts.py::build_prompt(task)`构造；Dataset调用，environment不得重复拼接 |
| 数据目录 | `data/swegym/SWE-Gym__SWE-Gym/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`与`data/swegym/SumanthRH__SWE-Gym-Subset/3f22e68f673027edbaebe3424e4c20ae580563fd`；值直接来自旧loader已资格常量，不留路径占位符 |
| private Evaluation配对 | loader返回`sample, evaluation`并建立进程内只读`task_id -> (Sample, Evaluation)`映射；Dataset只有`task_id+prompt`；不依赖row位置或pool slot，不给Evaluation增加ID |
| Python package/CLI | 显式`src/swe_agent/__init__.py`；`pyproject.toml`固定`swe_agent = swe_agent.cli:main`；唯一正式命令`.venv/bin/swe_agent grpo --config <完整YAML路径>`；`cli.py`只进入`train.run()` |
| submit | `submit`成功后environment进入terminal-pending，返回terminal observation；TRL再生成一次无工具final assistant turn结束episode；该额外turn计入20次上限 |
| tokenizer/tool template | 先用目标revision原生模板；失败时只允许一次基于官方Transformers/TRL合同的最小training-safe适配；仍失败即停止，不引入XML/provider协议 |
| batch/group/step记录 | batch/group是run内部索引；第一阶段验收run创建`batch-0000/group-0000/0000..0003`；runner在train前分配，reset顺序由TRL fixture证明，成功后只读公开`trainer.state.global_step==1` |
| batch/step关联 | 初始`consumed_by_global_steps=[]`；一次optimizer step成功且公开global step为1后写`[1]`；失败保持空数组；无callback、无Trainer subclass、无私有tensor |
| run writer | 每个environment/verifier仅追加本地内存事件缓冲；main process在generation、reward和finally边界归并，作为唯一写者原子更新`run.json`；不建writer线程、queue或共享文件写 |
| signal | 必须实现薄SIGTERM handler：设置终止状态并让Python栈进入finally，best-effort返回143；KeyboardInterrupt返回130；SIGKILL/主机/daemon崩溃只保留人工检查边界 |
| 7B单run控制 | 正式group固定`num_generations=4`，第一阶段验收run使用`max_steps=1`；一次CLI只创建一个run、一次Trainer/vLLM和一个seed；reward、advantage、gradient或参数变化不触发新run；failure/interruption直接结束CLI |
| 7B LoRA | BF16 base；r16/alpha32/dropout0；target仅`q_proj,k_proj,v_proj,o_proj`且必须全部存在；失败停止，不扩为all-linear |
| 30B QLoRA | 第一阶段只冻结完整YAML静态合同：NF4、BF16 compute、double quant、r16/alpha32/dropout0、q/k/v/o projection、context32768和`runtime_qualified=false`；不加载模型，不选TP/DP/FSDP拓扑 |
| 7B生成后端/GPU | 固定单GPU TP1 colocate+sleep、`use_vllm=true`、`vllm_gpu_memory_utilization=0.3`；不资格0.4/0.5或其他拓扑 |
| 失败合同 | vLLM任一gate失败即停止并按第12.2节记录；不得切Transformers generate、独立server、自研loop、TP2/TP4、多卡、其他模型/任务/reward或训练框架，依赖与配置保持不变 |
| post-step SWE | 不属于第一阶段要求；vLLM同步机制在正式run前用受控adapter权重探针验证，正式run只记录参数是否发生数值变化，不自动再跑真实SWE rollout |
| verifier文件名 | 类型固定`Verification`，文件名固定`verifier.json`；本计划不再讨论重命名 |

### 16.2 正式run前必须依次通过的A–J资格门禁

| 门禁 | 通过证据 | 失败动作 |
|---|---|---|
| A manifest/resolver | Agent按第1.4节在既定范围内最多评估3组不重复Torch/vLLM候选；首个成功lock冻结`pyproject.toml/uv.lock` hash；全部计划依赖可解析且不意外落sdist/source build | 单个候选失败可进入下一候选；三组均失败则保留最后provisional manifest与候选评估摘要并停止。冻结后不删依赖、不换版本或后端 |
| B install | locked wheel全部安装成功并记录wheel来源，包括bitsandbytes wheel | 任一声明包wheel失败都意味着唯一环境未建立并停止；这是BNB安装硬门，不是BNB运行能力或7B模型能力gate |
| C base import | Python、Torch、Transformers、TRL、PEFT、Accelerate、datasets、jmespath、PyArrow、Pydantic、PyYAML和vLLM导入成功 | 任一必需import失败即记录并停止；不执行bitsandbytes import，BNB运行能力仍不参与7B判定 |
| D CUDA ABI | 一次实施授权下证明Torch wheel/driver/CUDA以及vLLM extension/ABI可用 | 任一失败即停止7B；不改driver、依赖、配置、后端或拓扑 |
| E tokenizer/native tools | 本地7B tokenizer完成schema、render/parse、EOS和native tool round-trip；原生失败时允许的一次官方模板适配也通过 | 仍失败则停止原生工具路径 |
| F minimal environment | fixture证明factory/reset、多个call、字符串tool result、submit terminal turn、completion/environment顺序和异常cleanup | 停止，不写完整SWEEnvironment |
| G 7B LoRA | 一次实施授权下证明BF16构造、冻结base、固定target、trainable set、最小forward/backward及原生save/reload | 停止7B模型路径 |
| H 7B vLLM | 以受控adapter权重证明固定单GPU TP1 colocate、sleep/wake、缓存和PEFT merge/full sync机制 | 任一失败即按第12.2节停止并记录；不得修改配置、引入独立server、自研loop或其他拓扑；该资格不要求正式run产生非零parameter update |
| I recording合同 | fixture证明run内batch/group索引、固定四条reset顺序、`trainer.state.global_step`在成功后为1，且main writer正确归并多environment缓冲 | 失败则修正项目recording适配；不得修改或subclass `GRPOTrainer` |
| J Docker/tools/verifier | 一次实施授权下证明固定image、真实六工具、fresh verifier及全部cleanup合同 | 停止，不启动真实训练 |

30B static boundary是第一阶段的独立静态合同：30B完整YAML必须可严格解析、`runtime_qualified=false`、CLI拒绝真实启动且业务模块无模型名分支。第一阶段不执行BNB/MoE/QLoRA/vLLM/拓扑资格；未来进入30B阶段时先基于4×A100实测制定资格计划，不影响7B结论。

上述状态必须严格区分：resolver成功≠安装成功≠import成功≠CUDA extension可用≠7B模型资格通过≠vLLM资格通过≠Docker/SWE系统闭环通过；7B全部通过不等于30B QLoRA通过。bitsandbytes因属于唯一lock而必须通过resolver/wheel安装门禁，但其import、CUDA扩展和4-bit能力仍只在未来30B gate验证。配置字段完整只表示能独立解析和表达方案，不表示对应运行资格已通过；gate失败也不得反向修改配置以制造“通过”。[用户决策]

### 16.3 正式run结束后的并列结果与run级cleanup

以下三项由同一次正式CLI产生，不是依次触发的门禁：[用户决策+纠偏结论]

| 结果维度 | 判定证据 | 结果影响 |
|---|---|---|
| `native_policy_path_reached` | 至少一条真实7B rollout形成native tool call、实际repo工具执行、非空patch、submit、fresh verifier和真实reward | true/false据实记录；false是正常policy验收未达标，不抛基础设施异常、不阻止同一group继续被Trainer消费 |
| `trainer_group_consumed` | 同一run、同一Trainer/vLLM/policy状态下，真实4-rollout group及其reward被`GRPOTrainer`消费并执行一次GRPO optimizer step | reward全0、advantage/gradient为0或参数不变只记录观察值；只有真实Trainer/基础设施异常才令执行失败 |
| run lifecycle/cleanup | run结束或失败时收束environment、container、Trainer、vLLM、已知worker和runtime handle | cleanup未恢复失败单独记录且不覆盖更早primary failure；进程内CUDA计数只作diagnostic；不在group间重建训练栈，不自动重试或resume |

`system_closed_loop=passed`要求前两项同时为true。它们的证据都来自同一次`trainer.train()`，不得在Trainer前增加预采样，也不得根据第一项的中间结果提前阻断第二项。

### 16.4 授权边界

| 授权 | 允许范围 | 明确不允许 |
|---|---|---|
| 当前计划审查 | 只修改本报告 | 任何项目实施或资格 |
| 用户明确说“计划修改完毕，可以开始实施” | 第0.3节和第13节第一阶段全范围：Agent自行建依赖环境、写源码/配置/测试，显式执行CPU、CUDA、7B模型、固定vLLM colocate TP1、固定Docker，以及一次CLI、一个run、一个真实4-rollout group和一次GRPO optimizer step | CLI内创建第二个run、根据reward/gradient结果自动重启、失败后自动resume、改配置/依赖/后端/拓扑、30B真实运行、第二任务/下载、系统driver/CUDA变更或第三方fork/patch |
| 第一阶段范围外的新指令 | 只在用户未来明确扩大范围后另行制定 | 不得把第一阶段一次授权外推到30B、第二任务或无上限训练 |

第二任务永远不由本计划自动启用。只有用户未来主动提出扩大任务范围，实施Agent先说明下载量、镜像本地占用、临时峰值、`data/assets/outputs`增长和剩余磁盘，再获得手动批准，才可修改计划；在此之前固定任务`getmoto__moto-7023`是唯一允许的Docker/SWE样本。[用户决策]

## 17. 独立架构反思

本节暂时退回核心目标，不预设提示中的每一项都必须按原形式实现。

### 17.1 哪些既定要求可能不合理或应收窄

1. **“TRL 控制 AgentLoop”合理，但若理解为旧 submit 可原样立即终止则与真实API不匹配。** environment没有done hook；保持形式一致会迫使自定义rollout或fork。应接受一个terminal final turn，而不是破坏原生路径。
2. **一开始就要求 4×A100 全量组合不提高7B核心闭环可信度。** 第一阶段7B只资格单GPU TP1 vLLM；多卡会额外引入DDP/TP/NCCL，却不能证明SWE领域链更正确。若固定TP1失败，应记录并停止，不临场探索多卡。30B不在第一阶段运行，未来拓扑不能从7B结论推导。
3. **vLLM colocate/sleep是本项目固定生成后端，但普通测试不能隐式启动大型vLLM。** 用户给出一次实施授权后，Agent应在资源preflight后主动完成7B vLLM资格和真实GRPO，无需逐次询问；若vLLM失败，忠实保留计划配置和失败证据并停止。把失败自动转换为另一后端会使最终系统与已审计划不一致，因此明确禁止。
4. **要求“保留完整Trajectory schema”若指字段逐字兼容，会把旧训练概率惯性带入新项目。** 应保留语义，不保留每Step全量messages/raw response和PolicyTrace；否则适配成本高于审计价值。
5. **第一阶段不能把特定reward分布或非零parameter update作为硬目标。** 7B能力和二值稀疏reward可能使固定任务group全0，但只要真实native tool call、repo交互、patch、fresh verifier、reward和TRL消费链成立，零advantage、零gradient和参数不变都是合法观察结果。用自动新run寻找非零parameter update会把随机结果误作系统资格，并破坏单run policy语义。
6. **为30B预建通用模型插件、FSDP状态机或MoE业务特例都应删除。** 第一阶段只需两份完整、平铺、独立YAML和共享代码边界；旧30B资源/生命周期复杂度不能自动搬到7B。30B的TP/DP/FSDP选择必须留给未来真实QLoRA与4×A100证据，不能提前锁成TP1。
7. **网络安全、认证和通用并发治理不是“以后再补”，而是永久非目标。** 旧项目中的owner receipt、GPU owner状态、provider/policy identity和digest链不能以“可审计”或“并发安全”为由迁移。组内多个rollout只需要显式资源handle和repo隔离；把它升级成接入认证、租约、锁或调度平台会直接偏离核心闭环。[用户决策]

### 17.2 哪些迁移会成为重复实现

- 完整保存 TRL `completion_ids/logprobs/tool_mask/advantages` 到 `PolicyTrace`，再从它构造 `GRPOBatch`：完全重复，删除；
- recorder保存每步完整messages，同时rollout又保存完整`messages.json`：只保留后者；领域Step只用连续`index`排序，不保存wire索引；
- `Action.arguments`、TRL tool call arguments和ToolExecutor内部再次canonicalize为三份可变对象：执行前只产生一个validated canonical dict，wire call只读；
- environment `get_reward()`和custom reward各跑一次verifier：禁止，选择单一custom reward→finalizer；
- patch同时嵌Trajectory、Verification、run记录和独立文件：只保存`final_patch.diff`；同一rollout目录表达归属，领域对象不再保存path/ref/hash；
- 自己重算group mean/std、clip、mask或梯度证据：不得；只读TRL public metrics；
- 复制 `ManagedVLLMService/AdapterPublisher` 后又启用 colocate：两套policy lifecycle冲突，删除旧链；
- 把owner digest、policy identity、tool contract digest全部迁移：只在`run.json`保留复现输入所需revision/文件hash和验证LoRA变化所需比较；不建立digest chain；
- 为Signal Reshaping、context manager或30B动态plugin先建接口：当前无消费者，推迟。

### 17.3 哪些既定选择应原样保留

- **真实Task/Environment/Sample与私有Evaluation**：Task锁定repo/base，Environment锁定image/limits，Sample校验配对，Evaluation只向verifier提供offline script；TRL不提供这套SWE任务数据契约；
- **Action/Observation语义记录**：TRL持有wire messages，但不表达路径校验、exit code、diff、timeout归因；不是重复；
- **Docker/repository隔离**：TRL environment只是调用壳，不提供repo、base commit和固定任务运行边界；这里的no-network/no-mount是复现合同，不是网络安全系统；
- **工具实际实现和领域校验**：native schema不能替代任务路径边界、exact-once edit、命令限制和submit non-empty语义；这些约束防止实验越界，不触发认证/授权框架；
- **fresh verifier、patch、pytest marker**：这是reward事实来源，TRL只调度reward；
- **termination、Verification与run级infra分离**：同样的数值0不能解释为什么停止，且infra不能进入训练；
- **可复盘conversation + 精简trajectory**：二者主从明确后，足以重建行为和环境事实，而不是复刻Trainer；
- **已资格化的固定SWE-Gym样例/image/eval**：重新选容易demo任务会降低与旧证据的连续性和闭环可信度。

### 17.4 如果从零只做核心闭环：更直接的最小架构

我会采用同一组领域模型和四个运行组件，不创建`QualifiedTask`或其他bundle：

1. loader直接返回`sample, evaluation`，Dataset只带`task_id + prompt`，factory闭包持有这两个值；
2. `DockerSandbox`负责fresh container、exec/diff/close；
3. `SWEEnvironment`提供native tool methods、内存中的`Step`列表、submit/finalize；
4. `binary_reward(completions,environments)`运行fresh verifier，一个`train.py`按完整YAML构造7B LoRA（未来同入口构造30B QLoRA）/Trainer并用顶层`finally`收束资源。

只使用一个`outputs/<run-id>`：根级`config.yaml/run.json/train.log`，`rollouts/batch-*/group-*`下每条SWE rollout保存`messages.json/trajectory.json/final_patch.diff/verifier.json`中的实际适用项；全部container、子进程和runtime handle cleanup历史及CUDA diagnostic只进入动态`run.json`；Trainer自己写`checkpoint-<global_step>/`并由原生`save_model(output_dir)`写根级final adapter。没有单独ToolSpec runtime registry（只有固定六工具定义与共享validator）、没有`PolicyVersion/PolicyTrace/GRPOBatch`、没有外部vLLM service manager、没有第二套report/failure/manifest体系，也没有post-step rollout硬门。

| 对照 | 前文推荐迁移架构 | 从零最小架构 |
|---|---|---|
| 相同点 | TRL原生loop、真实task/Docker/verifier、binary reward、精简轨迹、7B LoRA/未来30B QLoRA边界 | 完全相同 |
| 不同点 | 保留精简领域模型、任务资产校验和ToolExecutor分层 | 可把ToolExecutor进一步内联进environment，文件更少 |
| 减少 | 已经删除自研Trainer链 | 进一步减少schema兼容、tool spec与run metadata |
| 失去 | — | 与旧Action/Observation/qualification tests的直接连续性；工具复用边界更弱 |
| 新风险 | 适配工作略多 | env类变成执行/记录大对象，长期更难测试；oracle隔离容易靠约定 |
| 是否值得 | 推荐 | 不建议完全采用；只吸收其“单一train入口、单一output_dir、最少文件、无生命周期框架”原则 |

因此最小架构并没有推翻主方案：成熟的Task/Action/Observation、ToolExecutor、Docker和Verifier确有独立价值。应删的是概率/控制面重复，而不是领域分层。

### 17.5 对主方案的反例检查

| 可能反例 | 当前证据判断 | 是否阻塞核心 | 本计划的确定处理 |
|---|---|---|---|
| native env无法表达submit立即done | **部分成立**：environment没有done hook | 不阻塞 | 固定采用terminal observation + 一次无工具final assistant turn；计入20次上限 |
| Qwen2.5原生template不兼容 | 官方模板/TRL映射支持，实际revision未测 | 阻塞native tool路径 | 只允许一次官方training-safe模板适配；仍失败即停止 |
| PEFT权重无法通过merge/full路径同步到colocate vLLM | 源码有merge/full sync路径，完整组合未测 | 阻塞第一阶段系统闭环 | 正式run前用受控adapter权重资格同步机制；保持`use_vllm=true`和原配置，失败时记录并停止；不要求正式run必须产生非零parameter update，也不引入其他后端、独立server或自研loop |
| sleep或单GPU TP1显存不成立 | 源码有路径，但wheel/显存未测 | 阻塞第一阶段系统闭环 | 只资格计划值`gpu_memory_utilization=0.3`；失败记录并停止，不改参数或拓扑 |
| pool env导致repo串扰 | 若reset不重建则必然成立 | 阻塞可信度 | reset必须先清理旧sandbox再创建fresh sandbox；cleanup失败则停止reset |
| TRL completion无法一一映射Observation wire ID | callable拿不到稳定wire ID，但environment掌握执行顺序 | 不阻塞 | `Step.index`是领域关联；`messages.json`独立保存wire事实，不做不可靠双向ID |
| generation batch没有独立公开ID | 锁定源码需fixture确认reset/completion顺序；第一闭环只有一个预分配batch/group | 不阻塞GRPO数学，只影响记录 | 固定目录由runner预分配，成功后读取公开global step；不为ID增加callback/subclass |
| 固定task的7B group reward全0或全1 | 旧30B真实group已有全0证据，风险现实 | 不阻塞系统闭环 | group固定4；如实记录reward、advantage、gradient和参数变化；不改变任务/group size，不启动新run |
| vLLM IS把全部有效序列mask | sequence-level mask在锁定版本真实存在 | 不阻塞系统路径事实，但会令本次更新无有效数值贡献 | 保持标准TRL配置并记录公开ratio/mask聚合；不关闭correction，不启动新run |
| 30B无需改SWE主流程但资源拓扑未知 | 软件边界成立，QLoRA/MoE/BNB与4×A100资源边界未证明 | 不阻塞7B | 第一阶段只验证静态配置与CLI拒绝运行；未来30B阶段先取实测证据再选择拓扑，不写模型专属业务补丁 |

“optimizer step后必须立即执行post-step真实SWE rollout”已从第一阶段删除。一次GRPO optimizer step不保证发生非零parameter update；vLLM权重同步机制在正式run前采用受控adapter权重探针资格。正式run记录step前后policy identity、参数是否变化及可获得的同步事实；参数不变时只声明`post-step policy is numerically unchanged`。该探针属于一次实施授权内的生成后端资格，不启动额外真实SWE/Docker episode。

### 17.6 独立最终判断

- **总体合理**：TRL控制通用rollout/GRPO，旧项目保留SWE领域资产，是正确分界；
- **原样保留**：固定Task/资产资格、私有offline evaluator输入、gold仅资格可见边界、Docker隔离、六工具领域校验、fresh verifier、二值reward、termination/Verification/run-infra分离、Action/Observation/patch证据；
- **简化**：Trajectory字段、run metadata和submit终止；第一阶段7B不建设多卡/FSDP平台，30B运行拓扑推迟到未来实证阶段；
- **推迟**：完整optimizer resume、长期吞吐、Signal Reshaping和context management；post-step真实SWE rollout从第一阶段删除；
- **删除**：raw XML production协议、旧model client/AgentLoop控制面、PolicyTrace、GRPOBatch、自研objective/trainer、外部vLLM service和adapter激活状态机、旧vLLM plugin；
- **当前不可原样实现**：旧submit立即终止；本计划已经固定采用terminal final turn，不保留另一条TRL路线；
- **当前暂停实施**：依赖环境、目标7B文件和大部分工程底座已经存在，但第21节记录的native policy path和系统闭环尚未通过；历史run只按旧CUDA基线硬门记录cleanup失败，修订后的container/PID/vLLM shutdown/runtime handle硬释放合同尚未由新run验证；只有用户审查修订计划并明确授权后，Agent才从实际未通过的门禁继续；
- **最终推荐**：采用第5节主方案，并吸收第17.4节最小架构的克制原则；
- **对前文修订**：先建立环境与入口，再做接口资格和领域迁移；vLLM colocate/sleep是7B唯一生成后端；一次CLI只产生一个run并构造一次Trainer/vLLM，失败时保留当前run、记录并停止；group是run内部采样/训练单位；非零parameter update仅作观察；post-step SWE rollout不进入第一阶段；custom reward+environment finalizer是唯一reward源；领域执行只用`Step.index`，不假设存在wire ID。

## 18. 单次 `output_dir` 的文件合同

### 18.1 最小布局

```text
outputs/<run-id>/                     # 直接传给GRPOConfig.output_dir
├── config.yaml                       # 完整解析配置；本run配置唯一主源
├── run.json                          # 唯一动态综合结构化记录
├── train.log                         # 单份主进程文本日志
├── rollouts/
│   ├── batch-0000/
│   │   ├── batch.json
│   │   ├── group-0000/
│   │   │   ├── group.json
│   │   │   ├── 0000/
│   │   │   │   ├── messages.json
│   │   │   │   ├── trajectory.json
│   │   │   │   ├── final_patch.diff
│   │   │   │   └── verifier.json
│   │   │   ├── 0001/
│   │   │   ├── 0002/
│   │   │   └── 0003/
│   │   └── group-0001/...
│   └── batch-0001/...
├── checkpoint-<global_step>/         # Transformers Trainer原生checkpoint
├── adapter_config.json               # 最终save_model原生产物（PEFT时预期）
├── adapter_model.safetensors         # 最终save_model原生产物（PEFT时预期）
└── training_args.bin                 # Transformers save_model原生产物
```

不是所有文件从run开始就存在。初始化完成应有`config.yaml`、初始`run.json(lifecycle="running")`和可追加的`train.log`；rollout文件随真实generation写入；checkpoint只在Trainer save策略触发时出现；三个根级final model文件只在训练入口成功调用`trainer.save_model(output_dir)`且当前锁定Transformers/PEFT确实采用这些文件名时出现。官方/本地源码还可能原生写README、tokenizer或其他版本相关文件，本项目不建立白名单schema，也不手工补齐缺失文件。[官方源码事实+建议]

### 18.2 根级文件职责

`config.yaml`只保存完整解析配置。`train.log`只保存主进程人类可读诊断，不承担结构化事实唯一源，也不预拆runner/vLLM/Docker/reward日志。[建议]

`run.json`是唯一动态综合结构化记录，不是不可变manifest。第一阶段schema固定如下；实现可用Pydantic，但不得改名、增加泛型`metadata/public_metrics`垃圾桶或把同一事实拆到其他根级JSON：[用户决策]

```text
schema_version: "1"
identity:
  run_id: str
  output_dir: str                    # 本run的规范化绝对路径，形如.../outputs/<run-id>
  config_file: str                   # 固定为"config.yaml"
provenance:
  started_at: UTC-Z str
  finished_at: UTC-Z str | null
  code_commit: str | null
  code_dirty: bool
  dependency_versions: dict[str, str]
  model_path: str
  resolved_model_path: str
  model_revision: str | null
  generation_backend: "vllm"
  official_dataset_revision: str
  subset_dataset_revision: str
  task_id: str
  image_tag: str
  image_id: str
  image_platform: str
  seed: int
lifecycle:
  state: "running" | "completed" | "failed" | "interrupted"
failure: null | {
  category: "dependency" | "model" | "generation_backend" | "environment" |
            "docker" | "verifier" | "trainer" | "recording" | "cleanup" |
            "interrupted",
  primary_type: str,
  message: str,
  stage: str,
  traceback_log_ref: "train.log"
}
training:
  system_closed_loop: "pending" | "passed" | "failed"
  native_policy_path_reached: bool | null
  trainer_group_consumed: bool | null
  global_step: int
  groups_generated: int
  rollouts_generated: int
  reward_mean: float | null
  reward_std: float | null
  loss: float | null
  grad_norm: float | null
  frac_reward_zero_std: float | null
  checkpoints: list[str]
  final_model_ref: str | null
  observations:
    reward_degenerate: bool | null
    nonzero_advantage_observed: bool | null
    nonzero_gradient_observed: bool | null
    nonzero_parameter_update_observed: bool | null
    all_sequences_masked_by_is: bool | null
cleanup:
  state: "pending" | "completed" | "failed"
  clean_release: bool | null
  residuals: list[str]                # container ID/name、PID或runtime handle标识；不由CUDA计数单独生成
  containers: list[{
    episode_id: str | null,
    task_id: str,
    scope: "rollout" | "verifier",
    container_id: str | null,
    container_name: str,
    operations: list[{
      sequence: int,
      at: UTC-Z str,
      operation: "remove",
      result: "success" | "not_found" | "failed",
      error: str | null
    }],
    final_state: "not_created" | "removed" | "not_found" | "residual",
    residual: bool
  }]
  processes: list[{
    scope: "vllm_worker" | "trainer_worker",
    pid: int,
    operations: list[{
      sequence: int,
      at: UTC-Z str,
      operation: "terminate" | "join" | "verify_exit",
      result: "success" | "not_found" | "failed",
      error: str | null
    }],
    final_state: "exited" | "not_found" | "residual",
    residual: bool
  }]
  runtime_handles: list[{
    scope: "environment" | "trainer" | "vllm_engine" | "model",
    identifier: str,
    operations: list[{
      sequence: int,
      at: UTC-Z str,
      operation: "close" | "shutdown" | "release",
      result: "success" | "not_initialized" | "failed",
      error: str | null
    }],
    final_state: "closed" | "released" | "not_initialized" | "residual",
    residual: bool
  }]
  gpu_diagnostics: list[{
    device: str,
    owner_pid: int,
    allocated_bytes_before: int | null,
    reserved_bytes_before: int | null,
    allocated_bytes_after: int | null,
    reserved_bytes_after: int | null,
    baseline_allocated_bytes: int | null,
    baseline_reserved_bytes: int | null,
    observed_at: UTC-Z str | null,
    diagnostic_only: true,
    note: str | null
  }]
```

`native_policy_path_reached`与`trainer_group_consumed`按第15.1节在同一次`trainer.train()`后并列写入；`system_closed_loop`只在二者均为true时passed。两者任一为false时，正常返回的run可以是`lifecycle="completed"/failure=null/system_closed_loop="failed"`。`observations`中的布尔值为nullable数值事实，不是通过门禁；其中`nonzero_parameter_update_observed=false`明确表示optimizer step后参数数值不变，不能写成发生了parameter update。`pending/null`只用于仍在运行或未能完成观测的状态。[用户决策+纠偏结论]

cleanup初始为`state="pending"/clean_release=null`，只有run级硬释放检查结束后才写终态。execution failure与cleanup failure是正交维度：`failure`保存最早的primary根因；`cleanup.containers/processes/runtime_handles`保存释放操作和最终硬状态。cleanup错误不能覆盖更早primary failure；若没有更早错误，首个未恢复cleanup错误以`category="cleanup"`成为primary。只要这些硬资源存在residual，lifecycle不能声称completed。[用户决策+建议]

`processes[]`只登记本run明确启动的vLLM/Trainer子进程，不登记或扫描其他作业；colocate engine完全在主进程内且没有子PID时数组可以为空。`runtime_handles[]`记录vLLM engine shutdown是否返回、Trainer/model/environment句柄是否关闭或释放。`gpu_diagnostics[]`只记录主进程退出前PyTorch allocator的before/after/baseline数值；不提供`released/residual`状态，不参与`clean_release`或lifecycle判定。主CLI进程退出本身由调用方观察的退出码证明，进程内`run.json`不能证明自身退出后的GPU状态；未来如需该事实，只能由父launcher在训练子进程退出后单独观察。[建议]

`run.json`不复制完整config、messages、Trajectory、patch或verification stdout/stderr。run级reward摘要是跨batch聚合；每个group的reward vector以`group.json`为主。固定输入hash可记录，但不建立digest chain。[建议]

### 18.3 Batch、group 与 rollout 文件职责

`batch.json`表达一个run内部的一次generation batch而非checkpoint。字段集合和类型固定如下；索引属于run内部，第一阶段验收run只有index 0：[用户决策]

```text
schema_version: "1"
batch_index: int                     # >=0
batch_id: str                        # 精确等于batch-{batch_index:04d}
state: "running" | "completed" | "failed" | "interrupted"
task_id: str                         # 第一阶段固定getmoto__moto-7023
generation_backend: "vllm"
global_step_at_generation: int       # >=0
started_at: UTC-Z str
finished_at: UTC-Z str | null
groups: list[str]                    # 有序group-{group_index:04d}相对目录名
consumed_by_global_steps: list[int] # 严格递增的Trainer公开global_step，不从checkpoint倒推
```

`group.json`表达同一prompt/task的有序四条rollout，字段集合和类型固定为：[用户决策]

```text
schema_version: "1"
group_index: int                     # 在所属batch内>=0
group_id: str                        # 精确等于group-{group_index:04d}
state: "running" | "completed" | "failed" | "interrupted"
task_id: str                         # 第一阶段固定getmoto__moto-7023
prompt_sha256: str
rollout_dirs: ["0000", "0001", "0002", "0003"]
episode_ids: list[str]               # completed时恰好4项
rewards: list[int]                   # completed时恰好4项，每项0或1
reward_mean: float | null
reward_std: float | null
degenerate: bool | null
verification_counts:
  resolved: int
  unresolved: int
  not_run: int
```

第一阶段验收run固定`max_steps=1`，所以实际值必须为`batch_index=0`、`batch_id="batch-0000"`、`global_step_at_generation=0`、`groups=["group-0000"]`；成功完成一次optimizer step后`consumed_by_global_steps=[1]`，此前、失败或中断时为`[]`。捕获到`KeyboardInterrupt`或薄`SIGTERM`边界时，当前active batch/group都原子更新为`interrupted`，batch和run写各自`finished_at`，run lifecycle同步为`interrupted`；普通异常写`failed`。`SIGKILL`、主机崩溃或无法回到Python的runtime崩溃可能遗留`running`及仍为null的结束时间，它表示未完成写回的崩溃证据，不能被读取方猜成failed或interrupted。reward或更新数值不触发新的batch或run；同一CLI内不存在run级循环。[用户决策]

两者不保存完整messages、patch、verification正文、advantage、old/current/reference logprob或token mask。失败详情统一进入`run.json.failure`和`train.log`，避免再造batch/group泛型failure对象。

每条rollout目录的唯一事实源：[用户决策+建议]

- `messages.json`：TRL最终structured messages，含assistant tool call与tool-result message；不复制领域Observation字段集合。
- `trajectory.json`：只序列化`Trajectory(task_id, environment_id, steps, termination)`；不含messages、patch、reward、`Verification`、cleanup、文件路径或run信息。infra若发生在任何领域Step形成前可以不存在。
- `final_patch.diff`：最终冻结patch的唯一正文；无合法非空patch时可以不存在，不在其他JSON嵌正文/hash/path。
- `verifier.json`：仅在真实verifier形成resolved/unresolved结论时存在，内容schema是`Verification`及有界必要输出。verifier setup/timeout/无marker/Docker/cleanup失败不制造假result，详情进入`run.json.failure/cleanup`和`train.log`。

run recorder负责run内目录分配与索引文件写入。environment/verifier只维护领域事实和自己的本地内存事件缓冲，不能调用共享writer或直接竞争写`run.json`。第一阶段验收run分配一个`batch-0000/group-0000`；TRL按input顺序reset并将同序environments传给reward的合同先由fixture确认，然后四次episode reset依次绑定`0000..0003`。训练成功后只读取公开`trainer.state.global_step`写消费关系。不得为记录引入callback/Trainer subclass，不得用checkpoint编号倒推rollout，也不得读取TRL私有tensor。[官方源码事实+用户决策+待实施验证]

### 18.4 Checkpoint与最终adapter

`GRPOConfig`继承Transformers `TrainingArguments`。在无超参搜索时，`Trainer._save_checkpoint()`固定写`output_dir/checkpoint-<global_step>/`，并在默认`save_only_model=False`时保存adapter/model、optimizer、scheduler、RNG和`trainer_state.json`；`resume_from_checkpoint=True`也从同一`output_dir`寻找最后checkpoint。[官方源码事实]

第一阶段验收run显式`max_steps=1, save_strategy="steps", save_steps=1`，预期产生`checkpoint-1/`；checkpoint是模型/optimizer状态，不是rollout容器，一个batch/group与checkpoint不存在一一对应关系。因此明确拒绝自定义`checkpoints/0000 ↔ rollouts/0000`包装层，最小关系只写`batch.json.consumed_by_global_steps`和`run.json.training.checkpoints[]`。第一阶段保存完整Trainer checkpoint事实，但不自动resume。[官方源码事实+用户决策]

训练完成后，入口调用Trainer原生`save_model(output_dir)`。本地TRL main的官方`trl/scripts/grpo.py`确实在`trainer.train()`后这样调用；Transformers `save_model`写`training_args.bin`，PEFT `save_pretrained`通常写`adapter_config.json`与`adapter_model.safetensors`。这些根级文件是最终可加载adapter，不是resume checkpoint；具体额外文件集合必须以最终锁定版本实测为准。[官方源码事实]

项目不得从最后checkpoint手工复制adapter，不得移动/改名checkpoint内部文件，也不得给框架文件另建schema。Transformers负责rank写入、barrier、`args.should_save`与rotation；`run.json`只记录相对引用。未来人工挑选/导出模型是后续显式操作，本阶段不建exports层。[用户决策+建议]

### 18.5 简单写入生命周期

```text
解析配置并生成run-id
→ 以“必须不存在”方式创建output_dir
→ 原子写config.yaml和初始run.json(lifecycle=running；system_closed_loop=pending；cleanup.state=pending/clean_release=null)，打开train.log
→ 每次generation由run recorder分配batch/group/rollout目录
→ 原子写messages/trajectory/final_patch/verifier与batch/group索引
→ environment/verifier把cleanup operation追加到各自本地内存缓冲
→ Trainer原生写checkpoint-<global_step>/
→ train结束调用原生save_model(output_dir)
→ finally先冻结live container才可取得的patch/诊断，再遍历container cleanup，关闭vLLM engine、释放Trainer/model/environment句柄并join已知子进程
→ main process在generation/reward/finally边界归并缓冲，合并primary failure、两项并列验收事实、全部cleanup硬证据、进程内CUDA诊断数值和residual
→ 最后原子更新run.json的finished_at与lifecycle
```

`run.json`只有一个主进程写者：environment、verifier和reward不持有writer，不启动writer线程或queue，只在各自本地内存中追加事件；main process在generation、reward和`finally`边界确定性地drain/merge。signal handler本身只设置终止标志/抛出可展开异常，不在handler内做文件I/O，主控制流进入`finally`后再归并和写入。writer每次构造完整新文档，写同目录临时文件、flush/fsync后`os.replace`，必要时fsync父目录。`batch.json`/`group.json`及rollout JSON采用相同的单文件原子替换；`train.log`允许追加。Trainer checkpoint/final model完全交给框架，不套项目原子协议。[用户决策+建议]

普通异常令lifecycle=`failed`，`KeyboardInterrupt`/可展开SIGTERM令run及active batch/group为`interrupted`；两者都先在memory保留primary failure，再执行run级finally cleanup，最后一次原子写入同时保存原始根因、training事实和cleanup维度。正常策略未达到native path但`trainer.train()`正常返回时，lifecycle仍为`completed`、failure为null、system_closed_loop为failed。cleanup期间每个environment/verifier的`close()`、runtime handle释放或子进程退出操作返回/抛错后，main process立即drain全部operations并刷新，避免后续中断丢掉已知container ID/PID/handle；environment内部不直接写文件。一个资源cleanup失败不能阻止其余资源的best-effort收束，也不能覆盖更早primary failure；只有最终未恢复的cleanup失败才改变cleanup/lifecycle，CUDA diagnostic偏离基线本身不改变终态。[建议]

若SIGKILL/主机崩溃发生在最终写前，`run.json`可能仍是`running`或只含部分cleanup operations；这解释为异常中止/最终状态未知，人工根据mtime、train.log、已记录ID和labels检查，不自动恢复。原子替换保证读者看到旧完整版本或新完整版本，不保证最后一个事件必然落盘。[建议]

默认新run绝不复用已有目录。第一阶段不因发现`lifecycle=running/failed/interrupted`而自动恢复，也不把旧checkpoint复制到新run继续；任何重新执行都必须由用户显式启动新的CLI调用和新run。Trainer checkpoint仍保留原生完整性，但resume不属于本阶段自动控制流。[建议]

### 18.6 专项复审的十项结论

| 问题 | 结论 |
|---|---|
| 1. 是否取消新项目`artifacts/` | **是。** 旧项目路径仅保留在历史事实描述；新项目没有该目录 |
| 2. 是否单一`outputs/`根 | **是。** 只有真实run进入；固定输入、测试、探针不进入 |
| 3. run目录层级 | **`outputs/<run-id>`。** 无run name或模型/算法中间层 |
| 4. run目录是否直接作TRL `output_dir` | **是。** 这是继承Transformers Trainer语义的最小方案 |
| 5. 根级项目自有结构化文件 | **只保留`config.yaml + run.json`。** 前者完整配置，后者动态identity/lifecycle/failure/training/cleanup；另有`train.log`与框架原生产物 |
| 6. rollout索引 | **使用batch/group两层JSON。** 与长期GRPO粒度一致；不保存TRL私有tensor |
| 7. 是否长期保存普通环境资格日志 | **否。** 默认终端+退出码；特殊情况由用户显式重定向或指定普通output目录 |
| 8. 是否长期保存普通pytest输出 | **否。** 默认终端、`.pytest_cache`和`tmp_path`；CI/JUnit/coverage按命令外置 |
| 9. checkpoint是否保持`checkpoint-<global_step>` | **是。** 不嵌套、不复制、不按rollout编号；根级final adapter只由原生save_model产生 |
| 10. 还有哪些多余目录/文件 | 删除私有评测目录命名、run/episodes/trainer层、扁平rollout索引、独立状态/摘要/清理文件及TRL completion dump；只保留固定输入已有的小型任务`manifest.json`用于资产内容/hash校验 |

蓝图必须补充的两点是：`train.log`只能由主进程作为文本诊断写入，不能假设捕获所有rank/vLLM子进程；`run.json`和batch/group索引也只能由main process单写。第一阶段单进程满足该约束；未来多rank必须先资格gather/主进程事件汇聚，不能让每个rank共享写同一JSON。[官方源码事实+建议]

## 19. 关键源码与证据索引

### 19.1 旧项目定位

| 主题 | 文件/符号 |
|---|---|
| CLI/总流程 | `pyproject.toml`；`scripts/grpo_once.sh`；`src/swe_agent/cli.py::main`；`workflow.py::SingleInstanceWorkflow.run/_run_once` |
| rollout/reward | `workflow.py::_run_rollout` (`:697`)；`_score_rollout` (`:821`)；`_verify_rollout` (`:864`) |
| schema | 旧项目领域模型文件中的`Task/Environment/Evaluation/Action/Observation/Step/Trajectory/Sample` |
| loader | `swegym/loader.py::load_qualified_instance` (`:46`)、`transform_eval_script_offline` (`:137`) |
| loop/protocol | `runtime/agent.py::AgentLoop` (`:86`)；`tool_protocol.py::AcceptedCall`；`tool_spec.py::TOOL_SPECS/validate_tool_arguments` |
| tools | `runtime/tools.py::ToolExecutor` (`:71`) |
| Docker | `runtime/docker.py::DockerSandbox` (`:97`)、`build_create_command` (`:382`) |
| verifier | `verifier.py::SWEGymVerifier.verify`及其旧返回类型（`:33/:84/:96`）；新项目统一命名`Verification` |
| container退出/信号证据 | `workflow.py::SingleInstanceWorkflow._run_with_final_cleanup`；`cli.py::_SignalBoundary/main`；`tests/test_agent_and_docker.py` 的start/base/cleanup异常测试；`tests/test_synchronization_and_workflow.py` 的primary+cleanup测试；`tests/test_cli_and_independence.py` 的SIGTERM subprocess测试 |
| recorder/probability | `runtime/recorder.py::TrajectoryRecorder`；`training/policy_trace.py::PolicySegment/PolicyTrace/build_policy_trace` |
| old GRPO | `training/batch.py::GRPOBatch/build_batch`；`objective.py::compute_group_advantages/compute_grpo_loss`；`trainer.py::build_lora_policy/train_once` |
| lifecycle/sync | `runtime/resources.py::ManagedVLLMService`；`training/synchronization.py` |
| CPU evidence | `artifacts/test_evidence/20260717_cpu_pytest.xml` |
| real failures | `artifacts/grpo_runs/grpo-20260717T105121Z-c51237f1/`；`...20260716T143239Z-531b87be/`；`...20260718T024136Z-e714f845/`；`...20260718T122433Z-26370ae7/` |

### 19.2 官方版本资料

- [TRL `_BaseConfig`（继承`TrainingArguments`）](https://github.com/huggingface/trl/blob/main/trl/trainer/base_config.py)
- [TRL `GRPOTrainer`（checkpoint与可选completion dump）](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
- [Transformers `TrainingArguments.output_dir`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- [Transformers `Trainer._save_checkpoint`](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py)
- [veRL PPO Trainer配置（project/experiment、checkpoint、可选rollout/validation dump）](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml)
- [veRL Ray Trainer checkpoint与rollout dump实现](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py)
- [SkyRL Trainer配置（project/run、ckpt/export/log路径）](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/config/config.py)
- [SkyRL Trainer checkpoint与dump实现](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/trainer.py)
- [TRL v1.8.0 release](https://github.com/huggingface/trl/releases/tag/v1.8.0)
- [TRL v1.8.0 dependency metadata](https://github.com/huggingface/trl/blob/v1.8.0/pyproject.toml)
- [TRL v1.8.0 `GRPOTrainer`](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_trainer.py)
- [TRL v1.8.0 `GRPOConfig`](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_config.py)
- [TRL v1.8.0 chat template utilities](https://github.com/huggingface/trl/blob/v1.8.0/trl/chat_template_utils.py)
- [TRL v1.8.0 vLLM generation backend](https://github.com/huggingface/trl/blob/v1.8.0/trl/generation/vllm_generation.py)
- [TRL v1.8.0 tests](https://github.com/huggingface/trl/tree/v1.8.0/tests)
- [TRL GRPO Trainer 官方文档](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration)
- [Transformers tool-use/chat extras](https://huggingface.co/docs/transformers/chat_extras)
- [Qwen2.5-Coder-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Qwen2.5-Coder-7B tokenizer config](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/tokenizer_config.json)
- [Qwen3-Coder-30B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [vLLM 0.23 common requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/common.txt)
- [vLLM 0.23 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt)
- [vLLM 0.23 build/Python metadata](https://github.com/vllm-project/vllm/blob/v0.23.0/pyproject.toml)
- [uv project environment layout](https://docs.astral.sh/uv/concepts/projects/layout/)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uv PyTorch integration and `torch-backend` scope](https://docs.astral.sh/uv/guides/integration/pytorch/)

## 20. 最终迁移决策摘要

环境阶段已经实际冻结并安装`TRL 1.8.0 + vLLM 0.22.1+cu129 + Torch 2.11.0+cu129`等组合，项目骨架、领域实现、固定任务资产、Docker/tools/verifier和真实run产物也已存在；这些事实以第21节及当前lock/run记录为准，不再把阶段0描述成尚未执行。修订计划获批后只从实际未通过的门禁继续，同时仍以既有冻结lock、固定Qwen2.5模型和vLLM colocate TP1/sleep/sync合同为基线。普通pytest默认不触发GPU/Docker；大型资格和一次正式run由Agent在后续授权后经资源preflight显式执行。任一硬gate失败都保留依赖、配置和当前run证据并停止。`scripts/`只保留启动真实作业的`grpo.sh`。

Docker/SWE-Gym 实施始终受第0.2节约束：只运行 `getmoto__moto-7023` 和当前已存在的固定 image ID，不自动 pull/build/load/rmi/prune。任何第二任务在用户收到下载量、镜像本地占用、临时峰值、`data/assets/outputs`增长与磁盘余量说明并明确批准前，不得下载或运行。固定 image 长期保留；每条 rollout/verifier 的临时 container 才在 `finally` 中清理。

领域实现仍采用pooled `SWEEnvironment.reset()`为每次rollout创建fresh repo；custom binary reward只通过`finalize()`执行正常证据冻结、rollout close和fresh verifier，runner顶层`finally`则独立于reward遍历factory创建的全部environment并兜底`close()`。这让TRL独占messages、token/logprob/mask/advantage/loss/optimizer/checkpoint和vLLM同步，又不把container cleanup建立在reward一定被调用的假设上。一个实际执行tool call形成一个`Step`，只用连续`Step.index`排序和配对Action/Observation；wire messages独立落盘，不向核心领域模型注入tool call ID或message index。工具严格复用旧六工具语义，但新调用链只使用`Action(tool_name, arguments)`和精简`Observation`，没有XML或重复参数解析。

首个正式目标是7B BF16 LoRA + native multi-turn + real Docker/verifier + 一个基于真实在线group的GRPO optimizer step。生成后端固定为单GPU TP1 vLLM colocate/sleep/sync，`use_vllm=true`、memory utilization 0.3；任一前置gate失败时按已冻结合同保留配置、记录并停止，不切Transformers generate、独立server、自研loop、其他参数或多卡vLLM。7B资格不要求BNB import、BNB CUDA extension、4-bit load、Params4bit或vLLM BNB realization。实际tokenized prompt上限8192、completion上限22528并保留不可借用的2048-token context margin，资格与正式入口使用同一计数函数。正式group固定4条rollout；一次CLI只创建一个run、一次Trainer/vLLM和一条run内policy状态，第一阶段验收run完成一个group和一次GRPO optimizer step。`native_policy_path_reached`与`trainer_group_consumed`在同一次`trainer.train()`后并列判定，二者均为true才是系统闭环；正常policy未达标允许`lifecycle=completed/failure=null/system_closed_loop=failed`。reward全0、advantage/gradient为0或参数不变只作为观察事实，不启动新run。CLI结束只汇总当前run。后续30B-A3B只在第一阶段提供完整QLoRA/vLLM配置入口与共享代码边界；BNB/MoE/PEFT/forward-backward、vLLM realization/sync与4×A100拓扑全部尚未运行证明，也不被预先锁为TP1。

新项目唯一Python包是`src/swe_agent/`，内部导入统一`from swe_agent...`，核心领域模型只定义于`src/swe_agent/models.py`；当前filesystem目录名不进入import路径，也不是命名债务。两份配置`configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml`与`configs/grpo_swegym_qwen3_coder_30b_a3b_qlora.yaml`平铺在`configs/`，都是完整、独立、无继承的配置入口：每份可独立解析并表达完整方案，但实际可执行性必须由各自资格决定，尤其30B不得描述为已经通过。锁定数据集放`data/`，版本化小型任务输入放`assets/`，所有真实run只进入`outputs/<run-id>`并直接作为TRL/Transformers `output_dir`；run ID始终由`datetime.now(UTC)`生成UTC-Z时间戳并追加4位十六进制随机码。

Trainer原生写`checkpoint-<global_step>/`，训练入口原生`save_model(output_dir)`写根级final adapter；项目不搬运、不改名、不按rollout编号包装checkpoint。第一阶段验收run分配`batch-0000/group-0000/0000..0003`，训练成功后只读公开`trainer.state.global_step==1`写`consumed_by_global_steps=[1]`；不增加callback或Trainer subclass。batch/group捕获中断时显式写`interrupted`，只有无法返回Python的崩溃才可能遗留`running`。每个environment/verifier只缓冲本地事件，main process在确定边界归并并作为唯一写者原子更新`run.json`；两项正式run验收事实、非门禁数值观察和cleanup正交记录。cleanup硬判据只使用container、已知子进程、vLLM shutdown和runtime handle终态；进程内CUDA allocated/reserved只记录为diagnostic，不能单独制造residual或failure，且cleanup不能覆盖更早primary failure。旧项目artifact路径只作为历史证据，不迁入新项目。

结论是：**报告正处于实施后纠偏审查阶段，当前没有继续实现授权。只有用户明确确认修订计划可以继续实施后，实施Agent才从实际未通过的A–J前置门禁推进，并在一次授权内完成剩余代码/测试修正、必要资格，以及一次CLI、一个run、一个真实4-rollout group和一次GRPO optimizer step。大型动作必须先打印资源与命令preflight，但不再逐项等待批准。任一前置硬gate或run内execution故障都应保留当前配置与run证据、忠实记录并停止，不临场换依赖、参数、后端或拓扑，也不自动创建另一run；正常policy未达到native path则以completed run和failed验收事实结束，不伪装成execution failure。30B真实运行、第二任务、重新执行正式CLI和系统环境变更仍在授权外。** 当前路线保留已经验证且TRL不提供的SWE事实与执行边界，删除旧多run抽样层级，并以TRL作为唯一通用rollout/训练控制面。

## 21. 实施过程中产生的 blocked 与目标偏移纠偏事实记录

### 21.1 记录信息与性质

- 记录时间：`2026-07-20 16:26:51 CST (+0800)`；
- 记录阶段：原计划已经进入实施并产生真实7B、vLLM、Docker、GRPO输出之后；
- 记录性质：实施过程中产生的`blocked`与目标偏移纠偏记录；
- 事实来源：当前`docs/plan.md`、当前项目源码、`outputs/`中的六个真实run、对应checkpoint/trajectory/group/run记录，以及同日只读宿主机进程、固定任务容器和GPU 2计算进程审计；
- 当前goal状态：`paused`。本节记录形成时未继续训练、未清理输出、未修改运行产物、未查询或使用GPU 1。

本节只追加截至记录时间已经能够由代码或落盘产物确认的事实、偏移事实及其形成基础，不把尚未实施的纠偏方向写成既定方案，也不覆盖前文作为原始计划的历史文本。

本节出现的`attempt`、多run预算、跨run汇总及相关字段仅用于保存旧计划和既有实现的历史错误证据，不对第0—20节修订后的正式流程、配置、schema、阶段、门禁或验收产生规范效力。

### 21.2 原计划自身限定的运行形态

原计划第13、15、16、20节把第一阶段正式运行固定为以下形态：

- 训练数据只有固定任务`getmoto__moto-7023`；
- 每个group固定4条rollout；
- 每个run固定`max_steps=1`；
- 有效学习资格最多8个相互独立的run，每个run重新建立单独`output_dir`、Trainer、checkpoint和final adapter；
- run之间不复用已完成run的adapter或optimizer状态；
- 完整optimizer resume、长期吞吐、post-update真实SWE rollout不属于第一阶段；
- 第一阶段结束后只保留未来30B静态边界，没有定义一个7B连续训练阶段。

因此，原计划第一阶段在训练语义上是一次真实在线GRPO update及非零更新路径的资格验证，而不是一条持续存在的policy训练链。原计划将“有效学习通过”用于命名一次非退化group、非零gradient和LoRA参数变化的资格结果，但同时明确不要求单步后能力提升、连续更新或训练前后真实评估。

当前实现忠实固化了上述边界，而不是只在YAML中把它们作为可修改默认值：

- `src/swe_agent/config.py`把`max_steps`声明为`Literal[1]`，把`effective_learning_max_runs`声明为`Literal[8]`；
- `src/swe_agent/train.py::run()`的函数合同是“最多八个相互独立的单步run，首个有效学习后停止”；
- 每次`_run_attempt()`都以配置中的base model路径重新构造`GRPOTrainer`，没有传入`resume_from_checkpoint`或上一个run的adapter；
- 每次attempt只调用一次`trainer.train()`，要求`trainer.state.global_step == 1`，随后保存并释放Trainer/vLLM；
- `_recording_reward()`只允许reward被调用一次，并要求恰好4条对齐rollout；
- recorder只预分配并消费`batch-0000/group-0000`；
- `build_training_dataset()`只构造含一个固定task/prompt的单行Dataset。

由此产生的多个run在policy关系上是多条从base独立出发的单步路径，而不是`π₀ → π₁ → π₂ → …`的累积训练路径。每个run保存的optimizer checkpoint是真实框架产物，但当前正式入口不消费这些checkpoint继续下一步训练。

### 21.3 六个真实run的落盘事实

截至本节记录时间，`outputs/`包含6个run目录，总占用约`1.1G`：

1. `20260720T061446Z-3e59`；
2. `20260720T062645Z-3b52`；
3. `20260720T063219Z-e1b9`；
4. `20260720T064252Z-baee`；
5. `20260720T064759Z-2239`；
6. `20260720T065825Z-dec5`。

六个run均确认了以下相同事实：

- `global_step=1`；
- 生成1个group、4条rollout；
- reward为`[0, 0, 0, 0]`，`reward_mean=0`、`reward_std=0`；
- `frac_reward_zero_std=1`、`loss=0`、`grad_norm=0`；
- `lora_parameters_changed=failed`；
- 产生`checkpoint-1`和根级adapter；
- `training.system_closed_loop=failed`；
- `training.effective_learning=not_applicable`；
- run最终`lifecycle.state=failed`，落盘primary failure均在cleanup阶段。

六个run的24条trajectory全部是：

- `steps=[]`；
- `termination="no_tool_call"`；
- 没有实际工具执行；
- 没有提交patch；
- 没有构造`Verification`；
- group中的`verification_counts`均为`resolved=0, unresolved=0, not_run=4`；
- Trainer指标中的`tools/call_frequency=0.0`。

代表性模型输出不是TRL识别的native tool call，而是assistant文本content中的Markdown代码围栏和JSON文本。因此真实run虽然为每条rollout创建并随后删除了fresh rollout container，但没有沿模型策略路径执行六工具、冻结patch或启动fresh verifier。

六个输出目录是彼此独立的单步执行，但不能准确描述为六个不同seed的attempt：前5个run记录的seed均为`20260714`，第6个run的seed为`20260719`。落盘事实只证明使用了两个不同seed值。

### 21.4 门禁状态与实施偏移事实

原计划第15.1节把真实native多轮tools、repo实际diff、fresh verifier、binary verifier reward、一次optimizer step、checkpoint和clean release共同列为系统闭环条件。原计划第13节将阶段7命名为“系统闭环后的限定工作”，第16.2节也规定先通过K system loop，随后才进入L effective learning。

当前落盘状态中，K system loop从未通过：六个run均为`system_closed_loop=failed`。在K未通过期间仍继续产生了后续单步run；这些run不能按原计划门禁顺序归类为已经进入L effective learning，也不能合并表述为多步训练。

当前`src/swe_agent/train.py`中的`system_passed`实际判定只组合以下条件：

- `training_succeeded`；
- recorder中的group为completed；
- cleanup completed且`clean_release=true`；
- 没有failure。

该代码判定没有另外要求至少一次真实tool call、至少一次submit、至少一份实际patch或至少一次fresh verifier执行。因而当前run虽然因为cleanup失败而没有被错误标记为system passed，但若只改变cleanup结果，现有判定路径本身无法排除“4条rollout全部no_tool_call、verifier全部not_run”的group被标为系统闭环通过。这个实现判定与第15.1节文字验收条件之间存在事实上的覆盖缺口。

目标偏移的形成基础是：原始goal要求继续实施`docs/plan.md`，而原计划把第一阶段终点明确收缩为单步系统/非零更新资格，并把连续7B训练、optimizer resume和训练前后评估排除在交付外；实现又用严格schema、单次reward recorder和单group目录合同把该资格形态固定为唯一正式入口。在此基础上，有限attempt预算被用于重复寻找一次非退化group，资格验收逐步成为实际实施的终点，但这些独立单步run没有形成一条累积policy训练轨迹。

同时，真实run首先暴露的是policy没有形成native tool call，而不是仅仅固定任务reward稀疏。24条trajectory没有进入工具/verifier路径，因此这些全0结果不能只解释为“已完成真实SWE尝试但任务未解决”。cleanup failure又成为`run.json`记录的primary failure，使native tool path未进入这一事实没有成为run级primary failure。

### 21.5 GPU、进程与容器资源记录事实

六个run的四个rollout container均记录了成功remove，当前固定任务label `swe_agent.task_id=getmoto__moto-7023`下没有残留container。六个`run.json`的`cleanup.processes`均为空数组。

六个run的cleanup失败来自当前Python owner进程内的CUDA allocation/reservation没有在`_finalize_gpu()`检查时回到run开始前的零基线，而不是来自落盘记录中的容器删除失败。前两次记录的检查时显存分别约为：

- `36,460,141,568 allocated / 36,601,593,856 reserved bytes`；
- `40,254,586,368 allocated / 40,359,690,240 reserved bytes`。

后四次检查时的allocated约为`15,382,024,704`、`15,382,024,704`、`15,382,024,704`、`15,382,023,680 bytes`，reserved约为`25.06 GiB`。每次run因此记录`cleanup.state=failed`、`clean_release=false`和GPU device `2` residual。

当前GPU清理判定的代码事实是：

- `_gpu_baseline()`把当前主Python进程的PID写成`owner_pid`；
- `_release_trainer()`在同一进程内尝试sleep/shutdown vLLM engine、清理optimizer并把Trainer model移到CPU；
- `_finalize_gpu()`在该主进程尚未退出时读取`torch.cuda.memory_allocated()`和`torch.cuda.memory_reserved()`；
- 只有二者都回到本run的进程内基线才标记released；
- `_run_attempt()`随后无条件调用`recorder.set_processes([])`，当前实现没有把实际发现、跟踪和退出确认的vLLM/worker子PID写入该数组。

因此，现有`run.json`中的GPU residual直接证明的是“主进程退出前的PyTorch CUDA记账未回到零基线”。它本身不等价于“run结束后宿主机仍存在项目进程”，空的`cleanup.processes`也不构成对子进程不存在的独立发现证据。

同日稍后的只读宿主机审计未发现包含本项目路径、`swe_agent`、`test_7b_vllm`或项目vLLM命令的残留进程，也未发现固定任务container。审计时GPU 2仍有两个属于其他用户`YYL@ZJU`且路径与本项目无关的计算进程：PID `573307`占用约`30994 MiB`，PID `2149942`占用约`5246 MiB`。这两个外部进程不是上述run的cleanup对象。该宿主机审计说明审计时点没有项目残留进程，但不反向改写各run在主进程退出前记录的CUDA residual事实。

当前资源生命周期按原计划的单步attempt划分：每个group都重新构造Trainer/vLLM，并在该唯一step后立即执行整套Trainer/vLLM/GPU teardown和零基线验收。因此，GPU teardown问题与“独立单步attempt是正式运行单位”的计划形态直接耦合；它记录的是每个资格run终止时的资源收束结果，而不是一条连续训练作业内部多个generation/update step之间的资源状态。

### 21.6 截至记录时间的纠偏状态事实

截至`2026-07-20 16:26:51 CST`，本项目不能表述为已经完成实际GRPO后训练，也不能表述为已经通过原计划的真实SWE系统闭环。准确状态是：

- 真实TRL、vLLM、LoRA、Docker、SWE领域对象、工具适配、verifier、recording和checkpoint基础设施已经形成；
- 真实Trainer已六次到达单个`global_step=1`并保存框架checkpoint/adapter；
- 六次step均为零reward方差、零gradient、LoRA未变化；
- 24条rollout均未形成native tool call，真实工具和verifier策略路径未进入；
- 系统闭环K为failed，有效学习L为not applicable；
- 多个run彼此不继承，未形成连续训练；
- GPU cleanup hard gate在主进程退出前的CUDA零基线检查上失败，但审计时没有项目进程或固定任务container残留；
- 当前goal保持paused，本节记录的是实施中blocked与目标偏移纠偏事实，而不是对原计划后续实施路径的确认。

2026-07-20第二次计划审查后，第0—20节又完成以下纠偏；这些是计划文本状态，不表示实现已经修改或重新运行：

- 进程内`torch.cuda.memory_allocated/reserved`及其基线差异降为`gpu_diagnostics`，不再单独构成GPU residual、cleanup failure或lifecycle failure；正式硬释放证据改为container、明确worker PID、vLLM shutdown返回和本run runtime handle终态；
- 原K/L顺序门禁从正式设计删除，替换为同一次`trainer.train()`后的`native_policy_path_reached`与`trainer_group_consumed`两项并列事实；
- 正常policy未形成native tool path时，正式状态合同明确允许`lifecycle=completed`、`failure=null`、`system_closed_loop=failed`；仅cleanup发生未恢复失败时使用独立`failure.category="cleanup"`；
- 正式术语以GRPO optimizer step表示必然执行事实，只有参数确实发生数值变化时才记录non-zero parameter update；参数不变时记录post-step policy numerically unchanged。
