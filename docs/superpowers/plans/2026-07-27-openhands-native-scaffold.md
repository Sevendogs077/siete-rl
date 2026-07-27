# OpenHands 7B 单一原生 Scaffold 迁移实施计划

> 制定日期：2026-07-27
>
> 当前授权：只允许审计代码并编写本计划；本文落地前不得修改 `src/`、`tests/`、`configs/`、`scripts/`、依赖或运行产物，不得启动训练。
>
> 实施方式：后续获得明确授权后，逐任务执行；每个任务先写/更新测试，再改实现，再运行该任务列出的验证。

## 当前阶段、唯一目标与职责

当前处于**计划固化阶段**，不是实现阶段。此阶段只修改本计划，目标是把已经核实的源码调用关系、目标 checkpoint 协议、固定依赖版本和验收门禁写成无歧义的实施合同。

唯一目标是：

> 用本地 `NovaSky-AI/SWE-Gym-OpenHands-7B-Agent` 权重替换原 Qwen 权重，并把模型可见协议及其必要的 TRL 耦合点一次性替换为该 checkpoint 所训练的 OpenHands 协议。

目标 checkpoint 本身仍是 `Qwen2ForCausalLM`，因此不增加 model adapter、wrapper 或第二种模型架构。现有 LoRA、GRPO、vLLM server、Docker、verifier、reward、recording 和任务资产继续使用；旧 Qwen 工具协议被删除，不需要继续兼容。

本阶段职责：

- Agent：完成只读证据核对、修改本计划、检查计划内部一致性并汇报；不修改实现。
- 用户：当前无需克隆仓库、无需整理 dirty worktree、无需选择 GPU，也无需运行命令。
- 后续开始实现前：用户只需明确授权按本计划修改源码；GPU 资格执行前再确认可用设备即可。
- 若实现时需要再次核对 `openhands-aci`，Agent 可把锁定 tag 临时克隆到 `/tmp` 做只读审计；该 checkout 不进入项目，也不成为运行时或测试依赖。

## 0. 结论

采用**单一 OpenHands mock-function-calling scaffold 替换当前 Qwen 原生工具协议**，不做双协议，不保留旧协议 fallback，也不增加 scaffold 配置开关。

这里的 scaffold 不是只指 parser，而是模型能看到或能改变对话状态的完整边界：

1. system prompt 与三工具说明；
2. 首轮 SWE 问题 prompt；
3. `<function=...><parameter=...>` 生成格式；
4. reasoning 前缀与单函数调用解析；
5. `EXECUTION RESULT of [...]` observation；
6. 普通 assistant message 后的固定 fake-user 回复；
7. `finish` 的立即终止语义；
8. `execute_bash`、`str_replace_editor`、`finish` 的工具行为；
9. TRL 多轮历史的 token 拼接与 tool/user suffix mask；
10. 模型、tokenizer、sampling 参数和资格检查。

只改 parser 会失败，因为 TRL 1.8.0 当前还会：

- 从 environment 的公开方法推导 tools；
- 把 tools 传给 Qwen chat template；
- 把 tool observation 渲染成 `role=tool`；
- 上游 TRL 对不含 tool call 的普通 assistant 文本直接退出 loop；本项目当前 `SWEGRPOTrainer` 又把这类输出重分类为 format error。

如果仅替换 parser，模型会同时看到 OpenHands XML 说明和 Qwen `<tool_call>` 说明，历史 observation 角色也与训练轨迹不一致。因此必须整体替换上述模型可见边界。

任何改动若不能直接归因于以下三类之一，即为越界：

1. 加载目标 checkpoint/tokenizer；
2. 实现模型可见的 OpenHands 协议；
3. 接通该协议所必需的本地 TRL 1.8.0 plumbing（粘合逻辑）。

保留现有、与协议无关的训练和 SWE 基础设施：

- TRL 1.8.0 `GRPOTrainer`、GRPO loss、LoRA、vLLM 权重同步；
- SWE-Gym 数据选择、任务资产和 private evaluator；
- 专用 Docker daemon、容器创建/校验/清理；
- patch 提取、独立 verifier、binary reward；
- recording、reporting、launcher、supervisor；
- `Task/Environment/Evaluation/Sample/Action/Observation/Step/Trajectory/Verification` 领域模型。

## 1. 不可违反的边界

### 1.1 `.external` 零运行时依赖

`.external` 只作为本次只读调研的上游证据位置。实施完成后：

- `src/`、`tests/`、`configs/`、`scripts/` 不得 import、继承、调用或运行 `.external` 中的任何文件；
- 不得通过 `sys.path`、动态 import、subprocess、环境变量或相对路径间接调用 `.external`；
- 不得新增 `openhands`、`openhands_aci`、LiteLLM、BrowserGym 运行依赖；
- 不得要求部署机器存在 `.external`；
- 需要的 converter、schema、prompt、editor 算法必须复制到本项目的本地模块并按现有 Docker 边界适配；
- 本地测试 fixture 也必须自包含，不能在测试时读取 `.external`。

允许在本计划和第三方来源说明中记录上游路径、commit 与 hash；这只是 provenance（来源记录），不是运行依赖。

### 1.2 单协议

实施完成后只有 OpenHands scaffold：

- 删除当前 Qwen `<tool_call>`、bare JSON、fenced JSON fallback；
- 不调用 `trl.chat_template_utils.add_response_schema`；
- 不保留 `native_tool_calling`、`add_response_schema` 配置字段；
- 不新增 `protocol: qwen|openhands` 或 `scaffold: ...` 选择器；
- 不保留六工具别名；
- 不把 OpenHands XML 转成旧六工具再执行。

parser 把 OpenHands 参数从 JSON string 形态转换为 TRL 调用所需的 Python `dict`，只属于类型桥接，不是第二套协议。

### 1.3 非协议源码保持不变

以下文件不需要协议适配，计划冻结为不修改：

- `src/swe_agent/asset_generation.py`
- `src/swe_agent/cli.py`
- `src/swe_agent/docker.py`
- `src/swe_agent/launcher.py`
- `src/swe_agent/models.py`
- `src/swe_agent/recording.py`
- `src/swe_agent/reporting.py`
- `src/swe_agent/rewards.py`
- `src/swe_agent/supervisor.py`
- `src/swe_agent/swegym.py`
- `src/swe_agent/verifier.py`
- `src/swe_agent/worker.py`

`pyproject.toml` 与锁文件也保持不变：本方案只使用 Python 标准库和项目已有模块，不增加依赖。

如果实施时发现必须修改这份冻结清单中的文件，应停止实施、补充证据并重新审查计划，不能顺手扩展范围。

冻结的是这些模块的源码，同时还要冻结跨模块接口。实施不得改变：

- `swegym.py` 继续 import 并调用 `build_prompt(sample.task)`，所以保留 `build_prompt(task: Task)`；
- `rewards.py` 继续 import `SWEEnvironment` 并调用 `environment._finalize(completion)`；
- `SWEEnvironment` 类名和构造参数；
- `reset(task_id: str, **kwargs)`，其中 `**kwargs` 继续容纳 TRL 数据行中的 `prompt` 等字段；
- `trajectory`、`verification`、`frozen_patch`、`terminated`、`episode_id`；
- `_finalize()`、`_close()`、`_drain_events()` 以及 `train.py` 当前依赖的清理状态。

唯一需要改变的返回合同是：`reset()` 必须在完成容器和 alias 初始化后返回 `None`。TRL 1.8.0 会把非 `None` reset 返回值追加到最后一条 user message；保留当前 “Fresh repository...” 文本会污染 trajectory-exact 的 OpenHands 首轮 prompt。这个变化不改函数签名，但必须由接口测试锁定。

### 1.4 共享服务器

后续 GPU 资格只允许使用 `nvidia-smi` 选择显存足够的设备：

- 不结束、暂停、迁移或调整其他人的进程；
- 不把“GPU 上有人使用”当成可清理对象；
- 没有足够空闲显存时跳过 GPU gate，并如实报告；
- 加载测试只加载本地模型，不下载模型或依赖；
- 当前计划阶段不加载模型。

## 2. 已核实事实

### 2.1 目标 checkpoint

本地目标目录：

```text
/home/2025user/zyp/.cache/modelscope/hub/models/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent
```

已核实 `config.json`：

- architecture：`Qwen2ForCausalLM`
- dtype：`bfloat16`
- context：32768
- hidden size：3584
- layers：28
- attention heads：28
- KV heads：4

已核实 `generation_config.json`：

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `repetition_penalty=1.1`
- EOS IDs：151645、151643
- PAD ID：151643

### 2.2 训练轨迹是协议真值

本地轨迹：

```text
data/swegym/OpenHands-SFT-Trajectories/data/train.success.oss-00000-of-00001.parquet
```

SHA-256：

```text
ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1
```

已完整扫描的事实：

- 491 条 trajectory；
- 19,751 条 message：491 system、9,630 user、9,630 assistant；
- 491 条 system prompt 完全相同；
- system prompt 长度 4,758 字符；
- system prompt SHA-256：
  `1120aa8819abb372428afb82f6a5f49d1d243e4bf58cb27fd481809acd339e84`；
- assistant function call 共 8,779 条：
  - `str_replace_editor` 5,884；
  - `execute_bash` 2,441；
  - `finish` 454；
- 4,976 条 assistant message 只有 function call；
- 3,803 条 function call 前有普通 reasoning；
- 851 条 assistant message 是普通文本；
- 其中 840 条普通文本后紧跟固定 fake-user，11 条位于 trajectory 末尾；
- 454 次 `finish` 全部是最后一个 function call，后面没有 observation；
- 数据中没有空 `execute_bash.command`、`ctrl+c` 或只有 `cd` 的 bash 调用；
- editor 的有效 command 是 `view/create/str_replace/insert/undo_edit`；
- 两条无效 editor command 进入了上游运行时错误，而不是另一套协议；
- 模型工具路径主要是 `/workspace/...`，也存在 `/testbed/...`；
- 37 条 trajectory 没有 `finish`，不能据此把普通文本当作成功终止；
- 47 条历史序列超过 checkpoint 的 32768 context，因此历史数据存在截断事实；本项目继续使用现有 context/iteration 退出归因，不声称能重放所有超长历史。

固定 fake-user 原文：

```text
Please continue working on the task on whatever approach you think is suitable.
If you think you have solved the task, please first send your answer to user through message and then finish the interaction.
IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
```

该常量在实际消息中以换行符结尾；fixture 和测试必须锁定末尾 `\n`，不能只用 `strip()` 后比较。

### 2.3 上游来源锁定

OpenHands SWE-Gym 只读 checkout：

```text
commit e644a2ca45c3623b27a7e6c169e3d479f0a87fbc
```

本计划使用的来源文件与 SHA-256：

| 上游文件 | SHA-256 | 本地用途 |
|---|---|---|
| `openhands/llm/fn_call_converter.py` | `6b3ec45ea4422e5c8067c107463c4ba331d1c86f0f68d363b819160a748d49db` | 本地复制 converter 的必要常量和函数 |
| `openhands/agenthub/codeact_agent/function_calling.py` | `95dbfa65dedc9289a322a4ea4069d2e5c58261c93e41ab569542be824fafa57f` | 本地复制三工具 plain-dict schema/description |
| `evaluation/utils/shared.py` | `907e0b5e3ec54b46d429ccf75c74e837d70ef6ee2826aee08354b11cf0ab183c` | 本地复制 fixed fake-user |
| `evaluation/swe_bench/run_infer.py` | `4bbed1c002e09e271bd9c1a23d676f8bbb7752409173bc0cb9ee8f90ab420a20` | 本地复制 SWE issue prompt 结构 |

文件编辑器来源是 `openhands-aci` tag `0.1.0`：

```text
commit 0698260b8e03ff2974ba81fd97ad8585a2255297
```

必要来源文件：

| 上游文件 | SHA-256 | 用途 |
|---|---|---|
| `openhands_aci/editor/editor.py` | `3290c4a1ad6339797b3d8feeed9e95e47f25b66167c9aa6abe9e449fd4dd3d79` | editor 状态机和文本算法 |
| `openhands_aci/editor/results.py` | `6bc6062f9de0763d228da437448473339bb9bad6effa4a33dbefa938bf9def12` | 本地结果类型 |
| `openhands_aci/editor/exceptions.py` | `1eb7a76eaae1336399527d1441a17b2cb2af86fee8a5175162e9d8a75c7150cb` | 本地异常类型 |
| `openhands_aci/editor/config.py` | `8f2420d150037dccb9e36fd316664d0b9d076b07c2f49540b4616b4c9f975d6a` | clipping/window 等常量 |
| `openhands_aci/editor/prompts.py` | `bfa07364bdb206c8fded80e645f0017fc01e49a236af06ab10c56d1f87ab0223` | editor 输出文本 |
| `openhands_aci/editor/__init__.py` | `ac1aa43c377566b7ed96859e5dac6944e1a5431b77dd9c542256c7358432a2a4` | 核对 `ERROR:\n` 包装行为 |
| `openhands_aci/editor/shell.py` | `8e4fb88a30ce35292ee3a84be2163e75ee2b17b736e56b3104fef64c7f5a3dc5` | 只用于理解接口，不复制 host subprocess |

`openhands-aci` 不是项目 `.external` 中的本地依赖；上述事实来自锁定 tag 的只读远程/临时 checkout 核对。实现只复制必要算法到本项目；不保留完整 vendor checkout，不建立依赖系统。`shell.py` 只作为语义参考，宿主 subprocess 实现由本地 `ContainerFileBackend` 替代。该 editor 子集只需要标准库。上游 tag 未包含独立 `LICENSE` 文件，但其 `pyproject.toml` 声明 MIT；实施时只记录事实和来源，不伪造不存在的 license 文件。

### 2.4 来源冲突的裁决规则

不能用一句“trajectory 永远优先”覆盖所有层。按以下矩阵裁决：

| 决策面 | 真值来源 |
|---|---|
| 模型可见 prompt、工具名/说明、observation wrapper、fixed fake-user | checkpoint 的实际 SFT trajectory |
| trajectory 未覆盖的 parser 边界 | 锁定 OpenHands converter |
| trajectory 中已经出现的 editor 成功/失败输出 | trajectory fixture |
| trajectory 未覆盖的 editor 算法与状态 | `openhands-aci==0.1.0` |
| reset、token 拼接、parser 激活与 mask | 固定的 TRL/Transformers 和本项目调用代码 |
| 路径、容器执行和 verifier 边界 | 本项目 Docker/Environment 合同 |

只有容器化所必需的适配或已证实的上游缺陷可以偏离来源；每一处偏离都必须在 provenance 文档列出，不能泛化为额外兼容层。

已经确认的版本差异：

- trajectory system prompt 是 4,758 字符的旧 prompt；当前 checkout 的 `system_prompt.j2` 已变化；
- trajectory 中 editor path 示例写 `/repo`，当前 checkout 的 schema 写 `/workspace`；
- trajectory traceback 指向 OpenHands 内置 `file_editor`，当前 checkout 已改为 `openhands_aci`；
- trajectory 的首轮 issue prompt 同时出现 uploaded `/workspace/<name>` 和“修改 `/repo`”。

因此本地常量必须以 trajectory 的精确文本为主，checkout 用于理解算法和状态机，不能盲目复制当前 HEAD 后声称与 checkpoint 对齐。

### 2.5 固定 TRL/Transformers 契约

项目锁定 `transformers==5.13.0`、`trl==1.8.0`，本迁移只实现这组接口，不增加旧版本兼容分支。

目标 tokenizer 已核实：

- `supports_tool_calling=True`；
- `is_chat_template_prefix_preserving=True`；
- `has_generation_markers=False`，但本项目 GRPO 路径只要求 prefix preserving。

TRL 1.8.0 的 `_generate` 只有在 tokenizer 的 `response_template` 或 `response_schema` 非空时才调用 `parse_response`。因此本地 processing class 必须显式激活 parser，不能仅挂载一个永远不会被调用的方法。固定实现合同是：

1. 安装只接受 TRL 实际调用形态 `parse_response(ids, prefix=...)` 的本地方法；
2. 设置一个本地、非空的 response-template 激活标记；
3. 用 wrapper 忽略 TRL 传入的 `tools=`，保留其余 chat-template 参数；
4. 不调用 `add_response_schema`，不注入 Qwen tool schema；
5. trainer 构造测试必须证明初始化成功、`trainer.chat_template is None`、parser 确实被调用且渲染结果没有 `<tool_call>`。

另保留一个与 tokenizer 无关的 `parse_openhands_text(text: str)`，只服务协议单元测试和内部复用；不为 string/batch/旧 Transformers 形态扩展 tokenizer `parse_response`。

## 3. 最终架构

```text
Task.problem_statement
  -> prompts.build_prompt()
       system = trajectory-exact base + locally rendered 3-tool suffix
       user   = OpenHands SWE issue template + /workspace/<safe-task-id>
  -> patched local tokenizer
       ignores TRL-supplied native tools during chat rendering
       uses ordinary Qwen system/user/assistant turns only
  -> vLLM generation
       raw OpenHands <function=...> text or plain assistant message
  -> local tool_protocol parser
       tool action | plain message | protocol error
  -> SWEGRPOTrainer custom loop
       tool action    -> environment public method
                      -> user "EXECUTION RESULT of [...]"
       plain message  -> fixed fake-user
       finish         -> immediate terminal, no observation
       protocol error -> user ErrorObservation text, bounded retry
  -> SWEEnvironment
       execute_bash | str_replace_editor | finish
  -> local ToolExecutor/OpenHandsEditor
       all filesystem and commands execute inside rollout container
  -> existing patch extraction/verifier/binary reward
```

### 3.1 为什么不保留 TRL 原生 tool rendering

TRL 仍可通过 environment 反射得到三个 callable，用它完成 Python 调用；但 tokenizer 的本地 wrapper 必须忽略 `tools=` 参数，防止 Qwen chat template 注入 `<tool_call>` 说明。

工具说明已经由本地 OpenHands system prompt 完整提供。历史中的 assistant function call 保留为普通 `assistant.content` 原文，observation/fake-user 保留为普通 `user.content`。任何一轮都不能把 OpenHands function call 重新渲染成 Qwen tool-call JSON。

### 3.2 唯一状态机

| 模型输出 | 解析结果 | 环境动作 | 下一轮模型可见内容 | 终止 |
|---|---|---|---|---|
| reasoning + 一个合法 `<function=execute_bash>` | tool action | 执行 bash | `EXECUTION RESULT of [execute_bash]:\n...` | 否 |
| reasoning + 一个合法 `<function=str_replace_editor>` | tool action | 执行 editor | `EXECUTION RESULT of [str_replace_editor]:\n...` | 否 |
| 合法 `<function=finish>` | finish action | 冻结当前 diff | 无 observation | 是 |
| 不含 function call 的普通文本 | plain message | 不调用 environment、不写 Step | 固定 fake-user | 否 |
| function 名、参数、类型或结构无效 | protocol error | 不调用工具 | user-role 错误文本 | 达到连续错误上限时终止 |
| completion/context 超预算 | loop exit | 不增加模型可见 suffix | 无 | 是 |
| 达到 tool-loop iteration 上限 | loop exit | 无 | 无 | 是 |

function call 前的 reasoning 属于生成 token，保留并参与训练。parser 以锁定 converter 的已核实行为为边界，不额外发明 duplicate-function 或 trailing-suffix 的严格规则；若实施者认为必须限制一轮只执行一个函数，应先用 converter/trajectory/vLLM 实际输出证明具体规则并更新本计划，不能把未验证 hardening 声称为“OpenHands 原生行为”。

### 3.3 token 与 mask

每轮生成的 assistant token mask 为 1。

由运行时补入的以下 token mask 为 0：

- `EXECUTION RESULT of [...]` user message；
- fixed fake-user；
- protocol error user message；
- 为补全 OpenHands 回合边界而由 chat template 添加的 user/generation suffix。

正式 generation 不配置字符串 stop sequence，只使用 checkpoint 的 EOS 配置，让已训练的完整 `</function>` 留在生成 token 中。parser 要求完整 closing tag；截断的 function call 是 protocol error。首版不实现 `_fix_stopword`，不合成 closing tag，也不为合成 token 设计额外 mask。只有真实 vLLM 资格提供“服务端系统性吞掉完整 closing tag”的证据时，才暂停实施并单独修订这一合同。

自定义 loop 不再调用 TRL 的 `_get_tool_suffix_ids()`，因为它专用于 `role=tool`。新增 OpenHands suffix helper，使用当前真实 conversation 做 prefix/full 两次渲染：

1. prefix 必须等于已有 `prompt_ids + completion_ids` 的 EOS 对齐前缀；
2. full 是追加 user observation/fake-user 后、带下一轮 generation prompt 的 token；
3. 只把 full 多出的 suffix 加入下一轮 input；
4. prefix 不一致立即报错，不静默重新 tokenize assistant 生成内容；
5. context overlong 时只回滚本轮新增的 prompt/suffix bookkeeping，不回滚已经发生的容器副作用，保持现有 TRL 行为。

每个 sample、每轮和 overlong rollback 后都必须满足：

- `len(tool_mask[i]) == len(completion_ids[i])`；
- raw model 生成的 assistant reasoning、function-call 和 plain-message token 全部 mask 为 1；
- observation、fixed fake-user、protocol-error 和 generation suffix token 全部 mask 为 0；
- `sum(tool_mask[i])` 等于该 sample 各轮 raw model token 数之和；
- completion length 只统计 mask=1 的 token；
- overlong rollback 后 ids、mask、messages 同步回到同一边界；
- `finish` 保留本轮 assistant token，不追加 observation suffix。

### 3.4 路径

不修改 `Task`、`Environment` 或 SWE-Gym loader。

protocol 层从 `task_id` 生成只含 `[A-Za-z0-9_.-]` 的稳定目录名：

```text
/workspace/<safe-task-id>
```

`SWEEnvironment.reset()` 在 rollout 容器打开且 base contract 通过后，调用固定的容器内 Python helper：

```text
ensure directory /workspace exists
create symlink /workspace/<safe-task-id> -> /testbed
create symlink /repo -> /testbed
```

helper 只接受 argv，不拼 shell。目标已是指向 `/testbed` 的 symlink 时幂等通过；目标不存在时创建；目标若是其他 symlink、普通文件或真实目录则报 infrastructure error，不删除或覆盖容器内已有路径。

三个路径指向同一容器 worktree：

- `/testbed`
- `/repo`
- `/workspace/<safe-task-id>`

prompt 中的 uploaded path 与实际 alias 完全一致。editor 接受容器内任意绝对路径；它不能访问宿主机，因为所有 stat/read/write/list 都通过 `DockerSandbox.exec()` 完成，rollout container 没有 host mount。

### 3.5 bash observation

`execute_bash`：

- 每次用 `/bin/bash -lc` 在 `/workspace/<safe-task-id>` 启动；
- 沿用现有 `DockerSandbox.exec()` timeout；
- 不再用旧六工具协议的 pip/apt/command denylist；
- Docker 容器仍是 `--network none`、`--cap-drop ALL`、无 host mount；
- stdout/stderr 合并后追加：

```text
[Command finished with exit code <code>]
```

- observation 超过 30,000 字符时，复制 OpenHands `truncate_content` 的“保留头尾、截断中间”算法；
- 本实现不会返回 OpenHands 的交互式 `exit_code=-1`，因此空 command/`ctrl+c` 的交互分支不会被虚假模拟；已扫描的目标 trajectory 也没有使用这些调用；
- 后台命令按 prompt 建议自行重定向文件，后续普通 bash 调用读取文件。

### 3.6 editor

新增本地 `src/swe_agent/openhands_editor.py`，复制并适配 `openhands-aci 0.1.0` 的：

- 五个 command；
- 参数错误文本；
- unique `old_str` 校验；
- tab expansion；
- insert line 语义；
- 每文件 LIFO undo history；
- `cat -n` 风格六位行号输出；
- 目录最多两层且排除 hidden item；
- snippet context window 4；
- 16,000 字符 editor 截断与固定 `<response clipped>` notice。

不复制 `Path` 直接访问宿主机的实现。改为本地 `ContainerFileBackend`：

- `stat(path)`
- `read_text(path)`
- `write_text(path, text)`
- `list_two_levels(path)`

backend 使用固定的容器内 Python helper；路径作为 argv，写入内容通过 `DockerSandbox.exec(input_text=...)` 传 stdin，不拼接 shell 字符串。返回结构由本地代码严格解析，容器错误转换成本地 `ToolError`。

history 存在 episode 内的 `OpenHandsEditor` 实例中；每次 environment reset 创建新 executor/editor，跨 episode 不共享。

明确适配：

- 上游 `create` 缺少/空 `file_text` 的 bare `raise` 改为有确定文本的 `EditorToolParameterMissingError`；
- 本地 exception 实现可靠 `__str__`；editor 错误对模型的精确 observation 是：

```text
EXECUTION RESULT of [str_replace_editor]:
ERROR:
<error>
```

- 无效 command 在 schema/parser 层返回工具协议错误，不复刻 trajectory 中旧版本的 `AttributeError`；
- `undo_edit` 保留上游 0.1.0 的文件内容历史语义；
- 不 import `openhands_aci`。

### 3.7 finish 与 reward

`finish` 不要求 diff 非空才能终止：

1. 调用时获取并冻结当前 git diff；
2. 设置现有内部 terminal kind `submitted`；
3. trainer 立即停止该 sample，并回滚 finish 的 observation，使 completion 结束于 assistant 的 `<function=finish>`；
4. diff 非空：沿用现有独立 verifier；
5. diff 为空：关闭 rollout，不启动 verifier，reward 为 0；
6. verifier infrastructure failure 仍按现有异常路径处理；
7. 不修改 verifier 的“只接受非空 patch”合同。

`submitted` 只是现有内部领域命名，不是暴露给模型的旧 `submit` 工具。

## 4. 文件级变更

| 文件 | 动作 | 责任 |
|---|---|---|
| `src/swe_agent/openhands_editor.py` | 新建 | 自包含 editor、结果、异常、container backend |
| `src/swe_agent/tool_protocol.py` | 重写 | 本地三工具 schema、prompt suffix、parser、observation/fake-user renderer、tokenizer wrapper |
| `src/swe_agent/prompts.py` | 重写 | trajectory-aligned system/user prompt |
| `src/swe_agent/tools.py` | 重写 | 三工具校验与执行；调用本地 editor |
| `src/swe_agent/environment.py` | 修改 | 只暴露三个工具；alias；raw observation；finish |
| `src/swe_agent/trainer.py` | 修改 | OpenHands action/message/error loop 与 suffix mask |
| `src/swe_agent/train.py` | 修改 | processing class、trainer 参数、GRPO 配置、native path 证据 |
| `src/swe_agent/config.py` | 修改 | 删除 Qwen response-schema 字段，重命名协议错误上限 |
| `src/swe_agent/qualify.py` | 修改 | 验证单一 OpenHands render/parser |
| `configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml` | 删除 | 不保留旧模型/旧协议配置 |
| `configs/grpo_swegym_openhands_7b_lora.yaml` | 新建 | 唯一正式配置 |
| `scripts/dry_run.sh` | 修改 | 默认配置路径 |
| `scripts/grpo.sh` | 修改 | 默认配置路径 |
| `scripts/qualify.sh` | 修改 | 默认配置路径 |
| `tests/fixtures/openhands_protocol/system_prompt.txt` | 新建 | 本地精确 fixture |
| `tests/fixtures/openhands_protocol/cases.json` | 新建 | 本地解析/渲染 fixture |
| `tests/unit/test_openhands_editor.py` | 新建 | editor 算法与 backend 合同 |
| 现有协议相关 unit/integration tests | 修改 | 删除旧协议断言并覆盖新状态机 |
| `docs/third_party/openhands-scaffold-provenance.md` | 新建 | commit、hash、复制范围、适配差异 |

## 5. 实施任务

### Task 1：建立本地 provenance 和零外部依赖门禁

**Files**

- Create: `docs/third_party/openhands-scaffold-provenance.md`
- Modify: `tests/unit/test_project_layout.py`

- [ ] 在 provenance 文档记录第 2.3 节的 commit、文件 hash、实际复制的函数/常量和适配差异。
- [ ] 在 `test_project_layout.py` 新增 AST/结构化门禁：
  - Python import AST 中不出现 `openhands`、`openhands_aci`、LiteLLM、BrowserGym；
  - 运行代码中的路径构造不指向 `.external`；
  - `pyproject.toml` 的依赖数组不出现这些包；
  - fixture 加载点不读取 `.external`；
  - `SWEEnvironment` 的公开 callable 精确为三个工具，不能靠宽泛字符串扫描和巨型 allowlist 判断。
- [ ] 门禁只扫描运行代码和测试代码；provenance 文档允许包含上游名称。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_project_layout.py -q
```

Expected：exit code 0。

### Task 2：复制并本地化协议核心

**Files**

- Rewrite: `src/swe_agent/tool_protocol.py`
- Create: `tests/fixtures/openhands_protocol/system_prompt.txt`
- Create: `tests/fixtures/openhands_protocol/cases.json`
- Rewrite: `tests/unit/test_tool_protocol.py`

- [ ] 先从本地 trajectory 复制精确 system fixture 和代表性 response，不让测试读取 parquet 或 `.external`。
- [ ] 在 `tool_protocol.py` 本地定义：
  - converter/validation exception；
  - trajectory-exact 三工具 plain-dict schema；
  - `SYSTEM_PROMPT_SUFFIX_TEMPLATE`；
  - tool description renderer；
  - `FN_REGEX_PATTERN`、`FN_PARAM_REGEX_PATTERN`；
  - integer/array/string 参数转换；
  - required/allowed/enum 校验；
  - tool-call string renderer；
  - observation renderer；
  - fixed fake-user；
  - `parse_openhands_text(text: str)` 的 plain message/tool/protocol-error 分类。
- [ ] parser 只允许 schema 中的三个函数；单次解析最多返回一个可执行 action，duplicate/suffix 如何分类严格沿用锁定 converter，不额外发明拒绝规则。
- [ ] parser 对 duplicate function、closing 后 suffix 等边界遵循锁定 converter；不添加未经来源支持的严格规则。
- [ ] `finish` 没有参数时产生 `{}`。
- [ ] `view_range` 用 JSON array 解析；`insert_line` 用 integer 解析。
- [ ] reasoning prefix 保存在 assistant content 中，同时生成一个 tool call。
- [ ] 测试：
  - system fixture 字节完全一致且 hash 匹配；
  - 三工具顺序固定为 `execute_bash, finish, str_replace_editor`；
  - reasoning + call；
  - multiline parameter；
  - array/integer；
  - 缺 closing tag 是 protocol error；
  - unknown function；
  - missing/unknown/invalid enum parameter；
  - plain assistant message；
  - converter 已定义的 suffix/duplicate 边界；
  - 成功 `EXECUTION RESULT` 精确文本；
  - editor 失败 observation 精确包含 `ERROR:\n`；
  - fake-user 逐字节相等且 `.endswith("\n")`。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_tool_protocol.py -q
```

Expected：exit code 0。

### Task 3：替换 prompt 与 tokenizer rendering

**Files**

- Rewrite: `src/swe_agent/prompts.py`
- Modify: `src/swe_agent/train.py`
- Rewrite: `tests/unit/test_prompts.py`
- Modify: `tests/integration/test_trl_interfaces.py`

- [ ] `build_prompt(Task)` 使用：
  - trajectory-exact system base；
  - 本地 tool description/suffix；
  - OpenHands SWE issue user template；
  - `/workspace/<safe-task-id>` uploaded path。
- [ ] `build_processing_class()` 不再 import/call `add_response_schema`。
- [ ] 在 tokenizer 上幂等安装本地 parser 和 chat-render wrapper：
  - 丢弃 TRL 传入的 `tools=`；
  - 不改变其他 chat template kwargs；
  - 保留原方法引用供测试和诊断；
  - tokenizer `parse_response` 只支持 TRL 1.8.0 实际使用的 ids + `prefix=`；
  - 设置本地非空 response-template 激活标记，确保 `_generate` 进入 parser；
  - 不设置 `response_schema`，不调用 `add_response_schema`。
- [ ] rendered prompt 必须：
  - 包含三个 `---- BEGIN FUNCTION` block；
  - 包含 `<function=example_function_name>`；
  - 不包含 Qwen `<tool_call>` 指令；
  - 不包含旧六工具名。
- [ ] 真实 tokenizer round-trip 使用本地 checkpoint，不加载模型、不访问 GPU。
- [ ] 用固定版本的真实 Trainer 构造验证：
  - `supports_tool_calling=True`；
  - `is_chat_template_prefix_preserving=True`；
  - `trainer.chat_template is None`；
  - tokenizer parser 被实际调用；
  - wrapper 忽略 `trainer._env_tools[None]` 传入的 tools，渲染仍只有 OpenHands scaffold。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_prompts.py tests/unit/test_tool_protocol.py tests/integration/test_trl_interfaces.py -q
```

Expected：exit code 0。

### Task 4：复制并适配 editor

**Files**

- Create: `src/swe_agent/openhands_editor.py`
- Create: `tests/unit/test_openhands_editor.py`

- [ ] 先用 fake `ContainerFileBackend` 写失败测试，覆盖第 3.6 节全部行为。
- [ ] 把 ACI editor 必要代码复制进本地文件，不保留相对 import。
- [ ] 实现 `ContainerFileBackend`，所有容器命令使用 argv；write content 用 stdin。
- [ ] fake backend 测试精确覆盖：
  - absolute path；
  - file/directory/missing；
  - full/ranged view；
  - `view_range=[start,-1]`；
  - create existing/missing content；
  - unique/zero/multiple str_replace；
  - deletion by omitted `new_str`；
  - insert at 0、middle、EOF、out of range；
  - multiple undo、no history；
  - line numbering；
  - hidden directory exclusion；
  - two-level listing；
  - snippet window；
  - 16,000-char clipping notice；
  - backend timeout/exit/error conversion；
  - episode history isolation。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_openhands_editor.py -q
```

Expected：exit code 0。

### Task 5：用三工具替换六工具

**Files**

- Rewrite: `src/swe_agent/tools.py`
- Rewrite: `tests/unit/test_tools.py`

- [ ] `TOOL_SPECS` 精确为：
  - `execute_bash(command)`
  - `finish()`
  - `str_replace_editor(command, path, file_text?, old_str?, new_str?, insert_line?, view_range?)`
- [ ] 删除 `list_files/read_file/search_code/edit_file/run_command/submit` 及旧 inline file program。
- [ ] `execute_bash` 使用 workspace alias 作为 cwd，返回 OpenHands command observation。
- [ ] `str_replace_editor` 委托 episode-local `OpenHandsEditor`。
- [ ] `finish` 无论 diff 是否为空都冻结 patch 并成功。
- [ ] output bounding：
  - bash 用 30,000-char middle truncation；
  - editor 使用自身 16,000-char clipping；
  - 不再把 `Observation.model_dump_json()` 发给模型。
- [ ] unit tests 使用 fake sandbox 验证 argv、stdin、timeout、diff freeze 和错误文本。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_tools.py tests/unit/test_openhands_editor.py -q
```

Expected：exit code 0。

### Task 6：替换 environment 的公开协议面

**Files**

- Modify: `src/swe_agent/environment.py`
- Rewrite: `tests/unit/test_environment.py`

- [ ] `reset(task_id, **kwargs)` 保持现有签名，接受并忽略 TRL 传入的额外数据列；创建 `/workspace/<safe-task-id>` 与 `/repo` alias，任一步失败作为 infrastructure error 清理容器；成功时严格返回 `None`，不向首轮 user prompt 注入 readiness 文本。
- [ ] environment 只有三个公开工具方法：
  - `execute_bash`
  - `finish`
  - `str_replace_editor`
- [ ] `reset`、property 和私有 hook 不暴露为工具。
- [ ] `_call_tool()` 返回 raw observation text，同时仍记录现有 `Step/Observation`。
- [ ] plain assistant message 只保留在 TRL completion/history；不调用 environment，不伪造 `Action(tool_name="message")`、`Observation` 或 `Step`。
- [ ] protocol error 没有合法 action，不写 `Step`；错误只进入模型历史、连续错误计数和最终 loop-exit 归因。
- [ ] `finish`：
  - 设置 terminal event；
  - 保存空或非空 frozen patch；
  - 内部 `Step` 使用 `Action(tool_name="finish", arguments={})` 和空文本成功 `Observation`，该 observation 不进入模型历史；
  - 后续任何调用保持终止，不产生第二次 verifier；
  - finalize 空 patch 为 0，不调用 verifier；
  - finalize 非空 patch 走现有 verifier。
- [ ] 不修改 `models.py` 和 `verifier.py`。
- [ ] 接口回归精确锁定 `SWEEnvironment` 构造参数、`trajectory/verification/frozen_patch/terminated/episode_id`、`_finalize/_close/_drain_events`，保证冻结的 `swegym.py`、`rewards.py`、`train.py` 无需修改调用方式。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_environment.py tests/unit/test_models.py tests/unit/test_verifier.py -q
```

Expected：exit code 0。

### Task 7：把 TRL loop 改成 OpenHands 对话状态机

**Files**

- Modify: `src/swe_agent/trainer.py`
- Rewrite: `tests/integration/test_trainer_loop.py`

- [ ] 以当前固定的 TRL 1.8.0 `_tool_call_loop` 为同步基线，保留已有 generation、logprob、overlong、multimodal 和 environment pool 逻辑。
- [ ] 删除 Qwen format sentinel/message。
- [ ] 增加三类 active sample：tool action、plain message、protocol error。
- [ ] 每类按第 3.2、3.3 节追加 assistant/user history 和 mask。
- [ ] finish sample：
  - 执行 environment finish；
  - completion 保留 finish assistant token；
  - 不追加 finish observation；
  - 从 active index 移除，不影响 batch 其他 sample。
- [ ] plain message：
  - 不当作 rollout 完成；
  - 追加 fixed fake-user 并继续；
  - 不调用 environment，不写领域 `Step`。
- [ ] protocol error：
  - 不调用 environment public tool；
  - 追加 user-role ErrorObservation；
  - 达到 `max_consecutive_protocol_errors` 后归因为现有 `format_exhausted`，不新增领域 enum。
- [ ] 合法 tool/message 重置连续协议错误计数。
- [ ] context/iteration 归因沿用现有值。
- [ ] 用真实 tokenizer 验证每轮 prefix preserving；不能只用手工 token list mock。
- [ ] 每轮断言 `len(tool_mask[i]) == len(completion_ids[i])`，raw model token 为 1，runtime suffix 为 0，`sum(mask)` 等于累计 raw model token 数。
- [ ] 集成覆盖 reasoning+tool、plain message+fake-user、protocol error、finish、overlong rollback；逐例验证 ids/mask/messages 同步，finish 无 observation suffix。
- [ ] 集成验证 `reset()` 返回 `None`，首轮 prompt 不出现旧 “Fresh repository...” 文本。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/integration/test_trainer_loop.py tests/integration/test_trl_interfaces.py -q
```

Expected：exit code 0。

### Task 8：切换正式配置、资格和 native-path 判定

**Files**

- Modify: `src/swe_agent/config.py`
- Modify: `src/swe_agent/train.py`
- Modify: `src/swe_agent/qualify.py`
- Delete: `configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml`
- Create: `configs/grpo_swegym_openhands_7b_lora.yaml`
- Modify: `scripts/dry_run.sh`
- Modify: `scripts/grpo.sh`
- Modify: `scripts/qualify.sh`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_train.py`
- Modify: `tests/unit/test_qualify.py`
- Update all test config path constants

已确认需要更新配置路径常量的测试文件：

- `tests/integration/test_7b_gpu_qualification.py`
- `tests/integration/test_7b_vllm_qualification.py`
- `tests/integration/test_docker_qualification.py`
- `tests/integration/test_trl_interfaces.py`
- `tests/unit/test_config.py`
- `tests/unit/test_docker.py`
- `tests/unit/test_environment.py`
- `tests/unit/test_launcher.py`
- `tests/unit/test_qualify.py`
- `tests/unit/test_recording.py`
- `tests/unit/test_supervisor.py`
- `tests/unit/test_swegym.py`
- `tests/unit/test_train.py`
- `tests/unit/test_verifier.py`

- [ ] `ChatConfig` 只保留 prompt/observation 长度；删除两个 Qwen response-schema literal。
- [ ] `max_consecutive_format_errors` 重命名为 `max_consecutive_protocol_errors`。
- [ ] 唯一 YAML 使用本地 NovaSky checkpoint/tokenizer 绝对路径。
- [ ] 模型保持 `Qwen2ForCausalLM`、BF16 LoRA 和 context 32768。
- [ ] sampling 改为 checkpoint generation config 的 0.7/0.8/20/1.1。
- [ ] `chat.max_observation_chars=30000`。
- [ ] 配置重命名时保持当前已实现的 vLLM server 拓扑不变：`mode=server`、`enable_sleep_mode=false`、`tensor_parallel_size=null`、`gpu_memory_utilization=0.25`、`max_model_length=32768`；协议迁移不改 launcher 的双 GPU 拆分。
- [ ] `generation_kwargs` 不设置字符串 stop；终止使用 checkpoint EOS。
- [ ] `build_grpo_config()` 不设置 structured output regex，也不注入 Qwen response schema。
- [ ] `train.py::REQUIRED_DOMAIN_MODULES` 加入本地 `tool_protocol.py` 和 `openhands_editor.py`，使 preflight 覆盖新增的自包含核心。
- [ ] `_validate_rendered_prompt_length()` 保持使用 `trainer._env_tools[None]` 的现有调用；processing wrapper 必须忽略该 `tools=` 值，测试证明长度检查渲染的仍是唯一 OpenHands prompt。
- [ ] `_native_policy_path_reached()` 改为真实 trajectory 中出现：
  - 成功 `str_replace_editor` 编辑动作；
  - `finish`；
  - non-empty frozen patch；
  - verification；
  - reward 0 或 1。
- [ ] `qualify.check_tokenizer()` 断言：
  - environment 反射得到三个 public tool；
  - rendered system 只有 OpenHands function blocks；
  - parser 能解析目标模型实际格式；
  - rendered prompt 无 `<tool_call>`。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_train.py tests/unit/test_qualify.py -q
```

Expected：exit code 0。

### Task 9：真实 Docker 三工具资格

**Files**

- Rewrite: `tests/integration/test_docker_qualification.py`

- [ ] 只使用项目专用 `docker-swegym` daemon，不接触共享 Docker socket。
- [ ] 在真实固定 task container 中验证：
  - 三个 alias 指向同一 worktree；
  - editor directory/file view；
  - str_replace；
  - insert；
  - undo；
  - create；
  - bash cwd 和 exit-code observation；
  - non-empty finish diff；
  - empty finish 在独立 fresh container 中也能终止；
  - container close 后无本测试残留。
- [ ] 测试不读取 `.external`，不访问网络，不安装依赖。
- [ ] 运行：

```bash
.venv/bin/python -m pytest tests/integration/test_docker_qualification.py -q -m docker
```

Expected：exit code 0。

### Task 10：真实 tokenizer/model/vLLM 资格

**Files**

- Modify: `tests/integration/test_7b_gpu_qualification.py`
- Modify: `tests/integration/test_7b_vllm_qualification.py`

- [ ] GPU 测试名称与断言改为 OpenHands checkpoint。
- [ ] 先运行 `nvidia-smi`，只选择显存足够且不需要干预他人进程的设备；不满足条件则不执行本 gate。
- [ ] BF16 LoRA gate 沿用现有 forward/backward/save/reload 验证，模型路径换为目标 checkpoint。
- [ ] vLLM gate 使用 `build_processing_class()`，不能直接 `add_response_schema`。
- [ ] 真实 generation prompt 包含 OpenHands system scaffold。
- [ ] 使用固定真实 SWE prompt 和固定 seed 做协议 generation；输出能被分类为合法 tool call 或合法 plain message 即通过生成协议检查，同时必须断言没有旧 `<tool_call>`。不能因为一次随机输出选择合法 plain message 就误判模型不合格。
- [ ] 另用从真实 trajectory 固化的完整 function-call fixture 执行 parser→environment→下一轮 suffix 闭环；fixture 只验证确定性的协议执行路径，不能冒充模型实际生成。
- [ ] vLLM gate 按正式配置使用 launcher 管理的 server mode：第一张显式可见 GPU 给 server，第二张给 Trainer；验证 server health、真实 generation、PEFT weight sync 和关闭清理，不测试未启用的 sleep/wake。
- [ ] 单模型 BF16 gate 显式指定一个可用 GPU；server gate 显式指定两个可用 GPU；两者都不得杀进程释放显存。

Expected：所执行 gate exit code 0；未满足显存前置条件时报告“未执行”，不能报告通过。

### Task 11：全量回归和残留审计

- [ ] 运行非 GPU/Docker 默认测试：

```bash
.venv/bin/python -m pytest
```

Expected：exit code 0。

- [ ] 运行 Task 1 的 AST/结构化门禁，验证：
  - `SWEEnvironment` 公开工具方法精确为 `execute_bash/finish/str_replace_editor`；
  - `TOOL_SPECS` 与 environment 反射工具集合相等；
  - Python import AST 和项目依赖中没有外部 OpenHands/ACI/LiteLLM/BrowserGym；
  - 新协议模块和正式配置中没有旧 Qwen response schema、`<tool_call>` 或六工具定义。
- [ ] 仅把 `rg` 用于人工辅助检查明确的协议/config 文件，不用 `submit`、`read_file` 等普通词的全仓字符串零匹配作为验收，也不维护巨型 allowlist。内部 termination 名称 `submitted` 和 migration 断言可以存在。

- [ ] 检查依赖没有变化：

```bash
git diff -- pyproject.toml uv.lock
```

Expected：空。

- [ ] 检查冻结源码没有变化：

```bash
git diff -- \
  src/swe_agent/asset_generation.py \
  src/swe_agent/cli.py \
  src/swe_agent/docker.py \
  src/swe_agent/launcher.py \
  src/swe_agent/models.py \
  src/swe_agent/recording.py \
  src/swe_agent/reporting.py \
  src/swe_agent/rewards.py \
  src/swe_agent/supervisor.py \
  src/swe_agent/swegym.py \
  src/swe_agent/verifier.py \
  src/swe_agent/worker.py
```

Expected：空。

- [ ] 检查实际变更范围，确认没有覆盖用户已有 dirty changes：

```bash
git status --short
git diff --stat
git diff --check
```

Expected：实施新增的 diff 只覆盖本计划列出的文件，用户在实施前已有的 dirty changes 原样保留；`git diff --check` exit code 0。仓库在本阶段或实施开始时无需 clean，不能把用户计划稍后 commit 的现状误判为迁移失败。

## 6. 最终验收标准

只有同时满足以下条件，才能称为 OpenHands scaffold 迁移完成：

1. 模型首次看到的 system/user prompt 只有本地 OpenHands scaffold；
2. 模型输出的真实 `<function=...>` 能由本地 parser 解析；
3. reasoning 前缀不丢失；
4. 三工具名称、参数和 observation 与目标 trajectory 对齐；
5. 普通 assistant message 触发 fixed fake-user，而不是直接终止或被当作旧 Qwen format error；
6. `finish` 立即终止且无 observation；
7. 空 patch finish reward 0，非空 patch 仍由现有 verifier 决定 0/1；
8. TRL history 不渲染 Qwen `<tool_call>` 或 `role=tool`；
9. runtime-added user suffix token mask 为 0；
10. editor 的持久 history 只存在当前 episode；
11. 真实 Docker 资格通过且不留下测试容器；
12. 有足够显存时，目标 checkpoint 的 BF16 LoRA 与 vLLM gate 通过；
13. `src/tests/configs/scripts` 对 `.external`、OpenHands package、ACI package、LiteLLM、BrowserGym 零依赖；
14. `pyproject.toml`、锁文件和冻结的非协议源码无变化；
15. 默认测试与已执行的显式资格测试 exit code 0。

## 7. 明确非目标

- 不运行完整 GRPO 训练；
- 不做 Qwen/OpenHands 双协议；
- 不做 scaffold registry、adapter layer 或 config selector；
- 不把完整 OpenHands controller/event stream/runtime 搬进项目；
- 不引入 browser、IPython、delegation、LLM-based editor；
- 不复刻 OpenHands UI、server、session、security、confirmation；
- 不修改 verifier、reward 算法、Docker daemon 管理或任务资产；
- 不保证重放 47 条超过 checkpoint context 的历史 trajectory；
- 不把本地 copied code 再包装成对 `.external` 的薄代理；
- 不下载或安装新依赖；
- 不触碰共享服务器上其他人的 GPU 进程。

## 8. 回滚原则

本迁移是单协议替换，回滚单位是整个迁移变更，不保留运行时旧协议开关。

实施过程中任何 gate 失败：

1. 停在当前 Task；
2. 保留失败测试和原始错误证据；
3. 不增加 fallback 协议；
4. 不从 `.external` 临时 import 以绕过失败；
5. 不修改冻结的非协议源码规避问题；
6. 重新审查本地复制/适配是否遗漏，再更新计划或请求用户决策。

## 9. 阶段流程与用户操作

### 9.1 当前已完成

计划固化阶段完成以下目标后才结束：

1. 从 checkpoint 配置、SFT trajectory、锁定 OpenHands/ACI 来源和本地源码确认事实；
2. 画清模型可见协议、TRL token/history/mask、environment、Docker 和 verifier 的边界；
3. 决定采用单一 OpenHands 协议替换，排除旧 Qwen fallback、双协议和完整 OpenHands runtime；
4. 冻结非协议源码及其调用接口；
5. 把实现拆成协议、editor、environment、trainer、配置、Docker/GPU 资格和全量回归门禁；
6. 对本计划执行一致性与 diff 检查。

完成这些项目只代表**实施合同已固化**，不代表迁移源码已经完成，也不代表模型/GPU 资格已经通过。

### 9.2 获得实现授权后的固定流程

```text
保存实施前 diff 基线
  -> Task 1 provenance/结构门禁
  -> Task 2–3 协议、prompt、tokenizer
  -> Task 4–6 editor、三工具、environment
  -> Task 7 TRL loop 与 mask
  -> Task 8 配置、preflight、资格逻辑
  -> Task 9 Docker 真实闭环
  -> Task 10 有空闲显存时执行模型/vLLM gate
  -> Task 11 全量回归和变更范围审计
  -> 汇报；不自动 commit/push
```

每个 Task 均按“先失败测试 → 最小实现 → 该 Task 验证 → 再进入下一 Task”执行。若固定合同被真实证据推翻，暂停在该 Task，先修订计划，不用兼容层绕过。

### 9.3 用户需要做什么

当前不需要用户执行任何命令。仓库 dirty 状态留到计划固化后处理，不是当前门禁。

进入实现阶段只需要用户明确授权。实现和测试由 Agent 完成；不会自动 commit 或 push。GPU gate 只在显存足够时加载目标模型，不停止或调整他人进程。

若用户希望独立复核 ACI provenance，可选执行以下只读临时 checkout；这不是实施前置条件：

```bash
git clone --depth 1 --branch 0.1.0 \
  https://github.com/All-Hands-AI/openhands-aci.git \
  /tmp/openhands-aci-0.1.0-audit
git -C /tmp/openhands-aci-0.1.0-audit rev-parse HEAD
```

预期 commit 为：

```text
0698260b8e03ff2974ba81fd97ad8585a2255297
```
