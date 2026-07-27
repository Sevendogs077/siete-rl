# Liger Kernel 接入 GRPO 训练 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 TRL 1.8.0 原生的 `use_liger_kernel` 路径（Liger fused-linear GRPO loss + 基础算子 patch），消除 trainer 卡在 policy 前向/反向中的 logits 物化尖峰，为后续上调 `max_completion_length` 腾出显存。

**Architecture:** 只动配置层和装配层：`pyproject.toml` 加依赖 → `GenerationConfig` 加 `use_liger_kernel` 开关（含 fail-fast 校验）→ `build_grpo_config` 透传给 TRL `GRPOConfig` → `dry_run.sh` 显存估算适配 liger 模式 → `recording.py` run 元数据记录该开关。rollout 侧（vLLM server）完全不动。

**Tech Stack:** TRL 1.8.0（`LigerFusedLinearGRPOLoss`）、transformers 5.13.0（`apply_liger_kernel`）、liger-kernel ≥0.8.0、uv 依赖管理、pytest。

**提交约定（用户指定，优先级高于各 Task 内的通用惯例）：** 实施过程中**不做任何 git commit**；全部 Task 完成、验证（Task 6）通过后，把本次改动连同工作区已有改动（`context_safety_margin: 0` 配置、docs 变更与新增计划文档等）一起 `git add -A` 做**唯一一个 commit**，message 采用仓库历史格式（`feat: <中文摘要>`，参考 `git log --oneline`）。

---

## 背景与调研结论（只读分析，已核实）

### 为什么是这个方案

当前 OOM 压力的主项不是参数/梯度/优化器（LoRA r16，三者合计 <1GB），而是 **logits 物化尖峰**：Qwen2-7B 词表 V=152k，`logits_to_keep` 段（≈ `max_completion_length`）的 logits bf16 每 8k token ≈ 2.4GB，加上 accelerate autocast 的 fp32 转换与 log_softmax 副本，最坏约为 bf16 的 6 倍（`scripts/dry_run.sh` 的估算模型即按此记账）。FSDP/TP 分片参数对此无效（见前序讨论），Liger fused-linear loss 让 policy 前向/反向**从不物化 logits**，是单卡内唯一对症的手段。

### TRL 1.8.0 的 liger 机制（源码核实）

`use_liger_kernel=True` 在 GRPOConfig（继承 TrainingArguments）下同时触发两件事：

1. **基础算子 patch**：`transformers/trainer.py:1375` 在 `Trainer.__init__` 中调用 `apply_liger_kernel(self.model, args.liger_kernel_config)`；`transformers/integrations/liger.py:28` 明确支持 "a PreTrainedModel **or a PEFT wrapper around one**"——内部 `unwrap_peft_model` 后对 base model 调 `_apply_liger_kernel_to_instance`（RMSNorm/RoPE/SwiGLU 等）。TRL 在 `super().__init__` **之前**完成 `get_peft_model`（`grpo_trainer.py:437`），时序安全。
2. **fused GRPO loss**：`grpo_trainer.py:949` 构造 `LigerFusedLinearGRPOLoss`；`compute_loss`（`grpo_trainer.py:2779`）改走 `compute_liger_loss`（`grpo_trainer.py:2710`）：backbone 取 `last_hidden_state`，直接读 `lm_head.weight` 做分块 fused matmul + loss，**logits 全程不物化**。`tool_mask`、`old/ref_per_token_logps`、`vllm_is_ratio`（vLLM importance sampling 修正）都作为输入传入，我们的多轮工具训练语义完整保留。

### 兼容性矩阵（逐项对照本仓库配置，全部通过）

| TRL 检查点（`grpo_trainer.py`） | 要求 | 本仓库现状 | 结论 |
|---|---|---|---|
| `:756-768` | PEFT adapter 不能打在 `lm_head` 上 | `peft.target_modules` 被 `config.py` validator 锁定为 `q,k,v,o_proj`（`LORA_TARGET_MODULES`），结构上不可能违规 | ✓ |
| `:773-780` | 不支持 prompt-learning PEFT | 纯 LoRA | ✓ |
| `:783-786` | `top_entropy_quantile` 必须 = 1.0 | 未设置，默认 1.0 | ✓ |
| `:787-791` | `importance_sampling_level ∈ {token, sequence}` | `token` | ✓ |
| `:754-755` | `off_policy_mask_threshold` 必须为 None | 未设置 | ✓ |
| `:804-805` | 不支持 entropy bonus | `entropy_coef`/`use_adaptive_entropy` 未设置，默认关闭 | ✓ |
| 依赖 | `liger-kernel>=0.8.0`（TRL METADATA `Requires-Dist`） | 未安装，本计划引入 | 待办 |

其余配置交互：`loss_type="grpo"`、`beta=0.04`（PEFT 下 ref 走 adapter-disable 前向，经 liger patch 后的 forward，labels=None 时 liger lce_forward 回退普通 logits 计算）、`vllm_importance_sampling_mode="sequence_mask"`（ratio 以 `(B,1)` 传入 liger loss）、`mask_truncated_completions=false`——均不受限。

### 显存收益与残余尖峰

- **消除**：policy 前向+反向的 logits 链（bf16 logits + fp32 转换 + log_softmax 副本 + 反向图），这是 `dry_run.sh` 里 `peak_typ/peak_worst` 的全部 logits 项。按当前 `max_completion_length=8192` 估算省 ~14–28GB 峰值；若后续提到 24576，省的更多。
- **残余**：`old_per_token_logps`（vLLM IS 修正开启时每步都算，`grpo_trainer.py:2416`）和 `ref_per_token_logps`（`beta≠0`）仍走普通 forward，no-grad 瞬态物化 bf16 logits，8k ≈ 2.4GB / 24k ≈ 7.3GB，无 fp32 副本、无反向图。可接受，不处理。
- **不变**：vLLM server 卡（rollout 数值、KV 预算）完全不受影响。

### 风险与回退

- **数值漂移**：liger 算子与 HF 参考实现不逐位一致，loss 曲线会有小幅偏移；train/infer mismatch 本就由 vLLM IS correction 覆盖，风险低。观察手段：对比切换前后 `train.log` 的 loss/kl/clip_ratio 量级。
- **可观测性损失**：liger loss 路径不计算 entropy 指标（`_compute_loss` 才有），日志中 entropy 消失。已知取舍。
- **回退**：YAML 单字段 `use_liger_kernel: false` 即完全回到现状，无代码分支污染。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `pyproject.toml` + `uv.lock` | Modify | 新增 `liger-kernel>=0.8.0,<1.0.0` 依赖 |
| `src/swe_agent/config.py` | Modify | `GenerationConfig` 加 `use_liger_kernel: bool`；`validate_contract` 加 IS-level 前置校验 |
| `configs/grpo_swegym_openhands_7b_lora.yaml` | Modify | `generation.use_liger_kernel: true` |
| `src/swe_agent/train.py` | Modify | `build_grpo_config` 透传 `use_liger_kernel` |
| `scripts/dry_run.sh` | Modify | liger 模式下 logits 尖峰记账改为「policy=0，old/ref 瞬态 bf16」 |
| `src/swe_agent/recording.py` | Modify | run 元数据记录 `use_liger_kernel` |
| `tests/unit/test_config.py` | Modify | schema 字段 + 校验 guard 测试 |
| `tests/unit/test_train.py` | Modify | 透传断言 |
| `tests/unit/test_recording.py` | Modify（如需要） | 元数据新字段断言 |

---

### Task 1: 新增 liger-kernel 依赖

**Files:**
- Modify: `pyproject.toml`（`dependencies` 列表，约第 11–26 行）
- Modify: `uv.lock`（由 `uv lock` 生成）

- [ ] **Step 1: 加依赖并锁定**

`pyproject.toml` 的 `dependencies` 列表中新增一行（保持现有列表顺序风格）：

```toml
  "liger-kernel>=0.8.0,<1.0.0",
```

```bash
cd /home/2025user/zyp/work/2607_trl_swe_agent
uv lock && uv sync
```

- [ ] **Step 2: 验证安装与 TRL 所需的符号可用**

```bash
.venv/bin/python -c "
import importlib.metadata, triton
print('liger-kernel', importlib.metadata.version('liger-kernel'))
print('triton', triton.__version__)
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
print('LigerFusedLinearGRPOLoss OK')
"
```

Expected: 打印版本号且 `LigerFusedLinearGRPOLoss OK`（TRL `grpo_trainer.py:103` 的导入符号；`>=0.8.0` 保证 TRL 构造参数 `sapo_*`/`vespo_*` 存在）。triton 应已随 torch/vllm 存在；若缺失，`uv sync` 会随 liger-kernel 依赖带入。

（本 Task 不 commit，见文首「提交约定」。）

### Task 2: config schema 增加 `use_liger_kernel` 开关与校验

**Files:**
- Modify: `src/swe_agent/config.py`（`GenerationConfig` 第 88–96 行；`validate_contract` 第 200–213 行）
- Modify: `configs/grpo_swegym_openhands_7b_lora.yaml`（`generation` 段）
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: 写失败测试**

`tests/unit/test_config.py` 末尾追加：

```python
def test_generation_config_exposes_use_liger_kernel() -> None:
    config, _, _ = load_config(CONFIG_7B)
    assert config.generation.use_liger_kernel is True


def test_use_liger_kernel_rejects_sequence_token_importance_sampling() -> None:
    config, _, _ = load_config(CONFIG_7B)
    payload = config.model_dump(mode="python")
    payload["grpo"]["importance_sampling_level"] = "sequence_token"
    with pytest.raises(ValidationError, match="use_liger_kernel"):
        ProjectConfig.model_validate(payload)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/unit/test_config.py -k liger -v
```

Expected: 第一个测试 `AttributeError: 'GenerationConfig' object has no attribute 'use_liger_kernel'`；第二个测试在 schema 未加字段时 `model_validate` 不抛错（`sequence_token` 本身是合法 Literal），`pytest.raises` 因无异常而 FAIL。

- [ ] **Step 3: 实现 schema 字段、YAML 与校验**

`src/swe_agent/config.py` 的 `GenerationConfig` 中，`context_safety_margin` 行后加：

```python
    use_liger_kernel: bool
```

`validate_contract` 中，`prompt and completion exceed vLLM max model length` 检查之后加：

```python
        if self.generation.use_liger_kernel and self.grpo.importance_sampling_level not in (
            "token",
            "sequence",
        ):
            raise ValueError(
                "use_liger_kernel requires importance_sampling_level 'token' or 'sequence'"
            )
```

`configs/grpo_swegym_openhands_7b_lora.yaml` 的 `generation:` 段，`context_safety_margin: 0` 行后加：

```yaml
  use_liger_kernel: true
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/unit/test_config.py -v
```

Expected: 全部 PASS（含既有测试——它们经 `load_config(CONFIG_7B)` 加载真实 YAML，`model_dump` round-trip 自动携带新字段，无需改 fixture）。

（本 Task 不 commit，见文首「提交约定」。）

### Task 3: `build_grpo_config` 透传到 TRL

**Files:**
- Modify: `src/swe_agent/train.py:181`（`build_grpo_config` 内 `max_completion_length=...` 行后）
- Test: `tests/unit/test_train.py`

- [ ] **Step 1: 写失败测试**

`tests/unit/test_train.py` 中 `test_public_peft_and_grpo_configs_construct_without_gpu` 之后新增：

```python
def test_grpo_config_enables_liger_kernel(tmp_path: Path) -> None:
    config, _, _ = load_config(CONFIG_7B)
    grpo_config = build_grpo_config(
        config,
        tmp_path / "output",
        seed=config.runtime.base_seed,
        use_cpu=True,
    )
    assert grpo_config.use_liger_kernel is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/unit/test_train.py::test_grpo_config_enables_liger_kernel -v
```

Expected: FAIL——`use_liger_kernel` 取到 TRL 默认值 `False`。

- [ ] **Step 3: 透传实现**

`src/swe_agent/train.py` 的 `build_grpo_config` 中，`max_completion_length=generation.max_completion_length,` 行后加：

```python
        use_liger_kernel=generation.use_liger_kernel,
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/unit/test_train.py -v
```

Expected: 全部 PASS。

（本 Task 不 commit，见文首「提交约定」。）

### Task 4: `dry_run.sh` 显存估算适配 liger 模式

**Files:**
- Modify: `scripts/dry_run.sh`（内嵌 Python 的 logits 记账与打印段）

说明：该脚本无自动化测试（纯配置推算打印），以运行输出人工核对为验收。

- [ ] **Step 1: 修改记账逻辑**

把脚本中这一段：

```python
steady = weights_gb + lora_gb + adam_gb + grad_gb + act_gb
peak_typ = steady + logits_bf16 + logits_fp32
peak_worst = peak_typ + logits_extra
```

替换为：

```python
steady = weights_gb + lora_gb + adam_gb + grad_gb + act_gb
use_liger = config.generation.use_liger_kernel
if use_liger:
    # Liger fused GRPO loss：policy 前向/反向不再物化 logits；
    # 仅 old/ref 的 no-grad logp 前向仍瞬态物化 bf16 logits（无 fp32 副本、无反向图）。
    peak_typ = steady + logits_bf16
    peak_worst = peak_typ
else:
    peak_typ = steady + logits_bf16 + logits_fp32
    peak_worst = peak_typ + logits_extra
```

- [ ] **Step 2: 修改打印段**

把脚本中这一段：

```python
print("[trainer 卡]  logits 尖峰（accelerate bf16→fp32 转换）")
print(f"  bf16 logits       {p(logits_bf16)} GB")
print(f"  fp32 转换          {p(logits_fp32)} GB")
print(f"  典型峰值 ≈ {p(peak_typ)} GB")
print(f"  最坏峰值 ≈ {p(peak_worst)} GB（含 log_softmax 第二份 fp32 副本）")
```

替换为：

```python
if use_liger:
    print("[trainer 卡]  logits 占用（liger 模式）")
    print("  policy logits      0.0 GB（fused loss 不物化）")
    print(f"  old/ref 瞬态      {p(logits_bf16)} GB（no-grad，bf16）")
else:
    print("[trainer 卡]  logits 尖峰（accelerate bf16→fp32 转换）")
    print(f"  bf16 logits       {p(logits_bf16)} GB")
    print(f"  fp32 转换          {p(logits_fp32)} GB")
print(f"  典型峰值 ≈ {p(peak_typ)} GB")
if use_liger:
    print(f"  最坏峰值 ≈ {p(peak_worst)} GB")
else:
    print(f"  最坏峰值 ≈ {p(peak_worst)} GB（含 log_softmax 第二份 fp32 副本）")
```

并在 `print(f"序列预算: prompt {T_prompt} + completion {T_comp}")` 行后加：

```python
print(f"Liger fused loss: {'开启' if use_liger else '关闭'}")
```

- [ ] **Step 3: 运行核对输出**

```bash
scripts/dry_run.sh
```

Expected: 输出含 `Liger fused loss: 开启`、`policy logits 0.0 GB`，且 `典型峰值` 相比修改前下降约 `logits_fp32` 一项（当前配置约 4.8GB）。`uv run` 不需要——脚本走 `.venv/bin/python`。

（本 Task 不 commit，见文首「提交约定」。）

### Task 5: run 元数据记录开关

**Files:**
- Modify: `src/swe_agent/recording.py:188`（`"max_completion_length": ...` 行后）
- Test: `tests/unit/test_recording.py`（按需）

- [ ] **Step 1: 加字段**

`src/swe_agent/recording.py` 元数据 dict 中，`"max_completion_length": config.generation.max_completion_length,` 行后加：

```python
                "use_liger_kernel": config.generation.use_liger_kernel,
```

- [ ] **Step 2: 跑测试并按需更新断言**

```bash
.venv/bin/pytest tests/unit/test_recording.py -v
```

Expected: PASS；若既有测试对元数据 dict 做精确键集合断言而 FAIL，则在该测试的期望 dict 中补 `"use_liger_kernel": True`（只加键，不改其他断言）。

（本 Task 不 commit，见文首「提交约定」。）

### Task 6: 全量验证

**Files:** 无修改，纯验证。

- [ ] **Step 1: 单元测试全量**

```bash
.venv/bin/pytest
```

Expected: 全量 PASS（addopts 已排除 gpu/docker/vllm/system marker）。

- [ ] **Step 2: 全套资格检查（含 GPU）**

```bash
scripts/qualify.sh
```

Expected: 全部检查通过。重点确认 trainer 构造阶段不抛 liger 相关异常（`apply_liger_kernel` 对 PeftModel 的 unwrap patch、`LigerFusedLinearGRPOLoss` 构造）。若 GPU 正被其他租户占满导致此项无法执行，记录原因并改在 GPU 空闲窗口执行，不得跳过。

- [ ] **Step 3: 真实短 run 观察（手动，GPU 空闲时）**

用当前配置跑一次 `scripts/grpo.sh`（或等 GPU 空闲的常规 run），观察：

- trainer 卡峰值显存相比上一个 run（`outputs/20260727T073521Z-524a` 时期）明显下降；
- `train.log` 中 loss / kl / clip_ratio 量级与历史 run 同数量级（liger 数值漂移应很小）；
- 日志不再出现 entropy 指标（预期内的可观测性变化）。

Expected: run 完整走完 `max_steps`，无 OOM、无 liger 报错。

- [ ] **Step 4: 统一单次 commit（本计划唯一的提交）**

确认 Step 1–3 全部通过后，把本次实施改动连同工作区已有改动（`context_safety_margin: 0`、docs 变更与新增计划文档等）一起提交为**唯一一个 commit**，message 用仓库历史格式（`类型: 中文摘要`）：

```bash
git add -A
git commit -m "feat: 接入 Liger fused GRPO loss 消除训练 logits 显存尖峰"
git status --short
git log --oneline -3
```

Expected: 仅新增 1 个 commit；`git status` 干净。
