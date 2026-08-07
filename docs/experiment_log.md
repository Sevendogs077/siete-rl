# Experiment Log

## 固定评测口径

现有记录均使用 SWE-bench Verified 全部 500 个任务，每题一条 rollout，`temperature=0`、`top_p=1.0`、`max_model_length=32768`、最多 40 轮工具调用。数据 revision 为 `91aa3ed51b709be6457e12d00300a6a596d4c6a3`，评分 harness revision 为 `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`。

`Resolved` 的分母固定为全部 500 个任务；未完成 harness 评分的任务也按未解决计入。这里记录的是本地固定 harness 结果，不等同于 SWE-bench leaderboard 的独立认证。

结局栏的归位规则：每道题恰好计入一栏，优先级 infra > resolved > submitted > overlong > itercap；超窗或超轮的轨迹仍会以容器内 diff 兜底评分，因此这两栏只计未解决的部分。五栏合计恒为 500。`Empty Patch` 为交叉统计，不参与加总。

## 结果总览

| 完成日期 | Train run | 配置差异 | Resolved | Unresolved | Context Limit | Iteration Cap | Infra | Empty Patch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-08-05 | `20260801T235407Z-149b` | `loss_type=grpo` | **56（11.2%）** | 190（38.0%） | 173（34.6%） | 78（15.6%） | 3 | 152（30.4%） |
| 2026-08-05 | `20260802T065312Z-b873` | `loss_type=dapo`（legacy label） | 53（10.6%） | 179（35.8%） | 196（39.2%） | 69（13.8%） | 3 | 150（30.0%） |

两条 run 的 resolved 差异（+3 题）在 500 题单次评测下无统计显著性，仅作基线参照。

## 失败解剖（`20260801T235407Z-149b`）

未正常提交的轨迹合计 254 / 500（50.8%）。对其逐条取证（动作签名 = 工具 + 命令 + 路径 + old_str 前缀，同一签名出现 ≥3 次判定为循环）：

- **高重复占比**：context limit 轨迹 57% 存在动作循环，iteration cap 轨迹 76%。循环是未提交的主因，合法长探索只占约两三成。
- **三种典型循环模式**：
  - `str_replace` 空转：`old_str` 不匹配后原样重试（context limit 中 33 条连续失败 ≥3 次，iteration cap 中 23 条）。案例 `django__django-11734`：同一文件 view 19 次、同一编辑重试 5 次后超窗。
  - 反复 view 同一文件而不动作。
  - `pip install` 撞墙：容器 `network_mode: none`，模型沿用 SFT 分布中"缺包就装"的行为反复重试（12 条）。
- **环境失配佐证**：SFT 轨迹中 `pytest` 直接可调用（11 次中 9 次成功），而本 scaffold 镜像默认 PATH 未激活 conda testbed 环境；SFT 中 `pip install` 27 次成功，本 scaffold 容器无网络。两者已在 scaffold v2 修复。

## 训练侧指标

| Train run | reward mean | 非退化组占比 | steps |
|---|---:|---:|---:|
| `20260801T235407Z-149b` | 0.084 | 17% | 100 |
| `20260802T065312Z-b873` | 0.074 | 15% | 100 |

binary reward 下，组内 16 条 rollout reward 全同时 GRPO advantage 为零（退化组），不产生梯度。两条 run 的退化组占比分别为 83% 与 85%，即奖励稀疏的直接量化，也是引入 layered reward 的动机。

## Scaffold 版本

| 版本 | 起始 commit | 变更 |
|---|---|---|
| v1 | — | 初始 OpenHands 三工具 scaffold（149b、b873 均属 v1） |
| v2 | `451d0a2` | 截断标记对齐 SFT 原文（`[... Observation truncated due to length ...]`）；rollout 容器登录 shell 自动激活 conda testbed 环境；新增可配置的重复动作警告 `max_repeat_action`（默认关闭） |

评测结果只在同一大版本 scaffold 下直接可比；跨版本对比需在记录中注明。

## 外部参照

SWE-Gym 报告 Qwen2.5-Coder-7B-Instruct zero-shot 为 1.8%，SWE-Gym OpenHands-7B-Agent SFT 为 10.6%；SkyRL 公开记录 OpenHands-7B-Agent base 为 11.0%，SkyRL-Agent-7B-v0 为 14.6%。来源见 [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) 与 [SkyRL](https://github.com/NovaSky-AI/SkyRL)。这些结果的 scaffold、预算、解码参数或版本可能不同，只提供背景，不进入同条件比较。
