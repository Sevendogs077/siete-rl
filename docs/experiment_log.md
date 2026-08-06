# Experiment Log

## 固定评测口径

现有记录均使用 SWE-bench Verified 全部 500 个任务，每题一条 rollout，`temperature=0`、`top_p=1.0`、`max_model_length=32768`、最多 40 轮工具调用。数据 revision 为 `91aa3ed51b709be6457e12d00300a6a596d4c6a3`，评分 harness revision 为 `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`。

`Resolved` 的分母固定为全部 500 个任务；未完成 harness 评分的任务也按未解决计入。这里记录的是本地固定 harness 结果，不等同于 SWE-bench leaderboard 的独立认证。

## Runs

| 完成日期 | Train / Eval run | 记录变体 | Reward | Resolved | Harness completed | Empty patch | Context-limit termination |
|---|---|---|---|---:|---:|---:|---|
| 2026-08-05 | `20260801T235407Z-149b`<br>`20260804T055430.140908Z` | `loss_type=grpo` | `binary_verifier` | **56/500（11.2%）** | 500/500 | 152 | 179/500（35.8%）：84 empty，95 non-empty，5 resolved |
| 2026-08-05 | `20260802T065312Z-b873`<br>`20260804T055429.868982Z` | `loss_type=dapo`（legacy label） | `binary_verifier` | 53/500（10.6%） | 499/500 | 150 | 204/500（40.8%）：98 empty，106 non-empty，8 resolved |

## 外部参照

SWE-Gym 报告 Qwen2.5-Coder-7B-Instruct zero-shot 为 1.8%，SWE-Gym OpenHands-7B-Agent SFT 为 10.6%；SkyRL 公开记录 OpenHands-7B-Agent base 为 11.0%，SkyRL-Agent-7B-v0 为 14.6%。来源见 [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) 与 [SkyRL](https://github.com/NovaSky-AI/SkyRL)。这些结果的 scaffold、预算、解码参数或版本可能不同，只提供背景，不进入同条件比较。
