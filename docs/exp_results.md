# SWE-bench Verified

## Baseline

| 来源 | 模型 | Resolve rate |
|---|---|---:|
| SWE-Gym | Qwen2.5-Coder-7B-Instruct zero-shot | 1.8% |
| SWE-Gym | SWE-Gym OpenHands-7B-Agent SFT | 10.6% |
| SkyRL | OpenHands-7B-Agent base | 11.0% |
| SkyRL | SkyRL-Agent-7B-v0 | 14.6% |

## 实验

| 日期 | Run | Eval 目录 | Loss type | Resolve rate | Resolved | Official grading | Harness error | Empty patch | Context-limit termination |
|---|---|---|---|---:|---:|---:|---|---:|---|
| 2026-08-05 | `20260801T235407Z-149b` | `outputs/20260801T235407Z-149b/evals/20260804T055430.140908Z` | GRPO | 11.2% | 56/500 | 500/500 | 0 | 152 | 179/500（35.8%）；empty 84，non-empty 95，resolved 5 |
| 2026-08-05 | `20260802T065312Z-b873` | `outputs/20260802T065312Z-b873/evals/20260804T055429.868982Z` | DAPO | 10.6% | 53/500 | 499/500 | 1（模型补丁死循环超时） | 150 | 204/500（40.8%）；empty 98，non-empty 106，resolved 8 |

## Layered reward 消融（2026-08-06 起）

- 实验臂：`configs/grpo_swegym_openhands_7b_lora.yaml`，`reward_type: layered`，`λ=8.0`，`μ=ln2（0.6931471805599453）`
- 对照臂：同一配置仅 `reward_type: binary`（`configs/dapo_swegym_openhands_7b_lora.yaml` 保持 binary 不变）
- 关注指标：`rewards/layered_reward` 均值、全零组比例（all-zero group ratio）、resolve rate
- 注意：binary 模式的 TRL 指标列名为 `rewards/binary_reward`（列名按 `reward_type` 动态生成）

| 日期 | Run | Loss type | Reward | `rewards/layered_reward` 均值 | 全零组比例 | Resolve rate | 备注 |
|---|---|---|---|---:|---:|---:|---|
| 待回填 | 待回填 | GRPO | layered（λ=8, μ=ln2） | 结果待训练后回填 | 结果待训练后回填 | 结果待训练后回填 | 结果待训练后回填 |
