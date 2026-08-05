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
