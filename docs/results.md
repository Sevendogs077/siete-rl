# Results

Local records for SieteRL training and SWE-bench Verified evaluation, listed newest first under a fixed [evaluation protocol](#evaluation-protocol).

## 2026-08-13 — Reference reward ablation

[![Scaffold v3](https://img.shields.io/badge/Scaffold-v3-04648c.svg)](#scaffold-compatibility)

> **Result:** Adding a fixed `1.0` reference reward resolved three more tasks, from 54/500 to 57/500, without materially changing training reward diversity.

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified | Δ vs. No reference |
|---|---:|---:|---:|---:|
| No reference reward | 0.13138 | 31/100 | **54/500 (10.8%)** | — |
| **Reference reward (`1.0`)** | 0.13143 | 32/100 | **57/500 (11.4%)** | +3 |

**Analysis.** The variants differed only in `extra_reference_rewards`: the reference variant included `[1.0]`, while the ablated variant used `[]`. The reference reward is included in the GRPO group baseline but is not a sampled rollout. The 0.6-point Verified gap from one training seed and one deterministic evaluation rollout is directional rather than statistically conclusive.

## 2026-08-12 — Credit attribution refinement

[![Scaffold v3](https://img.shields.io/badge/Scaffold-v3-04648c.svg)](#scaffold-compatibility)

> **Result:** Refined credit attribution reached 58/500 on SWE-bench Verified while preserving more valid training credit.

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified |
|---|---:|---:|---:|
| **SieteRL-Agent-7B-Credit** | 0.13477 | 36/100 | **58/500 (11.6%)** |

**Analysis.** Layered verifier rewards were combined with sign-aware process masking: invalid or repeated actions lose positive credit but remain exposed to negative advantage. Infrastructure failures are excluded from the group baseline, while valid credit is retained across truncated or iteration-limited trajectories.

## 2026-08-10 — Reward shaping ablation

[![Scaffold v3](https://img.shields.io/badge/Scaffold-v3-04648c.svg)](#scaffold-compatibility)

> **Result:** Reward shaping produced a richer training signal without improving Verified performance.

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified | Δ vs. Base |
|---|---:|---:|---:|---:|
| **SieteRL-Agent-7B-Base** | — | — | **56/500 (11.2%)** | — |
| **SieteRL-Agent-7B-Vanilla-GRPO** | 0.10625 | 25/100 | **54/500 (10.8%)** | −2 |
| **SieteRL-Agent-7B-v1** | 0.11432 | 31/100 | **47/500 (9.4%)** | −9 |

**Analysis.** Vanilla-GRPO uses pass/fail rewards; v1 adds partial credit from bug-fix-test progress and regression retention, together with process masking. Because reward shaping and masking changed together, this single-seed result does not isolate either mechanism.

## 2026-08-05 — Loss baseline

[![Scaffold v1](https://img.shields.io/badge/Scaffold-v1-04648c.svg)](#scaffold-compatibility)

> **Result:** GRPO resolved three more tasks than DAPO.

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified | Δ vs. GRPO |
|---|---:|---:|---:|---:|
| GRPO | 0.084 | 17/100 | **56/500 (11.2%)** | — |
| DAPO | 0.074 | 15/100 | **53/500 (10.6%)** | −3 |

**Analysis.** Both variants used binary verifier rewards and differed only in `loss_type`. The three-task gap from one 500-task evaluation is directional rather than statistically conclusive.

## Evaluation protocol

- **Setup:** all 500 SWE-bench Verified tasks, one rollout per task, `temperature=0`, `top_p=1.0`, `max_model_length=32768`, and at most 40 tool turns.
- **Revisions:** dataset `91aa3ed51b709be6457e12d00300a6a596d4c6a3`; harness `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`.
- **Metrics:** a non-degenerate group contains at least two distinct rewards among its 16 rollouts.
- **Accounting:** `Resolved` always uses 500 as the denominator, and harness errors count as unresolved. Mutually exclusive outcomes are assigned infra → resolved → submitted → overlong → iteration cap; overlong and iteration-cap trajectories may still be scored from the container diff, so those buckets contain only unresolved tasks. `Empty patch` is cross-cutting and is not added to them.
- **Scope:** these are fixed local-harness results, not independently certified leaderboard submissions. One training seed and one deterministic evaluation rollout support directional, not definitive, comparisons.

## Scaffold compatibility

| Version | Commit | Changes |
|---|---|---|
| Scaffold v1 | — | Initial three-tool OpenHands-compatible scaffold used for the loss baseline. |
| Scaffold v2 | `451d0a2` | Matched the SFT truncation marker, injected conda testbed activation for login shells, and added the optional `max_repeat_action` warning. |
| Scaffold v3 | `d4c57fe` | Removed redundant activation injection; task images activate testbed through `.bashrc`. |

Only results within the same major scaffold version are directly comparable; cross-version comparisons must be labeled.

## External reference points

| Source | Model | SWE-bench Verified |
|---|---|---:|
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | Qwen2.5-Coder-7B-Instruct zero-shot | 1.8% |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | SWE-Gym OpenHands-7B-Agent SFT | 10.6% |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | OpenHands-7B-Agent base | 11.0% |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | SkyRL-Agent-7B-v0 | 14.6% |

Different scaffolds, budgets, decoding parameters, or versions make these contextual references rather than controlled comparisons.
