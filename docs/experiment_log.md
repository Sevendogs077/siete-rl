# Experiment Log

Local records for SieteRL training and SWE-bench Verified evaluation, listed newest first under a fixed [evaluation protocol](#evaluation-protocol).

## 2026-08-10 — Reward shaping ablation

[![Scaffold v3](https://img.shields.io/badge/Scaffold-v3-04648c.svg)](#scaffold-compatibility)

> **Takeaway:** SieteRL-Agent-7B-v1 produced a richer training signal without improving verified performance.

### Results

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified | Δ vs. Base |
|---|---:|---:|---:|---:|
| **SieteRL-Agent-7B-Base** | — | — | **56/500 (11.2%)** | — |
| **SieteRL-Agent-7B-Vanilla-GRPO** | 0.10625 | 25/100 | **54/500 (10.8%)** | −2 |
| **SieteRL-Agent-7B-v1** | 0.11432 | 31/100 | **47/500 (9.4%)** | −9 |

**Method.** Vanilla-GRPO learns only from pass/fail verifier outcomes; v1 reshapes that sparse signal into an exponential transform of the bug-fix-test pass rate multiplied by regression-test retention, while excluding invalid-call and third-or-later repeated-action turns from the policy loss.

**Caveat.** This is not a single-factor ablation: v1 changed both reward shaping and process masking, used one training seed, and evaluated one rollout per task.

<details>
<summary><strong>Evidence and run details</strong></summary>

| Variant | Run | Trajectories | Exact / partial / zero reward | Scored / empty / error |
|---|---|---:|---:|---:|
| SieteRL-Agent-7B-Base | OpenHands-7B-Agent | — | — | 360 / 139 / 1 |
| SieteRL-Agent-7B-Vanilla-GRPO | `20260808T080700Z-8a86` | 1,600 | 170 / 0 / 1,430 | 352 / 148 / 0 |
| SieteRL-Agent-7B-v1 | `20260808T065031Z-f4cf` | 1,600 | 177 / 9 / 1,414 | 355 / 144 / 1 |

| Comparison | Newly solved | Regressed | Net change |
|---|---:|---:|---:|
| Vanilla-GRPO vs. Base | 12 | 14 | −2 |
| v1 vs. Base | 12 | 21 | −9 |
| v1 vs. Vanilla-GRPO | 15 | 22 | −7 |

“Scored” is a non-empty patch that completed harness evaluation; empty patches are separate, and evaluation errors count as unresolved.

- Partial credit reached 9/1,600 trajectories across 4/100 groups; six came from one group, and one scored 0.8333 despite breaking an existing behavior, showing that reward could overstate patch quality.
- The v1 mask suppressed 148 overlong negative-advantage trajectories; the masked negative/positive candidate-turn count was 807/238.
- On normal submissions, v1 resolved 2 more tasks than Vanilla-GRPO; on context-limit or iteration-cap exits, it resolved 9 fewer.
- All three variants shared only 29 resolved tasks. Vanilla-GRPO and v1 shared 32, with 69 in their union.
- One verified patch from each trained policy included 773 and 669 virtual-environment files, respectively, and was not directly deliverable.
- A group of 16 zero-reward trajectories produces no GRPO advantage; overlong context tokens receive no gradient.

</details>

## 2026-08-05 — Loss baseline

[![Scaffold v1](https://img.shields.io/badge/Scaffold-v1-04648c.svg)](#scaffold-compatibility)

> **Takeaway:** GRPO resolved three more tasks than DAPO.

### Results

| Variant | Mean reward | Non-degenerate groups | SWE-bench Verified | Δ vs. GRPO |
|---|---:|---:|---:|---:|
| GRPO | 0.084 | 17/100 | **56/500 (11.2%)** | — |
| DAPO | 0.074 | 15/100 | **53/500 (10.6%)** | −3 |

**Method.** Both runs used binary verifier rewards; the recorded configuration difference was `loss_type=grpo` versus `loss_type=dapo`.

**Caveat.** The three-task difference is not statistically meaningful under a single 500-task evaluation.

<details>
<summary><strong>Evidence and run details</strong></summary>

| Run | Submitted unresolved | Context limit | Iteration cap | Infra | Empty patch |
|---|---:|---:|---:|---:|---:|
| `20260801T235407Z-149b` | 190 | 173 | 78 | 3 | 152 |
| `20260802T065312Z-b873` | 179 | 196 | 69 | 3 | 150 |

Run `149b` had 254/500 (50.8%) non-submitting trajectories. Defining a loop as the same tool, command, path, and `old_str` prefix appearing at least three times, loops occurred in 57% of context-limit and 76% of iteration-cap trajectories.

- Common patterns were repeated `str_replace` failures, repeatedly viewing one file without acting, and retrying `pip install` in a network-isolated container.
- The SFT data assumed an active test environment and network access; Scaffold v2 corrected both training/evaluation mismatches.

</details>

## Evaluation protocol

- **Setup:** all 500 SWE-bench Verified tasks, one rollout per task, `temperature=0`, `top_p=1.0`, `max_model_length=32768`, and at most 40 tool turns.
- **Revisions:** dataset `91aa3ed51b709be6457e12d00300a6a596d4c6a3`; harness `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`.
- **Metrics:** a non-degenerate group contains at least two distinct rewards among its 16 rollouts.
- **Accounting:** `Resolved` always uses 500 as the denominator, and harness errors count as unresolved. Mutually exclusive outcomes are assigned infra → resolved → submitted → overlong → iteration cap; overlong and iteration-cap trajectories may still be scored from the container diff, so those buckets contain only unresolved tasks. `Empty patch` is cross-cutting and is not added to them.
- **Scope:** these are fixed local-harness results, not independently certified leaderboard submissions. One training seed and one deterministic evaluation rollout support directional, not definitive, comparisons.

## Scaffold compatibility

| Version | Commit | Changes |
|---|---|---|
| Scaffold v1 | — | Initial three-tool OpenHands-compatible scaffold; used by runs `149b` and `b873`. |
| Scaffold v2 | `451d0a2` | Matched the SFT truncation marker, injected conda testbed activation for login shells, and added the optional `max_repeat_action` warning. |
| Scaffold v3 | `d4c57fe` | Removed redundant activation injection; task images activate testbed through `.bashrc`. |

Only results within the same major scaffold version are directly comparable; cross-version comparisons must be labeled.

## External reference points

For context, SWE-Gym reports 1.8% for Qwen2.5-Coder-7B-Instruct zero-shot and 10.6% for SWE-Gym OpenHands-7B-Agent SFT; SkyRL reports 11.0% for OpenHands-7B-Agent base and 14.6% for SkyRL-Agent-7B-v0. See [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) and [SkyRL](https://github.com/NovaSky-AI/SkyRL). Different scaffolds, budgets, decoding parameters, or versions make these references contextual rather than controlled comparisons.
