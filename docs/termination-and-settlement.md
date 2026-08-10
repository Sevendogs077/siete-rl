# Termination, Settlement, and Credit

Termination records why generation stopped. Settlement records how the final workspace was evaluated. Credit determines which generated tokens may receive an already-computed advantage. These are independent facts.

## Termination

| Termination | Meaning | Final workspace settlement |
| --- | --- | --- |
| `submitted` | The agent called `finish` successfully. | Yes |
| `iteration_cap` | The tool loop exhausted its iteration budget. | Yes |
| `context_overlong` | The rollout exhausted its completion or context budget. | Yes |
| `format_exhausted` | Consecutive protocol errors reached the configured limit. | Yes |
| `infra_error` | External infrastructure prevented a healthy rollout. | No; censored as infrastructure |

All healthy terminations capture the final patch against the task's immutable base commit. The patch includes agent commits, staged changes, unstaged changes, and untracked files.

## Settlement

| Settlement | Meaning | Reward | Scorable |
| --- | --- | --- | --- |
| `resolved` | The patch applied and the verifier resolved the task. | Always `1.0` | Yes |
| `unresolved` | Patch application failed normally, or pytest completed without resolving the task. | Binary `0.0`, or the configured unresolved-only layered score | Yes |
| `empty_patch` | The final workspace has no patch relative to the base commit. | `0.0` | Yes |
| `agent_error` | The workspace Git command started, but policy-produced repository state prevented reliable settlement. | `0.0` | Yes |
| `infra_error` | External execution or verifier infrastructure prevented reliable settlement. | `None` | No; censored |

Verifier timeouts are currently `infra_error`, because task qualification does not establish that every verifier completes within the active timeout budget. Cleanup failures are telemetry only and never replace a settlement or reward that was already obtained.

## Credit and Fixed Group Size

The effective token mask is the TRL base mask multiplied by the always-on credit mask and, when enabled, the optional process mask. Infrastructure rows receive an all-zero credit mask. They remain in the physical GRPO group, their `None` rewards are excluded from the reward baseline, and the loss keeps the original fixed-G denominator without compensation.

A fully censored microbatch contributes exactly zero new gradient to the current gradient accumulator. It does not erase gradients from earlier healthy microbatches. If an entire accumulation window is censored after `zero_grad(set_to_none=True)`, model parameters and optimizer state remain unchanged, while the outer framework still advances `global_step` and the scheduler.

## `context_overlong`

| Case | Settlement | Token credit |
| --- | --- | --- |
| A complete action executed, but appending its observation exceeded the context budget. | Settle the resulting final workspace normally. | Keep the complete assistant turn. |
| Model generation was physically truncated. | Settle the resulting final workspace normally. | Mask only the final truncated assistant turn; keep all earlier complete turns. |

Physical truncation is independent of semantic turn kind: a parseable and executed action may remain a `step` while its final truncated token interval is excluded from credit.
