# Termination

Termination records why generation stopped. After generation, settlement records
the result of evaluating the final workspace. They are separate: termination does
not determine whether the task was resolved.

| Termination | Meaning | Evaluate final workspace |
| --- | --- | --- |
| `submitted` | The agent called `finish` successfully. | Yes |
| `iteration_cap` | The tool loop exhausted its iteration budget. | Yes |
| `context_overlong` | The rollout exhausted its completion or context budget. | Yes |
| `format_exhausted` | Consecutive protocol errors reached the configured limit. | Yes |
| `infra_error` | Infrastructure failure stopped the interaction. | No |

For every termination except `infra_error`, SieteRL captures the final patch against
the task's base commit. The patch includes agent commits, staged changes,
unstaged changes, and untracked files.

An infrastructure failure during patch capture or verification does not replace
the recorded termination reason. It is recorded in settlement instead.

## Settlement

| Settlement | Meaning | Reward | Scorable |
| --- | --- | --- | --- |
| `resolved` | The patch applied and the verifier resolved the task. | Always `1.0` | Yes |
| `unresolved` | Patch application failed normally, or pytest completed without resolving the task. | Binary `0.0`, or the configured unresolved-only layered score | Yes |
| `empty_patch` | The final workspace has no patch relative to the base commit. | `0.0` | Yes |
| `agent_error` | The workspace Git command started, but policy-produced repository state prevented reliable settlement. | `0.0` | Yes |
| `infra_error` | Infrastructure failure prevented a reliable result. | `None` | No |

Verifier timeouts before pytest starts use `infra_error`. Once pytest has
started, a timeout is `unresolved` and receives zero reward. Cleanup failures
are recorded separately and never replace an existing result.

## Credit

A rollout with `settlement=infra_error` remains in its physical GRPO group but
does not contribute a reward or gradient. Its reward is `None`, it is excluded
from the group reward baseline, and its token weights are zero. The recorder
counts these rollouts in `censored_count` and `censored_rollouts`.

If every rollout in a microbatch has `settlement=infra_error`, that microbatch
contributes no gradient. Previously accumulated gradients are unchanged.

## `context_overlong`

| Case | Settlement | Token credit |
| --- | --- | --- |
| A complete action executed, but appending its observation exceeded the context budget. | Settle the resulting final workspace normally. | Keep the complete assistant turn. |
| Model generation was physically truncated. | Settle the resulting final workspace normally. | Mask only the final truncated assistant turn; keep all earlier complete turns. |

Physical truncation is independent of semantic turn kind: a parseable and executed action may remain a `step` while its final truncated token interval is excluded from credit.
