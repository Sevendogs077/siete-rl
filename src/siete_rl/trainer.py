"""GRPOTrainer 子类：OpenHands 对话状态机 + 循环结束归因。

`_tool_call_loop` 镜像 `trl==1.8.0` 的 `GRPOTrainer._tool_call_loop`，
仅在若干处插入 swe_agent 逻辑（以 `# >>> swe_agent` 标记）：

1. 所有未产生 ``tool_calls`` 的输出均规范为 ``parse_error``，并以哨兵
   tool call 留在循环内；唯一例外是被预算截断为空的生成（长度耗尽而
   非格式错误），直接归因 ``context_overlong`` 退出，不进格式恢复；
2. 哨兵样本跳过工具执行，注入 ``format_error`` 反馈消息（token 经既有
   suffix 机制 mask=0），命中配置的连续格式错误上限则退出；
3. 每轮工具执行后轮询 ``environment.terminated`` 与熔断计数，命中样本按
   TRL 自身的 overlong 回滚模式撤出循环；格式上限样本保留其最后一条
   assistant 输出以便记录；
4. overlong 撤出归因 ``context_overlong``，真实 tool call 重置连续计数；
5. 循环出口为迭代耗尽的样本归因 ``iteration_cap``，并把全部归因写回环境；
6. process mask：首段生成与每次 post-tool 生成各记录一条
   ``TurnRecord``（token 区间 + pending_action/step/invalid_call/plain_message
   分类）到 ``environment.turn_records``；生成的真实调用先记为
   ``pending_action``，真实工具执行前快照 ``len(env._steps)``，执行后按是否
   追加 Step 原子回填为 ``step`` 与 ``step_index``，或降级为 ``invalid_call``。

trainer 始终记录 turn facts，并在父类完成 reward/advantage 后先应用
always-on credit eligibility，再可选应用 process mask。返回的二元
``token_weights`` 替换 Liger reward-credit loss mask。

另新增 ``_generate_tool_loop_turn``：backport huggingface/trl#6673 —— tool loop
post-tool 再生成的 K 条 entry 各自携带独立采样的 history，不满足 server 模式
"num_generations 连续重复"的 stride 去重假设，必须 num_generations=1 逐条生成；
首 turn 仍走 TRL ``_generate_single_turn``（去重优化不动）。TRL 合并该 PR 后删除
此 helper，调用点改回 ``_generate_single_turn(..., 1)``。

另将 tool loop 单轮迭代内的跨样本工具执行并行化（``tool_parallel_workers``，
默认 1 = 原串行）：仅真实工具执行段进线程池，每样本一个 worker 按原顺序执行其
全部 tool calls；消息拼装、TurnRecord 回填、撤出与计数聚合仍由主线程按 idx 升序
完成。TRL 升级镜像同步时需保留该三段式结构。

TRL 升级时必须对照 `trl.trainer.grpo_trainer.GRPOTrainer._tool_call_loop`
人工同步本方法（镜像一致性由测试守护）。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch
from trl import GRPOTrainer
from trl.chat_template_utils import parse_response
from trl.extras.profiling import profiling_context
from trl.trainer.utils import nanstd

from siete_rl.models import LoopExit
from siete_rl.process_mask import (
    CreditMaskStats,
    ProcessMaskStats,
    TurnRecord,
    build_credit_token_weights,
    build_process_token_weights,
)


_PARSE_ERROR_SENTINEL = "swe_agent_parse_error"
"""哨兵 tool call：parse_error 样本留在循环内接受格式反馈，不进入真实工具执行。"""

_FORMAT_FEEDBACK_TEMPLATE = (
    "ERROR: the previous assistant response could not be parsed as one complete "
    "OpenHands function call ({reason}). Use <function=...> with valid parameters."
)
_PLAIN_MESSAGE_SENTINEL = "swe_agent_plain_message"


def _global_active_counts(local_count: int) -> list[int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [local_count]
    counts: list[int | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(counts, local_count)
    return [int(count) for count in counts]


# >>> swe_agent: process mask 的 turn 分类与记录 helper
def _classify_turn(tool_calls: list) -> str:
    """按生成态分类；真实 tool call 仅在执行成功后回填为 step。"""
    if tool_calls == [_PARSE_ERROR_SENTINEL]:
        return "invalid_call"
    if tool_calls == [_PLAIN_MESSAGE_SENTINEL]:
        return "plain_message"
    return "pending_action"


def _record_turn(
    environment,
    start: int,
    end: int,
    kind: str,
    step_index: int | None,
) -> None:
    """空区间不记录；生成的 pending_action 在工具执行后回填。"""
    if end <= start:
        return
    environment.turn_records.append(TurnRecord(start, end, kind, step_index))
# <<< swe_agent


class SWEGRPOTrainer(GRPOTrainer):
    """在官方 GRPOTrainer 上加入环境信号终止；其余行为与 TRL 完全一致。"""

    def __init__(
        self,
        *args,
        max_consecutive_protocol_errors: int,
        use_process_mask: bool,
        tool_parallel_workers: int = 1,
        extra_reference_rewards: tuple[float, ...] = (),
        **kwargs,
    ) -> None:
        if max_consecutive_protocol_errors < 1:
            raise ValueError("max_consecutive_protocol_errors must be positive")
        if tool_parallel_workers < 1:
            raise ValueError("tool_parallel_workers must be positive")
        self.max_consecutive_protocol_errors = max_consecutive_protocol_errors
        self._tool_parallel_workers = tool_parallel_workers
        self._use_process_mask = use_process_mask
        self._extra_reference_rewards = extra_reference_rewards
        super().__init__(*args, **kwargs)
        if not self.use_liger_kernel:
            raise ValueError("credit mask requires use_liger_kernel=true")

    def _await_environment_resets(self) -> float | None:
        """收束整批 reset；发生异常时也先 drain 其余 future。"""

        errors: list[tuple[int, BaseException]] = []
        timings: list[tuple[float, float]] = []
        for index, environment in enumerate(self.environments):
            try:
                observation = environment._await_reset()
                if observation is not None:
                    raise RuntimeError(
                        "SWEEnvironment.reset must return None; "
                        f"environment {index} returned {type(observation).__name__}"
                    )
            except BaseException as exc:
                errors.append((index, exc))
            try:
                timing = environment._reset_timing()
                if timing is None:
                    raise RuntimeError(
                        f"environment {index} is missing completed reset timing"
                    )
                timings.append(timing)
            except BaseException as exc:
                errors.append((index, exc))

        if errors:
            _, primary = errors[0]
            for index, extra in errors[1:]:
                primary.add_note(
                    "additional reset failure at environment "
                    f"{index}: {type(extra).__name__}: {extra}"
                )
            raise primary
        if not timings:
            return None
        started_at = min(start for start, _ in timings)
        finished_at = max(finish for _, finish in timings)
        if finished_at < started_at:
            raise RuntimeError("environment reset timing has a negative batch span")
        return finished_at - started_at

    def _generate(self, prompts: list):
        """在父类开始 GPU generation 前形成 environment reset barrier。"""

        reset_time = self._await_environment_resets()
        if reset_time is not None:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["environment/reset_time"].append(reset_time)
        return super()._generate(prompts)

    # >>> swe_agent: #6673 backport — tool loop 再生成禁止 server stride 去重
    def _generate_tool_loop_turn(self, prompt_ids, images, multimodal_fields):
        """Post-tool 再生成：每条 entry 携带各自独立采样的 history，必须 n=1 逐条生成。

        TRL 1.8.0 server 模式 ``VLLMGeneration.generate`` 假设 prompts 为
        ``num_generations`` 份连续重复并按 ``[::num_generations]`` 去重；tool loop
        的 K 条 distinct history 不满足该假设，会被塌缩到第一条活跃 trajectory 的
        lineage 再分发回所有 trajectory（见 run 20260807T034912Z-91b6 vllm.log：
        全部 /generate/ 请求只 render 1 个 prompt）。显式 ``num_generations=1``：
        server 下 ``[::1]`` 为恒等、n=1，K 进 K 出一一对应；colocate 本就 n=1，
        行为不变。对应 huggingface/trl#6673；TRL 合并后删除本方法。
        """
        if not self.use_vllm:
            return self._generate_single_turn(prompt_ids, images, multimodal_fields)
        if self.state.global_step != self._last_loaded_step:
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step
        if self.vllm_mode == "server" and torch.distributed.is_initialized():
            local_images = images if images is not None else [None] * len(prompt_ids)
            if len(local_images) != len(prompt_ids):
                raise RuntimeError("post-tool images must align with local prompts")
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            counts = _global_active_counts(len(prompt_ids))
            gathered_prompts = [None] * world_size
            gathered_images = [None] * world_size
            torch.distributed.all_gather_object(gathered_prompts, prompt_ids)
            torch.distributed.all_gather_object(gathered_images, local_images)
            all_prompts = [item for batch in gathered_prompts for item in batch]
            all_images = [item for batch in gathered_images for item in batch]
            if all(image is None for image in all_images):
                all_images = None

            payload = None
            if rank == 0:
                generation = self.vllm_generation
                output = generation.vllm_client.generate(
                    prompts=all_prompts,
                    images=all_images,
                    n=1,
                    repetition_penalty=generation.repetition_penalty,
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    top_k=generation.top_k,
                    min_p=0.0 if getattr(generation, "min_p", None) is None else generation.min_p,
                    max_tokens=generation.max_completion_length,
                    logprobs=generation.logprobs,
                    structured_outputs_regex=getattr(
                        generation, "structured_outputs_regex", None
                    ),
                    generation_kwargs=generation.generation_kwargs,
                )
                payload = (output["completion_ids"], output["logprobs"])
            objects = [payload]
            torch.distributed.broadcast_object_list(objects, src=0)
            all_completion_ids, all_logprobs = objects[0]
            start = sum(counts[:rank])
            stop = start + len(prompt_ids)
            completion_ids = all_completion_ids[start:stop]
            logprobs = all_logprobs[start:stop]
            logprobs = (
                [[lp[0] for lp in sequence] for sequence in logprobs]
                if logprobs is not None
                else None
            )
            return completion_ids, logprobs
        _, completion_ids, logprobs, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=images,
            num_generations=1,
            profiler=profiling_context(self, "vLLM.generate"),
        )
        logprobs = [[lp[0] for lp in seq] for seq in logprobs]
        return completion_ids, logprobs
    # <<< swe_agent

    # >>> swe_agent: 跨样本并行 worker 只动自己的 env，返回增量与结果供主线程聚合
    def _execute_tool_calls(self, tool_call_list, sync_tool_dict, async_tool_dict):
        """按原顺序执行单样本的全部 tool calls，返回计数增量与结果。"""
        n_calls = 0
        n_failures = 0
        async_coros = []
        tool_call_results = []
        for tool_call in tool_call_list:
            n_calls += 1
            if tool_call["type"] == "function":
                function = tool_call["function"]
                name = function["name"]
                try:
                    if name in sync_tool_dict:
                        tool_call_results.append(
                            (name, sync_tool_dict[name](**function["arguments"]))
                        )
                    elif name in async_tool_dict:
                        async_coros.append(
                            (name, async_tool_dict[name](**function["arguments"]))
                        )
                    else:
                        raise ValueError(f"Tool {name} not found.")
                except Exception as exc:
                    n_failures += 1
                    tool_call_results.append((name, {"error": str(exc)}))
            else:
                n_failures += 1
                name = tool_call.get("name", "unknown")
                tool_call_results.append(
                    (
                        name,
                        {
                            "error": (
                                "Unsupported tool call type: "
                                f"{tool_call['type']}"
                            )
                        },
                    )
                )

        if async_coros:

            async def _run_async_tools(coros_with_names):
                coros = [coro for _, coro in coros_with_names]
                results = await asyncio.gather(*coros, return_exceptions=True)
                return [
                    (name, result)
                    for (name, _), result in zip(
                        coros_with_names, results, strict=False
                    )
                ]

            async_results = asyncio.run_coroutine_threadsafe(
                _run_async_tools(async_coros), self.async_loop
            ).result()

            for name, result in async_results:
                if isinstance(result, Exception):
                    n_failures += 1
                    tool_call_results.append((name, {"error": str(result)}))
                else:
                    tool_call_results.append((name, result))
        return n_calls, n_failures, tool_call_results
    # <<< swe_agent

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        environments = self.environments
        if self._extra_reference_rewards:
            num_generations = (
                self.num_generations
                if self.model.training
                else self.num_generations_eval
            )
            rewards = torch.tensor(
                [
                    torch.nan if environment._reward is None else environment._reward
                    for environment in environments
                ],
                dtype=output["advantages"].dtype,
                device=output["advantages"].device,
            ).view(-1, num_generations)
            references = torch.tensor(
                self._extra_reference_rewards,
                dtype=rewards.dtype,
                device=rewards.device,
            ).expand(rewards.size(0), -1)
            baseline_rewards = torch.cat((rewards, references), dim=1)
            advantages = rewards - torch.nanmean(
                baseline_rewards, dim=1, keepdim=True
            )
            if self.scale_rewards == "group":
                advantages = advantages / (
                    nanstd(baseline_rewards, dim=1, keepdim=True) + 1e-4
                )
            elif self.scale_rewards == "batch":
                advantages = advantages / (nanstd(baseline_rewards) + 1e-4)
            output["advantages"] = torch.nan_to_num(advantages, nan=0.0).flatten()
        completion_mask = output["completion_mask"]
        tool_mask = output.get("tool_mask")
        advantages = output.get("advantages")
        if (
            not environments
            or len(environments) != completion_mask.size(0)
            or tool_mask is None
            or tool_mask.shape != completion_mask.shape
            or advantages is None
            or advantages.ndim != 1
            or advantages.shape[0] != completion_mask.size(0)
        ):
            raise RuntimeError(
                "credit mask requires aligned environments, tool_mask, and advantages"
            )

        base_mask = (completion_mask * tool_mask).float()
        token_weights = torch.zeros_like(base_mask)
        credit_stats_rows = []
        process_stats_rows = []
        for row_index, environment in enumerate(environments):
            row, credit_stats = build_credit_token_weights(
                turns=environment.turn_records,
                termination=environment.trajectory.termination,
                settlement=environment.trajectory.settlement,
                base_mask=base_mask[row_index].tolist(),
            )
            credit_stats_rows.append(credit_stats)
            if self._use_process_mask:
                row, process_stats = build_process_token_weights(
                    turns=environment.turn_records,
                    steps=environment._steps,
                    termination=environment.trajectory.termination,
                    advantage=float(advantages[row_index].item()),
                    base_mask=row,
                )
                process_stats_rows.append(process_stats)
            token_weights[row_index] = torch.tensor(
                row,
                dtype=token_weights.dtype,
                device=token_weights.device,
            )
        output["token_weights"] = token_weights
        self._record_credit_mask_metrics(credit_stats_rows)
        if self._use_process_mask:
            self._record_process_mask_metrics(process_stats_rows)
        self._record_recovered_positive_support(
            environments, advantages, token_weights
        )
        return output

    def _record_process_mask_metrics(self, rows: list[ProcessMaskStats]) -> None:
        mode = "train" if self.model.training else "eval"
        n_rows = len(rows)
        if n_rows == 0:
            raise RuntimeError("process mask stats require at least one row")
        self._metrics[mode]["process_mask/candidate_turns"].append(
            float(sum(row.candidate_turns for row in rows))
        )
        self._metrics[mode]["process_mask/applied_turns"].append(
            float(sum(row.applied_turns for row in rows))
        )
        self._metrics[mode]["process_mask/retained_negative_turns"].append(
            float(sum(row.retained_negative_turns for row in rows))
        )
        self._metrics[mode]["process_mask/masked_token_frac"].append(
            sum(row.masked_token_frac for row in rows) / n_rows
        )

    def _record_credit_mask_metrics(self, rows: list[CreditMaskStats]) -> None:
        mode = "train" if self.model.training else "eval"
        n_rows = len(rows)
        if n_rows == 0:
            raise RuntimeError("credit mask stats require at least one row")
        self._metrics[mode]["credit_mask/infra_rows"].append(
            float(sum(row.infra_rows for row in rows))
        )
        self._metrics[mode]["credit_mask/truncated_turns"].append(
            float(sum(row.truncated_turns for row in rows))
        )
        self._metrics[mode]["credit_mask/masked_token_frac"].append(
            sum(row.masked_token_frac for row in rows) / n_rows
        )

    def _record_recovered_positive_support(
        self, environments, advantages, token_weights
    ) -> None:
        mode = "train" if self.model.training else "eval"
        recovered_indices = [
            index
            for index, environment in enumerate(environments)
            if environment.trajectory.termination
            in {"iteration_cap", "format_exhausted", "context_overlong"}
            and getattr(environment, "verification", None) is not None
            and environment.verification.result == "resolved"
        ]
        active_tokens = sum(
            float(token_weights[index].sum().item())
            for index in recovered_indices
            if float(advantages[index].item()) != 0.0
        )
        self._metrics[mode]["settlement/recovered_positive_rows"].append(
            float(len(recovered_indices))
        )
        self._metrics[mode][
            "settlement/recovered_positive_active_tokens"
        ].append(active_tokens)

    def compute_liger_loss(self, unwrapped_model, inputs):
        # token_weights 的支撑集已在 completion_mask×tool_mask 内，替换后父类的
        # loss_mask = completion_mask * token_weights ≡ token_weights。
        if "token_weights" in inputs:
            token_weights = inputs["token_weights"]
            completion_mask = inputs.get("completion_mask")
            effective_mask = (
                token_weights
                if completion_mask is None
                else token_weights * completion_mask
            )
            if not torch.any(effective_mask):
                dtype = (
                    token_weights.dtype
                    if token_weights.is_floating_point()
                    else torch.float32
                )
                return torch.zeros(
                    (), device=token_weights.device, dtype=dtype, requires_grad=True
                )
            inputs = dict(inputs, tool_mask=token_weights)
        return super().compute_liger_loss(unwrapped_model, inputs)

    @staticmethod
    def _next_action(completion):
        """将 parser 结果区分为工具、普通消息与协议错误。"""

        tool_calls = completion.get("tool_calls")
        if tool_calls:
            return tool_calls
        completion.setdefault("role", "assistant")
        completion.setdefault("content", "")
        if completion.get("parse_error"):
            return [_PARSE_ERROR_SENTINEL]
        return [_PLAIN_MESSAGE_SENTINEL]

    def _get_openhands_suffix_ids(self, messages):
        """渲染 runtime user turn 与下一轮 generation marker。

        OpenHands scaffold 只使用普通 assistant/user 对话；因此不能调用 TRL
        专用于 ``role=tool`` 的 `_get_tool_suffix_ids`。真实 tokenizer 的
        prefix-preserving 性质由接口集成测试覆盖；裸 trainer 测试可提供这个
        helper 的替身。
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                # Transformers 5 的 tokenizer 在 tokenize=True 时默认可返回
                # BatchEncoding；TRL 后续会把 suffix 当作 token list 拼接。
                return_dict=False,
            )
            return list(rendered)
        return self._get_tool_suffix_ids(messages)

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        # OpenHands 允许普通 assistant message；只有显式 parse_error 才进入恢复。
        format_error_counts = [0] * len(completions)
        tool_calls = [
            self._next_action(completion[0])
            for calls, completion in zip(tool_calls, completions, strict=True)
        ]
        # <<< swe_agent
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        last_turn_truncated = [False] * len(completions)
        # >>> swe_agent: 仅由项目终止语义写回循环退出原因
        loop_exit_reasons: dict[int, LoopExit] = {}
        # <<< swe_agent
        # >>> swe_agent: 首段生成记录为 turn，token 区间为 [0, len(completion_ids[idx]))
        if self.environments:
            for idx, calls in zip(idxs_with_tool, tool_calls, strict=True):
                environment = self.environments[idx]
                if environment is not None:
                    _record_turn(environment, 0, len(completion_ids[idx]), _classify_turn(calls), None)
                    last_turn_truncated[idx] = self._generation_truncated(
                        completion_ids[idx]
                    )
                    if last_turn_truncated[idx]:
                        loop_exit_reasons[idx] = "context_overlong"
        # <<< swe_agent
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0

        while (
            sum(_global_active_counts(len(idxs_with_tool))) > 0
            and iteration_num < self.max_tool_calling_iterations
        ):
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = {
                i: len(completions[i]) for i in idxs_with_tool
            }
            tool_images_len_before = {
                i: len(tool_images[i]) for i in idxs_with_tool
            }
            prompts_len_before = {i: len(prompts[i]) for i in idxs_with_tool}

            # Call the tools, and build the new prompt for generation
            # >>> swe_agent: 三段式——串行前处理 / 跨样本并行执行 / 串行聚合
            exec_jobs = []
            steps_before_by_idx: dict[int, int] = {}
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                # >>> swe_agent: 哨兵样本跳过工具执行，注入格式反馈消息（mask=0 由 suffix 机制保证）
                if tool_call_list == [_PARSE_ERROR_SENTINEL]:
                    format_error_counts[idx_with_tool] += 1
                    if format_error_counts[idx_with_tool] < self.max_consecutive_protocol_errors:
                        reason = completions[idx_with_tool][-1].get("parse_error", "unknown parse error")
                        feedback = {"role": "user", "content": _FORMAT_FEEDBACK_TEMPLATE.format(reason=reason)}
                        prompt_completion_tool.append(feedback)
                        completions[idx_with_tool].append(feedback)
                    continue
                if tool_call_list == [_PLAIN_MESSAGE_SENTINEL]:
                    # 普通消息不是工具失败，也不写领域 Step。
                    from siete_rl.tool_protocol import FIXED_FAKE_USER
                    prompt_completion_tool.append({"role": "user", "content": FIXED_FAKE_USER})
                    completions[idx_with_tool].append({"role": "user", "content": FIXED_FAKE_USER})
                    format_error_counts[idx_with_tool] = 0
                    continue
                # <<< swe_agent
                # >>> swe_agent: 记录执行前 step 数，用于 step_index 回填与契约错误降级
                environment = self.environments[idx_with_tool] if self.environments else None
                # 刻意的跨模块契约：_steps 是 SWEEnvironment 的私有属性，此处只读长度
                steps_before_by_idx[idx] = len(environment._steps) if environment is not None else 0
                # <<< swe_agent
                exec_jobs.append(
                    (
                        idx,
                        idx_with_tool,
                        tool_call_list,
                        sync_tool_dict,
                        async_tool_dict,
                    )
                )

            # >>> swe_agent: worker=1 或单样本时不建池，保持原串行基线
            if self._tool_parallel_workers > 1 and len(exec_jobs) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(len(exec_jobs), self._tool_parallel_workers)
                ) as pool:
                    exec_outcomes = list(
                        pool.map(
                            lambda job: self._execute_tool_calls(
                                job[2], job[3], job[4]
                            ),
                            exec_jobs,
                        )
                    )
            else:
                exec_outcomes = [
                    self._execute_tool_calls(job[2], job[3], job[4])
                    for job in exec_jobs
                ]
            # <<< swe_agent

            # >>> swe_agent: 主线程按 idx 升序聚合消息、计数与 step_index
            for (
                (idx, idx_with_tool, _, _, _),
                (n_calls, n_failures, tool_call_results),
            ) in zip(exec_jobs, exec_outcomes, strict=True):
                tool_call_count += n_calls
                tool_failure_count += n_failures
                prompt_completion_tool = prompt_completion_tools[idx]
                environment = self.environments[idx_with_tool] if self.environments else None
                steps_before = steps_before_by_idx[idx]
                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    # OpenHands observation 是普通 user turn，而不是 Qwen role=tool。
                    from siete_rl.tool_protocol import render_observation
                    tool_message = {"role": "user", "content": render_observation(name, str(content), error=isinstance(result, dict) and "error" in result)}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

                # >>> swe_agent: 原子回填 pending；契约错误（未追加 Step）降级为 invalid_call
                if environment is not None:
                    if not environment.turn_records or environment.turn_records[-1].kind != "pending_action":
                        raise RuntimeError(
                            "real tool execution requires a final pending_action turn"
                        )
                    last = environment.turn_records[-1]
                    if len(environment._steps) > steps_before:
                        environment.turn_records[-1] = replace(
                            last,
                            kind="step",
                            step_index=len(environment._steps) - 1,
                        )
                    else:
                        # infra 错误（DockerRuntimeError 未追加 Step）也走此降级，归类 invalid_call 是保守处理
                        environment.turn_records[-1] = replace(
                            last, kind="invalid_call", step_index=None
                        )
                # <<< swe_agent
            # <<< swe_agent

            # >>> swe_agent: 轮询环境终止信号与格式上限；上限样本保留最终无效输出以供记录
            if self.environments:
                terminated_flags = [
                    bool(getattr(self.environments[idx_with_tool], "terminated", False))
                    for idx_with_tool in idxs_with_tool
                ]
            else:
                terminated_flags = [False] * len(idxs_with_tool)
            breaker_flags = [
                format_error_counts[idx_with_tool] >= self.max_consecutive_protocol_errors
                for idx_with_tool in idxs_with_tool
            ]
            truncated_flags = [
                last_turn_truncated[idx_with_tool]
                for idx_with_tool in idxs_with_tool
            ]
            exit_flags = [
                terminated or breaker or truncated
                for terminated, breaker, truncated in zip(
                    terminated_flags, breaker_flags, truncated_flags, strict=True
                )
            ]
            if any(exit_flags):
                for idx in range(len(idxs_with_tool)):
                    if exit_flags[idx]:
                        idx_with_tool = idxs_with_tool[idx]
                        if truncated_flags[idx]:
                            del completions[idx_with_tool][completions_len_before[idx_with_tool] :]
                            del tool_images[idx_with_tool][tool_images_len_before[idx_with_tool] :]
                            del prompts[idx_with_tool][prompts_len_before[idx_with_tool] :]
                            loop_exit_reasons[idx_with_tool] = "context_overlong"
                        elif terminated_flags[idx]:
                            del completions[idx_with_tool][completions_len_before[idx_with_tool] :]
                            del tool_images[idx_with_tool][tool_images_len_before[idx_with_tool] :]
                            del prompts[idx_with_tool][prompts_len_before[idx_with_tool] :]
                        elif breaker_flags[idx]:
                            loop_exit_reasons[idx_with_tool] = "format_exhausted"
                idxs_with_tool = [
                    idx for idx, flag in zip(idxs_with_tool, exit_flags, strict=True) if not flag
                ]
            # <<< swe_agent

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "user":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_openhands_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx_with_tool] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx_with_tool] :]
                    del prompts[idx_with_tool][prompts_len_before[idx_with_tool] :]
            # >>> swe_agent: overlong 撤出的样本归因 context_overlong
            for idx, o in zip(idxs_with_tool, overlong, strict=True):
                if o:
                    loop_exit_reasons[idx] = "context_overlong"
            # <<< swe_agent
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]

            # Filter images and multimodal fields to match the current subset (index into full batch).
            # Merge tool response images so the model can see visual feedback during generation.
            merged_images = images
            if any(imgs for imgs in tool_images):
                if merged_images is None:
                    merged_images = [imgs if imgs else None for imgs in tool_images]
                else:
                    merged_images = [
                        (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                    ]
            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
            if multimodal_fields:
                loop_multimodal_fields = {}
                for k, v in multimodal_fields.items():
                    selected = [v[i] for i in idxs_with_tool]
                    # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                    if selected and isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    loop_multimodal_fields[k] = selected
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            # >>> swe_agent: #6673 backport — K 条 distinct history 必须 n=1 逐条生成（见 _generate_tool_loop_turn）
            post_tool_ids, post_tool_logprobs = self._generate_tool_loop_turn(
                prompt_completion_tool_ids, loop_images, loop_multimodal_fields
            )
            # <<< swe_agent
            post_tool_truncated = [
                self._generation_truncated(ids) for ids in post_tool_ids
            ]

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    post_tool_truncated[idx] = True
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self._tokenizer, ids, prefix=prompt_completion_tool_ids[idx]) if ids else {}
                for idx, ids in enumerate(post_tool_ids)
            ]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            # >>> swe_agent: 空生成归因 context_overlong；其余无 tool call 输出进入格式恢复
            tool_calls = []
            for pos, (idx, completion) in enumerate(zip(idxs_with_tool, post_tool_completions, strict=True)):
                if not completion:
                    # 生成被预算截断为空：长度耗尽而非格式错误，不进格式恢复
                    loop_exit_reasons[idx] = "context_overlong"
                    tool_calls.append(None)
                    continue
                calls = self._next_action(completion)
                if calls != [_PARSE_ERROR_SENTINEL]:
                    format_error_counts[idx] = 0
                tool_calls.append(calls)
                # >>> swe_agent: post-tool 生成记录为 turn，区间接在 prompt+completion+suffix 之后
                if self.environments and self.environments[idx] is not None:
                    start = len(prompt_completion_tool_ids[pos]) - len(prompt_ids[idx])
                    _record_turn(
                        self.environments[idx], start, start + len(post_tool_ids[pos]), _classify_turn(calls), None
                    )
                    last_turn_truncated[idx] = post_tool_truncated[pos]
                    if last_turn_truncated[idx]:
                        loop_exit_reasons[idx] = "context_overlong"
                # <<< swe_agent
            # <<< swe_agent
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        # >>> swe_agent: 迭代耗尽仍活跃的样本归因 iteration_cap，并把全部归因写回环境
        for idx in idxs_with_tool:
            loop_exit_reasons.setdefault(idx, "iteration_cap")
        if self.environments:
            for idx, reason in loop_exit_reasons.items():
                environment = self.environments[idx]
                if environment is not None:
                    if (
                        reason == "context_overlong"
                        and last_turn_truncated[idx]
                        and environment.turn_records
                    ):
                        environment.turn_records[-1] = replace(
                            environment.turn_records[-1], truncated=True
                        )
                    environment._record_loop_exit(reason)
        # <<< swe_agent
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images

    def _generation_truncated(self, token_ids) -> bool:
        if not token_ids:
            return False
        eos_and_pad = {
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._tokenizer, "pad_token_id", None),
        }
        eos_and_pad.discard(None)
        return bool(eos_and_pad) and token_ids[-1] not in eos_and_pad
