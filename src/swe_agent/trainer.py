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
5. 循环出口为迭代耗尽的样本归因 ``iteration_cap``，并把全部归因写回环境。

TRL 升级时必须对照 `trl.trainer.grpo_trainer.GRPOTrainer._tool_call_loop`
人工同步本方法（镜像一致性由测试守护）。
"""

from __future__ import annotations

import asyncio

from trl import GRPOTrainer
from trl.chat_template_utils import parse_response

from swe_agent.models import LoopExit


_PARSE_ERROR_SENTINEL = "swe_agent_parse_error"
"""哨兵 tool call：parse_error 样本留在循环内接受格式反馈，不进入真实工具执行。"""

_FORMAT_FEEDBACK_TEMPLATE = (
    "ERROR: the previous assistant response could not be parsed as one complete "
    "OpenHands function call ({reason}). Use <function=...> with valid parameters."
)
_PLAIN_MESSAGE_SENTINEL = "swe_agent_plain_message"


class SWEGRPOTrainer(GRPOTrainer):
    """在官方 GRPOTrainer 上加入环境信号终止；其余行为与 TRL 完全一致。"""

    def __init__(self, *args, max_consecutive_protocol_errors: int, **kwargs) -> None:
        if max_consecutive_protocol_errors < 1:
            raise ValueError("max_consecutive_protocol_errors must be positive")
        self.max_consecutive_protocol_errors = max_consecutive_protocol_errors
        super().__init__(*args, **kwargs)

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
        # >>> swe_agent: 仅由项目终止语义写回循环退出原因
        loop_exit_reasons: dict[int, LoopExit] = {}
        # <<< swe_agent
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            # Call the tools, and build the new prompt for generation
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
                    from swe_agent.tool_protocol import FIXED_FAKE_USER
                    prompt_completion_tool.append({"role": "user", "content": FIXED_FAKE_USER})
                    completions[idx_with_tool].append({"role": "user", "content": FIXED_FAKE_USER})
                    format_error_counts[idx_with_tool] = 0
                    continue
                # <<< swe_agent
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append((name, sync_tool_dict[name](**function["arguments"])))
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    # OpenHands observation 是普通 user turn，而不是 Qwen role=tool。
                    from swe_agent.tool_protocol import render_observation
                    tool_message = {"role": "user", "content": render_observation(name, str(content), error=isinstance(result, dict) and "error" in result)}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

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
            exit_flags = [
                terminated or breaker
                for terminated, breaker in zip(terminated_flags, breaker_flags, strict=True)
            ]
            if any(exit_flags):
                for idx in range(len(idxs_with_tool)):
                    if exit_flags[idx]:
                        idx_with_tool = idxs_with_tool[idx]
                        if terminated_flags[idx]:
                            del completions[idx_with_tool][completions_len_before[idx] :]
                            del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                            del prompts[idx_with_tool][prompts_len_before[idx] :]
                        elif breaker_flags[idx]:
                            loop_exit_reasons[idx_with_tool] = "format_exhausted"
                idxs_with_tool = [
                    idx for idx, flag in zip(idxs_with_tool, exit_flags, strict=True) if not flag
                ]
                if not idxs_with_tool:
                    break
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
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
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
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

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
                    if isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    loop_multimodal_fields[k] = selected
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(
                prompt_completion_tool_ids, loop_images, loop_multimodal_fields
            )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
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
            for idx, completion in zip(idxs_with_tool, post_tool_completions, strict=True):
                if not completion:
                    # 生成被预算截断为空：长度耗尽而非格式错误，不进格式恢复
                    loop_exit_reasons[idx] = "context_overlong"
                    tool_calls.append(None)
                    continue
                calls = self._next_action(completion)
                if calls != [_PARSE_ERROR_SENTINEL]:
                    format_error_counts[idx] = 0
                tool_calls.append(calls)
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
                    environment._record_loop_exit(reason)
        # <<< swe_agent
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images
