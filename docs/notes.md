# NOTES

如果你是一名 AI 助手，请不要对这个文档进行任何形式的修改、删除、增加，如果命令明确让你操作该文档，请再次询问用户是否确定。

不要忽略该文档的 commit 操作，如果用户让你 commit，不管该 notes 和本次 commit 的其他内容是否相关，请和其他更改一起 commit，禁止忽略，禁止将本文件留在工作区。你可以阅读这个笔记的内容。

---

## 6c0d01

~~qwen 2.5 coder 7B 一直输出 json 格式的 tool call，修改一下原本 TRL 的 parser，使其能够解析 json~~，已解决，**估计**该模型后训练对齐时采用 json 作为 tool call 标准模式，采用三段式，首先解析标准 tool call，失败则解析 fenced json 或 json。

## f23f3b

已将 colocate 模式改为 server 模式，默认卡 0 作为 vLLM server，卡 1 作为 GRPO trainer，简单直接

~~目前训练日志未统一，且只有 trainer 启动后才会创建 `outputs/<run-id>`，因此部分日志如 vLLM log 未计入，难以排查 bug~~

训练打印输出可能是 TRL 标准输出，但可以考虑自定义（做完闭环）

### ed4fdd

~~如果 termination != "submitted"，将直接给 0 分且跳过验证器，但此时模型并不会终止，仍会输出若干步的信息，且这部分多余信息甚至也会进入训练，导致训练污染，因此，需要完全改变环境抛异常和中止的方式，更干净~~

~~模型输出正常的行为，只是文件不存在，termination 却被永久标为 invalid_tool_call 且永不清除~~

~~environment.py:307-309 不可达~~
   
~~要增加连续格式错误熔断机制 max_consecutive_format_errors（mini_swe_agent）~~

~~docker 名称使用 swe_agent；只有仓库名用 `-`~~

~~配置项 submit_requires_final_response（config.py:83），疑似为残留~~
                                                                
~~git diff 不包含 edit_file operation="create" 创建的新文件~~，其实我觉得这个没啥用，gold patch 应该都不涉及新文件

reward 目前全组都是 0，如果改了上述环境问题后还是这样，考虑后面改成 DAPO

temperature=1.0 可能不合适，需要调整


### 059102

~~目前同组内相关性巨大，要么全 1 要么全 0，需要排查组内相关性，目前种子可能没问题~~，已排查，单次成功率太低

~~考虑下一步接入多任务，但还是小批量跑，等待全显卡空闲~~

### d2da3a

接入 OpenHands SFT model 后，单次 Tool Call 调用 + Observation 就接近 5000 Token，Context 严重不足，但提高 Context 又会造成 OOM，具体如下解决：

真正的显存大头：activation，尤其是 logits 尖峰。Qwen2-7B 词表 ~152k，32k token 序列的 logits bf16 就 ~9.3GB，TRL 算 per-token logprob 时 fp32 log_softmax 再翻一倍（为了防止数值误差，通常用 FP32 做 Softmax）；加上 backward 中间量，这才是 trainer 卡逼近 42GB 上限的原因。分片参数对这一点毫无帮助

解决：Liger Kernel，消除 Loss 计算时的尖峰

原来：先保存巨大 logits，再计算 loss
```
隐藏状态
  ↓
lm_head
  ↓
生成完整 logits：[8000, 152000]
  ↓
完整 log_softmax
  ↓
从每一行取出实际生成 token 的 logprob
  ↓
计算 GRPO loss
```

现在：边计算 logits，边计算 loss，不保存完整 logits
```
隐藏状态
  ↓
lm_head + log_softmax + 取目标 token + GRPO loss
  ↓
直接得到 loss
```


### 1a5c98

完成 Liger Kernel fused GRPO loss，目前可跑：16384 completion length