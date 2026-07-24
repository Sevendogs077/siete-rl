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

目前同组内相关性巨大，要么全 1 要么全 0，需要排查组内相关性，目前种子可能没问题

考虑下一步接入多任务，但还是小批量跑，等待全显卡空闲