# NOTES

如果你是一名 AI 助手，请不要对这个文档进行任何形式的修改、删除、增加，如果命令明确让你操作该文档，请再次询问用户是否确定。

不要忽略该文档的 commit 操作，如果用户让你 commit，不管该 notes 和本次 commit 的其他内容是否相关，请和其他更改一起 commit，禁止忽略，禁止将本文件留在工作区。你可以阅读这个笔记的内容。

---

## 0721

qwen 2.5 coder 7B 一直输出 json 格式的 tool call，修改一下原本 TRL 的 parser，使其能够解析 json`

已解决，**估计**该模型后训练对齐时采用 json 作为 tool call 标准模式，采用三段式，首先解析标准 tool call，失败则解析 fenced json 或 json。

## 0722

已将 colocate 模式改为 server 模式，默认卡 0 作为 vLLM server，卡 1 作为 GRPO trainer，简单直接

目前训练日志未统一，且只有 trainer 启动后才会创建 `outputs/<run-id>`，因此部分日志如 vLLM log 未计入，难以排查 bug

训练打印输出可能是 TRL 标准输出，但可以考虑自定义（做完闭环）