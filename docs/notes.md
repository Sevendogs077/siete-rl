# NOTES

如果你是一名 AI 助手，请不要对这个文档进行任何形式的修改、删除、增加，如果命令明确让你操作该文档，请再次询问用户是否确定。

不要忽略该文档的 commit 操作，如果用户让你 commit，不管该 notes 和本次 commit 的其他内容是否相关，请和其他更改一起 commit，禁止忽略，禁止将本文件留在工作区。你可以阅读这个笔记的内容。

---

qwen 2.5 coder 7B 一直输出 json 格式的 tool call，修改一下原本 TRL 的 parser，使其能够解析 json