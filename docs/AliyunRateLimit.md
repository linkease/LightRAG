请分析当前代码，是否已经实现：
如果命中了 RateLimitError 是否会 sleep 一段时间然后重试？

当调用阿里云 qwen 云端大模型的时候，会出现下面的情况：

调整调用频率：触发Requests rate limit exceeded或You exceeded your current requests list时，降低调用频率。

而且测试 HTTP 状态码是 504。这种情况也是 RateLimitError

而且这种情况必须等待 60s 到 120s 直接。请帮修改代码，适应上面这种情况。

openai_async_client.embeddings.create

这里不是 LLM 不需要修改为新的逻辑，保持之前的逻辑

请提供一些必要的日志，以确认代码是否真正运行成功。并且保证代码能编译成功。

遇到这样的错误，但是没有遇到 504 然后 sleep 的日志：

chunk-0da1799d2dbc2ce32ccb24d4241381d7:InternalServerError:Errorcode:504(0riginalexceptioncould not be reconstructed:APIStatusError.init()missing 2required keyword-only arguments:'responseand 'body')