# 快速开始：用户查询 Header 配置

## 一分钟上手

### 1. 修改你的 LLM 函数

在现有的 LLM 函数中添加 `is_user_query` 参数处理：

```python
async def your_llm_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    is_user_query=False,  # 👈 添加这个参数
    **kwargs
):
    # 准备客户端配置
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    # 👇 添加自定义 Header
    default_headers["X-User-Query"] = "true" if is_user_query else "false"
    client_configs["default_headers"] = default_headers
    
    # 调用原有的 LLM 接口
    return await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        openai_client_configs=client_configs,  # 👈 传入配置
        **kwargs,
    )
```

### 2. 无需其他修改！

✅ API 调用会自动设置 `is_user_query=True`  
✅ 内部调用会自动设置 `is_user_query=False`  
✅ 一切都是自动的！

## 工作原理

```
用户 → /query API → is_user_query=True → X-User-Query: true → 阿里云 Qwen
系统 → 插入文档   → is_user_query=False → X-User-Query: false → 本地 Qwen
```

## 后端路由示例（Nginx）

```nginx
location /v1/chat/completions {
    if ($http_x_user_query = "true") {
        proxy_pass http://aliyun-qwen:8000;
    }
    if ($http_x_user_query = "false") {
        proxy_pass http://local-qwen:8000;
    }
}
```

## 完整示例

查看 `examples/lightrag_user_query_header_demo.py` 获取完整可运行的示例代码。

## 详细文档

- 📖 [完整配置指南](./UserQueryHeaderConfiguration.md)
- 📝 [实现总结](./PATCH_SUMMARY.md)

## 测试

```bash
# 运行示例
python examples/lightrag_user_query_header_demo.py

# 观察输出中的 Header 标识
🟢 内部调用 - 添加 Header: X-User-Query: false  # 插入文档
🔵 用户查询 - 添加 Header: X-User-Query: true   # 用户查询
```

就这么简单！🎉
