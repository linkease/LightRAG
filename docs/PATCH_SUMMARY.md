# 用户查询与内部调用区分功能 - 实现总结

## 功能概述

此功能允许 LightRAG 在调用大模型时根据请求类型（用户查询 vs 内部调用）添加不同的 HTTP Header，从而：
- 使用同一个大模型服务器地址
- 通过 Header 区分请求类型
- 后端服务器可根据 Header 路由到不同的模型实例（如阿里云 Qwen vs 本地 Qwen）

## 代码修改清单

### 1. `lightrag/base.py` - 添加 `is_user_query` 字段

**位置**: `QueryParam` 类

```python
is_user_query: bool = False
"""If True, indicates this is a user query (API), otherwise internal (e.g. knowledge base build)."""
```

**作用**: 在 QueryParam 中添加标志位，用于标识是否为用户查询

---

### 2. `lightrag/lightrag.py` - 传递 `is_user_query` 到 global_config

**位置**: `aquery_llm` 方法

```python
global_config = asdict(self)
# 将 is_user_query 标志传递到 global_config 中，供底层 LLM 调用使用
global_config["is_user_query"] = getattr(param, "is_user_query", False)
```

**作用**: 将 `is_user_query` 标志传递给底层操作函数

---

### 3. `lightrag/operate.py` - 传递参数到 LLM 调用

**修改位置 1**: `kg_query` 函数（约第 2391 行）

```python
response = await use_model_func(
    user_query,
    system_prompt=sys_prompt,
    history_messages=query_param.conversation_history,
    enable_cot=True,
    stream=query_param.stream,
    is_user_query=global_config.get("is_user_query", False),  # 新增
)
```

**修改位置 2**: `naive_query` 函数（约第 4180 行）

```python
response = await use_model_func(
    user_query,
    system_prompt=sys_prompt,
    history_messages=query_param.conversation_history,
    enable_cot=True,
    stream=query_param.stream,
    is_user_query=global_config.get("is_user_query", False),  # 新增
)
```

**作用**: 在调用 LLM 函数时传递 `is_user_query` 参数

---

### 4. `lightrag/api/routers/query_routes.py` - API 路由自动设置标志

**修改位置 1**: `/query` 端点（约第 323 行）

```python
param = request.to_query_params(False)
param.stream = False
param.is_user_query = True  # 标记为用户query
```

**修改位置 2**: `/query/stream` 端点（约第 486 行）

```python
stream_mode = request.stream if request.stream is not None else True
param = request.to_query_params(stream_mode)
param.is_user_query = True  # 标记为用户query
```

**作用**: API 端点自动将所有查询标记为用户查询

---

## 使用方法

### 方法 1: 自定义 LLM 函数（推荐）

创建包装函数来根据 `is_user_query` 添加自定义 Header：

```python
async def llm_model_func_with_header(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    is_user_query=False,  # 接收参数
    **kwargs
):
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    # 根据 is_user_query 添加不同的 Header
    if is_user_query:
        default_headers["X-User-Query"] = "true"
    else:
        default_headers["X-User-Query"] = "false"
    
    client_configs["default_headers"] = default_headers
    
    return await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        openai_client_configs=client_configs,
        **kwargs,
    )
```

### 方法 2: 初始化 LightRAG

```python
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_model_func_with_header,
    embedding_func=EmbeddingFunc(...),
)
```

### 方法 3: 自动行为

- **API 调用**: `/query` 和 `/query/stream` 端点自动设置 `is_user_query=True`
- **内部调用**: 文档插入、实体提取等内部操作自动使用 `is_user_query=False`

## 示例文件

### 完整示例
- `examples/lightrag_user_query_header_demo.py` - 完整的演示代码

### 详细文档
- `docs/UserQueryHeaderConfiguration.md` - 完整的配置指南和使用说明

## 测试方法

运行示例：

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export LLM_BINDING_HOST="https://your-llm-server.com/v1"
export LLM_MODEL="gpt-4o-mini"

# 运行示例
python examples/lightrag_user_query_header_demo.py
```

预期输出：
- 插入文档时显示：`🟢 内部调用 - 添加 Header: X-User-Query: false`
- 用户查询时显示：`🔵 用户查询 - 添加 Header: X-User-Query: true`

## 后端服务器配置示例

### Nginx 路由配置

```nginx
location /v1/chat/completions {
    # 根据 X-User-Query Header 路由到不同后端
    if ($http_x_user_query = "true") {
        proxy_pass http://aliyun-qwen-backend:8000;
    }
    
    if ($http_x_user_query = "false") {
        proxy_pass http://local-qwen-backend:8000;
    }
}
```

## 向后兼容性

- ✅ 不设置 `is_user_query` 时默认为 `False`
- ✅ 旧代码无需修改即可继续工作
- ✅ 新功能完全可选，不影响现有功能

## 相关文件

修改的文件：
1. `lightrag/base.py` - QueryParam 类
2. `lightrag/lightrag.py` - aquery_llm 方法
3. `lightrag/operate.py` - kg_query 和 naive_query 函数
4. `lightrag/api/routers/query_routes.py` - API 路由

新增的文件：
1. `examples/lightrag_user_query_header_demo.py` - 示例代码
2. `docs/UserQueryHeaderConfiguration.md` - 详细文档
3. `docs/PATCH_SUMMARY.md` - 本文件

## 常见问题

**Q: 如何验证 Header 是否正确发送？**  
A: 在自定义 LLM 函数中添加日志，或使用网络抓包工具（如 Wireshark）查看 HTTP 请求。

**Q: 缓存的响应会触发 Header 吗？**  
A: 不会。缓存的响应直接返回，不会重新调用 LLM，因此不会发送新的 Header。

**Q: 可以添加其他自定义 Header 吗？**  
A: 可以。在自定义 LLM 函数中根据需要添加任意 Header。

**Q: 流式响应支持吗？**  
A: 支持。Header 在建立连接时发送，流式和非流式响应都可以使用。

## 总结

此功能通过简单的配置实现了用户查询和内部调用的区分，无需维护两个独立的 LLM 配置。通过 HTTP Header 传递调用类型信息，后端服务器可以灵活地进行路由、计费或监控。
