# 模型区分设计：用户查询 vs 知识库构建

## 概述

本文档说明了如何通过使用不同的模型来区分用户查询和内部知识库构建操作，取代之前使用不同 API Key 的方案。

## 设计理念

### 旧方案（已废弃）
- 使用 `OPENAI_API_KEY` 用于知识库构建
- 使用 `OPENAI_QUERY_KEY` 用于用户查询
- 问题：需要维护两个不同的 API Key，配置复杂

### 新方案（推荐）
- 使用 `LLM_MODEL` 用于知识库构建（如：`qwen-local`）
- 使用 `LLM_QUERY_MODEL` 用于用户查询（如：`gpt-4o`）
- 优势：通过模型名称区分，后端可根据模型路由到不同的实例

## 使用场景

1. **成本优化**
   - 用户查询：使用云端高性能大模型（如 GPT-4）
   - 知识库构建：使用本地经济型模型（如本地 Qwen）

2. **性能优化**
   - 用户查询：使用针对对话优化的模型
   - 知识库构建：使用针对批处理优化的模型

3. **后端路由**
   - 相同的 API 地址
   - 根据模型名称路由到不同的后端实例
   - 例如：`qwen-local` 路由到本地，`gpt-4o` 路由到 OpenAI

4. **灵活部署**
   - 单一 API 端点
   - 多个后端实例
   - 通过模型名称自动路由

## 配置示例

### .env 文件配置

```bash
# API 配置
OPENAI_API_KEY=your-api-key
LLM_BINDING_HOST=http://localhost:8000/v1

# 模型配置
LLM_MODEL=qwen-local              # 用于知识库构建的模型
LLM_QUERY_MODEL=gpt-4o            # 用于用户查询的模型（可选）

# Embedding 配置
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_DIM=1024
```

### 代码示例

```python
import os
from lightrag import LightRAG, QueryParam

# 初始化 LightRAG（会自动使用 LLM_MODEL）
rag = LightRAG(
    working_dir="./demoDir",
    llm_model_func=your_llm_func,
    embedding_func=your_embed_func,
)

# 场景 1: 知识库构建（自动使用 LLM_MODEL）
await rag.ainsert("这是要插入的文档内容...")

# 场景 2: 用户查询（使用 LLM_QUERY_MODEL）
query_model = os.getenv("LLM_QUERY_MODEL")
param = QueryParam(
    mode="hybrid",
    llm_query_model=query_model,  # 设置查询模型（设置此参数即表示这是用户查询）
)
result = await rag.aquery("用户的问题？", param=param)

# 场景 3: 内部查询（使用 LLM_MODEL）
param = QueryParam(mode="local")  # 不设置 llm_query_model，使用默认模型
result = await rag.aquery("内部查询", param=param)
```

## 实现细节

### 1. QueryParam 字段

在 `lightrag/base.py` 中：

```python
@dataclass
class QueryParam:
    # ... 其他字段 ...
    
    llm_query_model: str | None = None
    """用户查询时使用的模型名称（可选）
    如果设置了此字段，表示这是用户查询，将使用指定的模型而不是默认模型
    """
```

### 2. LightRAG 传递模型配置

在 `lightrag/lightrag.py` 的 `aquery_llm` 方法中：

```python
async def aquery_llm(self, query: str, param: QueryParam = QueryParam(), ...):
    global_config = asdict(self)
    # 如果设置了 llm_query_model，说明这是用户查询
    global_config["llm_query_model"] = getattr(param, "llm_query_model", None)
    # ...
```

### 3. operate.py 传递参数

在 `lightrag/operate.py` 中，所有 LLM 调用处添加：

```python
response = await use_model_func(
    user_query,
    system_prompt=sys_prompt,
    # ... 其他参数 ...
    llm_query_model=global_config.get("llm_query_model", None),
)
```

### 4. OpenAI LLM 实现

在 `lightrag/llm/openai.py` 的 `openai_complete_if_cache` 中：

```python
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    # ... 其他参数 ...
    **kwargs
):
    # 提取查询模型参数
    llm_query_model = kwargs.pop("llm_query_model", None)
    
    # 如果指定了查询模型，使用查询模型；否则使用默认模型
    _model = model
    if llm_query_model:
        _model = llm_query_model
        logger.debug(f"Using query model for user query: {_model}")
    
    # 使用 _model 进行 API 调用
    response = await openai_async_client.chat.completions.create(
        model=_model,
        messages=messages,
        **kwargs
    )
```

## API 集成

### 查询路由自动配置

API 路由（`lightrag/api/routers/query.py`）应该自动设置：

```python
from lightrag.base import QueryParam

@router.post("/query")
async def query_endpoint(request: QueryRequest):
    # 从环境变量读取查询模型
    query_model = os.getenv("LLM_QUERY_MODEL")
    
    param = QueryParam(
        mode=request.mode,
        llm_query_model=query_model,  # 设置查询模型（API 调用都是用户查询）
        # ... 其他参数 ...
    )
    
    result = await rag.aquery(request.query, param=param)
    return result
```

## 后端路由示例

### 方案 1: 反向代理路由

使用 Nginx 或其他反向代理，根据模型名称路由：

```nginx
location /v1/chat/completions {
    # 根据请求体中的 model 字段路由
    if ($request_body ~* "\"model\":\"qwen-local\"") {
        proxy_pass http://localhost:11434;  # 本地 Ollama
    }
    if ($request_body ~* "\"model\":\"gpt-4o\"") {
        proxy_pass https://api.openai.com;  # OpenAI API
    }
}
```

### 方案 2: 自定义路由服务

编写一个简单的路由服务：

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

MODEL_ROUTES = {
    "qwen-local": "http://localhost:11434/v1/chat/completions",
    "gpt-4o": "https://api.openai.com/v1/chat/completions",
}

@app.post("/v1/chat/completions")
async def route_model(request: Request):
    data = await request.json()
    model = data.get("model", "")
    
    # 根据模型路由
    target_url = MODEL_ROUTES.get(model)
    if not target_url:
        return {"error": "Unknown model"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(target_url, json=data)
        return response.json()
```

## 迁移指南

### 从旧方案迁移

如果你之前使用 `OPENAI_QUERY_KEY`，请按以下步骤迁移：

1. **更新 .env 文件**
   ```bash
   # 删除
   # OPENAI_QUERY_KEY=xxx
   
   # 添加
   LLM_MODEL=your-default-model
   LLM_QUERY_MODEL=your-query-model  # 可选
   ```

2. **更新代码**
   - 移除所有 `OPENAI_QUERY_KEY` 的引用
   - 使用新的 `llm_query_model` 参数

3. **测试**
   - 运行 `examples/lightrag_user_query_header_demo.py` 验证配置

## 常见问题

### Q: 如果不设置 LLM_QUERY_MODEL 会怎样？
A: 系统会默认使用 LLM_MODEL，即用户查询和知识库构建使用相同的模型。

### Q: 可以在运行时动态切换模型吗？
A: 可以，通过 QueryParam 的 `llm_query_model` 参数在每次查询时指定。

### Q: 这个方案支持其他 LLM 提供商吗？
A: 是的，只要后端实现了 OpenAI 兼容的 API 格式，就可以通过模型名称路由。

### Q: 如何监控不同模型的使用情况？
A: 可以在日志中记录使用的模型名称，或在后端路由服务中添加监控。

## 相关文件

- `lightrag/base.py` - QueryParam 定义
- `lightrag/lightrag.py` - 核心查询逻辑
- `lightrag/operate.py` - 操作函数
- `lightrag/llm/openai.py` - OpenAI LLM 实现
- `examples/lightrag_user_query_header_demo.py` - 完整示例

## 总结

新方案通过模型名称而不是 API Key 来区分用户查询和知识库构建，具有以下优势：

1. ✅ 配置更简单，只需一个 API Key
2. ✅ 更灵活，可以在运行时动态指定模型
3. ✅ 更容易实现后端路由和负载均衡
4. ✅ 支持更多使用场景（成本优化、性能优化等）
5. ✅ 向后兼容，不影响现有功能
