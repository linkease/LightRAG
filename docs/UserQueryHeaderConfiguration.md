# 用户查询与内部调用 Header 配置指南

## 概述

LightRAG 现在支持根据调用类型（用户查询 vs 内部调用）自动添加不同的 HTTP Header。这对于以下场景非常有用：

- **相同大模型服务器地址**，但需要根据调用类型区分请求
- **负载均衡或路由**：后端服务器根据 Header 路由到不同的模型实例
- **计费或监控**：区分用户查询和系统内部调用的资源使用

## 功能特性

### 自动 Header 注入

- **用户查询（API 调用）**：当用户通过 `/query` 或 `/query/stream` 端点查询时，`is_user_query=True`
- **内部调用（知识库构建）**：实体提取、关系构建、摘要等内部操作时，`is_user_query=False`

### Header 示例

```http
# 用户查询请求
X-User-Query: true

# 内部调用请求
X-User-Query: false
```

## 实现步骤

### 步骤 1：修改 LLM 函数以支持自定义 Header

创建一个包装函数来根据 `is_user_query` 参数添加自定义 Header：

```python
async def llm_model_func_with_header(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False,
    is_user_query=False,  # 新增参数
    **kwargs
) -> str:
    """
    自定义 LLM 函数，根据 is_user_query 标志添加不同的 Header
    """
    # 准备 OpenAI 客户端配置
    client_configs = kwargs.pop("openai_client_configs", {})
    
    # 设置默认 headers
    default_headers = client_configs.get("default_headers", {})
    
    # 根据 is_user_query 添加自定义 Header
    if is_user_query:
        default_headers["X-User-Query"] = "true"
        # 可以添加更多用户查询相关的 Header
        # default_headers["X-Request-Priority"] = "high"
        # default_headers["X-User-Type"] = "api"
    else:
        default_headers["X-User-Query"] = "false"
        # 可以添加更多内部调用相关的 Header
        # default_headers["X-Request-Priority"] = "normal"
        # default_headers["X-User-Type"] = "internal"
    
    client_configs["default_headers"] = default_headers
    
    # 调用原始的 openai_complete_if_cache
    return await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        openai_client_configs=client_configs,
        **kwargs,
    )
```

### 步骤 2：初始化 LightRAG

使用自定义的 LLM 函数初始化 LightRAG：

```python
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_model_func_with_header,  # 使用自定义函数
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=your_embedding_function,
    ),
)

await rag.initialize_storages()
```

### 步骤 3：API 调用自动设置标志

API 路由（`/query` 和 `/query/stream`）会自动设置 `is_user_query=True`：

```python
# 在 query_routes.py 中已自动实现
param = request.to_query_params(False)
param.is_user_query = True  # API 调用自动标记为用户查询
```

### 步骤 4：内部调用自动使用默认值

知识库构建等内部操作不设置此标志，默认为 `False`：

```python
# 插入文档（内部调用）
await rag.ainsert("文档内容...")
# is_user_query 默认为 False

# 直接调用（非 API）
result = await rag.aquery("查询内容", param=QueryParam(mode="mix"))
# is_user_query 默认为 False
```

## 完整示例

### 示例 1：基本使用

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

load_dotenv()

async def llm_model_func_with_header(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    is_user_query=False,
    **kwargs
):
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    # 添加自定义 Header
    default_headers["X-User-Query"] = "true" if is_user_query else "false"
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

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=llm_model_func_with_header,
        embedding_func=EmbeddingFunc(...),
    )
    
    await rag.initialize_storages()
    
    # 1. 插入文档（内部调用，X-User-Query: false）
    await rag.ainsert("人工智能是计算机科学的一个分支...")
    
    # 2. 用户查询（需要手动设置 is_user_query=True）
    param = QueryParam(mode="mix", is_user_query=True)
    result = await rag.aquery("什么是人工智能？", param=param)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 2：通过 API 调用（自动设置）

```python
# API 调用会自动设置 is_user_query=True
# 客户端代码
import requests

response = requests.post(
    "http://localhost:8020/query",
    json={
        "query": "什么是机器学习？",
        "mode": "mix"
    }
)

# 服务器端会自动添加 X-User-Query: true Header
```

## 高级配置

### 添加更多自定义 Header

```python
async def llm_model_func_with_advanced_headers(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    is_user_query=False,
    **kwargs
):
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    if is_user_query:
        # 用户查询使用阿里云 Qwen
        default_headers.update({
            "X-User-Query": "true",
            "X-Model-Provider": "aliyun",
            "X-Priority": "high",
            "X-Request-Type": "user-api",
        })
    else:
        # 内部调用使用本地 Qwen
        default_headers.update({
            "X-User-Query": "false",
            "X-Model-Provider": "local",
            "X-Priority": "normal",
            "X-Request-Type": "internal",
        })
    
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

### 服务器端根据 Header 路由

如果你使用 Nginx 或其他反向代理，可以根据 Header 路由到不同的后端：

```nginx
# Nginx 配置示例
location /v1/chat/completions {
    # 根据 X-User-Query Header 路由
    if ($http_x_user_query = "true") {
        proxy_pass http://aliyun-qwen-backend;
    }
    
    if ($http_x_user_query = "false") {
        proxy_pass http://local-qwen-backend;
    }
}
```

## 调试和验证

### 启用调试日志

在 LLM 函数中添加日志：

```python
import logging

async def llm_model_func_with_header(
    prompt, 
    is_user_query=False,
    **kwargs
):
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    default_headers["X-User-Query"] = "true" if is_user_query else "false"
    
    # 调试日志
    logging.info(f"[LLM] is_user_query={is_user_query}")
    logging.info(f"[LLM] Headers: {default_headers}")
    
    client_configs["default_headers"] = default_headers
    
    return await openai_complete_if_cache(...)
```

### 测试不同场景

```python
# 测试脚本
async def test_headers():
    rag = LightRAG(
        working_dir="./test_storage",
        llm_model_func=llm_model_func_with_header,
        embedding_func=EmbeddingFunc(...),
    )
    
    await rag.initialize_storages()
    
    print("=" * 60)
    print("测试 1: 插入文档（内部调用）")
    print("=" * 60)
    await rag.ainsert("测试文档内容")
    # 预期: X-User-Query: false
    
    print("\n" + "=" * 60)
    print("测试 2: 用户查询（API 模拟）")
    print("=" * 60)
    param = QueryParam(mode="mix", is_user_query=True)
    result = await rag.aquery("测试查询", param=param)
    # 预期: X-User-Query: true
    
    print("\n" + "=" * 60)
    print("测试 3: 普通查询（非 API）")
    print("=" * 60)
    result = await rag.aquery("测试查询", param=QueryParam(mode="naive"))
    # 预期: X-User-Query: false
```

## 注意事项

1. **向后兼容**：如果不设置 `is_user_query`，默认为 `False`，保持原有行为
2. **缓存影响**：缓存的响应不会重新调用 LLM，因此不会触发 Header 逻辑
3. **流式响应**：Header 在建立连接时发送，流式和非流式响应都支持
4. **API 自动设置**：通过 API 路由的查询会自动设置 `is_user_query=True`

## 总结

通过这个功能，你可以：

✅ 使用同一个大模型服务器地址  
✅ 根据调用类型添加不同的 HTTP Header  
✅ 用户查询自动添加 `X-User-Query: true`  
✅ 内部调用自动添加 `X-User-Query: false`  
✅ 后端服务器可根据 Header 进行路由、计费或监控  

无需维护两个独立的 LLM 配置，简化了部署和配置管理。
