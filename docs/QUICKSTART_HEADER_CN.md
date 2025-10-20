# 快速开始：用户查询 Header 配置（中文版）

## 💡 核心思路

当用户通过 API 查询时用**阿里云 Qwen**，知识库构建等内部操作用**本地 Qwen**。

它们使用**同一个服务器地址**，但通过 **HTTP Header** 区分请求类型。

---

## ⚡ 一分钟配置

### 步骤 1: 修改你的 LLM 函数

在现有 LLM 函数中添加一个参数处理：

```python
async def your_llm_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    is_user_query=False,  # 👈 新增这个参数
    **kwargs
):
    # 获取客户端配置
    client_configs = kwargs.pop("openai_client_configs", {})
    default_headers = client_configs.get("default_headers", {})
    
    # 👇 关键代码：根据调用类型添加不同的 Header
    if is_user_query:
        default_headers["X-User-Query"] = "true"  # 用户查询
    else:
        default_headers["X-User-Query"] = "false"  # 内部调用
    
    client_configs["default_headers"] = default_headers
    
    # 调用原有的 LLM 接口（不用改其他代码）
    return await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),  # 同一个地址！
        openai_client_configs=client_configs,    # 👈 传入配置
        **kwargs,
    )
```

### 步骤 2: 完成！

✅ 就这么简单！其他代码**一行都不用改**！

---

## 🔄 工作流程

```
┌─────────────┐
│  用户查询   │ → API (/query) → is_user_query=True
└─────────────┘                  ↓
                         X-User-Query: true
                                 ↓
                         阿里云 Qwen 服务器

┌─────────────┐
│ 插入文档    │ → 内部调用 → is_user_query=False
└─────────────┘             ↓
                    X-User-Query: false
                            ↓
                    本地 Qwen 服务器
```

---

## 🎯 自动行为

### ✅ 以下场景自动设置 `is_user_query=True`（用户查询）

- `/query` API 端点
- `/query/stream` API 端点
- `/query/data` API 端点

### ✅ 以下场景自动设置 `is_user_query=False`（内部调用）

- 插入文档：`await rag.ainsert(...)`
- 实体提取（知识库构建过程）
- 关系抽取（知识库构建过程）
- 摘要生成（知识库构建过程）
- 程序内直接调用：`await rag.aquery(...)`（不通过 API）

---

## 🔧 后端服务器配置

### 方案 1: Nginx 反向代理

在你的 Nginx 配置中：

```nginx
server {
    listen 8000;
    
    location /v1/chat/completions {
        # 根据 Header 转发到不同的后端
        if ($http_x_user_query = "true") {
            proxy_pass http://aliyun-qwen.com:8000;  # 用户查询 → 阿里云
        }
        
        if ($http_x_user_query = "false") {
            proxy_pass http://127.0.0.1:11434;       # 内部调用 → 本地
        }
    }
}
```

### 方案 2: 自定义网关

如果你有自己的网关服务：

```python
# 在你的网关中检查 Header
def route_llm_request(request):
    is_user_query = request.headers.get("X-User-Query", "false")
    
    if is_user_query == "true":
        # 用户查询 → 阿里云 Qwen
        return forward_to("https://aliyun-qwen-api.com/v1/chat/completions")
    else:
        # 内部调用 → 本地 Qwen
        return forward_to("http://localhost:11434/v1/chat/completions")
```

---

## 📋 完整示例

### 环境变量配置（.env）

```bash
# 大模型服务器地址（同一个地址，后端通过 Header 路由）
LLM_BINDING_HOST=http://your-gateway:8000/v1

# 大模型配置
LLM_MODEL=qwen-plus
OPENAI_API_KEY=your-api-key

# Embedding 配置
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_DIM=1024
```

### Python 代码

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# LLM 函数（带 Header 支持）
async def llm_with_header(prompt, system_prompt=None, 
                         history_messages=[], is_user_query=False, **kwargs):
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

# 初始化 RAG
async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=llm_with_header,
        embedding_func=EmbeddingFunc(...),
    )
    
    await rag.initialize_storages()
    
    # 场景 1: 插入文档（自动用本地 Qwen）
    await rag.ainsert("AI 是计算机科学的一个分支...")
    
    # 场景 2: 用户查询（通过 API 会自动用阿里云 Qwen）
    # 如果直接调用需要手动设置
    result = await rag.aquery("什么是 AI？", 
                              param=QueryParam(mode="mix", is_user_query=True))
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 测试验证

运行官方示例：

```bash
python examples/lightrag_user_query_header_demo.py
```

你会看到类似输出：

```
场景 1: 插入文档（内部调用 - 知识库构建）
--------------------------------------------------------------------------------
[DEBUG] 🟢 内部调用 - 添加 Header: X-User-Query: false
✓ 文档插入完成

场景 2: 用户查询（API 调用 - 模拟用户请求）
--------------------------------------------------------------------------------
[DEBUG] 🔵 用户查询 - 添加 Header: X-User-Query: true
查询结果预览: ...
```

---

## 📖 更多文档

- 📘 [详细配置指南](./UserQueryHeaderConfiguration.md)
- 📙 [实现技术文档](./PATCH_SUMMARY.md)
- 📗 [完整示例代码](../examples/lightrag_user_query_header_demo.py)

---

## ❓ 常见问题

**Q: 我一定要用两个不同的 Qwen 吗？**  
A: 不一定。你可以用于计费统计、监控、负载均衡等任何需要区分请求类型的场景。

**Q: 会影响性能吗？**  
A: 不会。只是在 HTTP Header 中多加了一个字段，开销可以忽略不计。

**Q: 如何调试 Header 是否正确？**  
A: 在 LLM 函数中添加 `print(f"Headers: {default_headers}")`，或使用抓包工具。

**Q: 兼容旧代码吗？**  
A: 完全兼容！不设置 `is_user_query` 时默认为 `False`，旧代码无需修改。

---

## 🎉 开始使用

1. **修改 LLM 函数**（添加 Header 处理）
2. **配置后端路由**（Nginx 或自定义网关）
3. **运行测试**（验证 Header 正确发送）
4. **部署上线**

就这么简单！🚀
