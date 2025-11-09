# 变更摘要 - X-Openai-Key Header 功能

## 功能说明

在 LightRAG API 的查询接口中添加了对 `X-Openai-Key` HTTP Header 的支持。当请求包含此 Header 时，系统会优先使用 Header 中提供的 API Key 而不是默认的环境变量配置。这使得多租户应用可以为不同用户使用独立的 API Key，以便分别追踪每个用户的 token 使用量。

## 主要变更

### 1. QueryParam 数据模型扩展 (`lightrag/base.py`)

```python
@dataclass
class QueryParam:
    # ... 现有字段 ...
    
    user_api_key: str | None = None
    """User-specific API key for LLM requests.
    When provided, this API key will be used instead of the default LLM_BINDING_API_KEY
    for this specific query. This allows per-user API key tracking for token usage billing.
    """
```

### 2. API 端点更新 (`lightrag/api/routers/query_routes.py`)

#### 导入 Header 依赖
```python
from fastapi import APIRouter, Depends, Header, HTTPException
```

#### 三个查询端点都添加了 Header 参数

**POST /query**
```python
async def query_text(
    request: QueryRequest,
    x_openai_key: Optional[str] = Header(None, alias="X-Openai-Key")
):
    # ... 处理逻辑 ...
    if x_openai_key:
        param.user_api_key = x_openai_key
        logging.info(f"Using user-specific API key from X-Openai-Key header")
```

**POST /query/stream**
```python
async def query_text_stream(
    request: QueryRequest,
    x_openai_key: Optional[str] = Header(None, alias="X-Openai-Key")
):
    # ... 处理逻辑 ...
    if x_openai_key:
        param.user_api_key = x_openai_key
        logging.info(f"Using user-specific API key from X-Openai-Key header")
```

**POST /query/data**
```python
async def query_data(
    request: QueryRequest,
    x_openai_key: Optional[str] = Header(None, alias="X-Openai-Key")
):
    # ... 处理逻辑 ...
    if x_openai_key:
        param.user_api_key = x_openai_key
        logging.info(f"Using user-specific API key from X-Openai-Key header")
```

### 3. LightRAG 核心处理 (`lightrag/lightrag.py`)

#### aquery_llm 方法
```python
async def aquery_llm(self, query: str, param: QueryParam = QueryParam(), ...):
    global_config = asdict(self)
    global_config["llm_query_model"] = getattr(param, "llm_query_model", None)
    
    # 新增：传递用户 API Key
    global_config["user_api_key"] = getattr(param, "user_api_key", None)
```

#### aquery_data 方法
```python
async def aquery_data(self, query: str, param: QueryParam = QueryParam()):
    global_config = asdict(self)
    
    # 新增：传递用户 API Key
    global_config["user_api_key"] = getattr(param, "user_api_key", None)
```

### 4. OpenAI LLM 绑定 (`lightrag/llm/openai.py`)

```python
async def openai_complete_if_cache(..., **kwargs):
    # ... 现有代码 ...
    
    # 检查是否指定了查询模型
    llm_query_model = kwargs.pop("llm_query_model", None)
    _model = model
    if llm_query_model:
        _model = llm_query_model
        logger.debug(f"Using query model for user query: {_model}")
    
    # 新增：API Key 处理 - 优先使用用户特定的 API key
    user_api_key = kwargs.pop("user_api_key", None)
    
    _api_key = user_api_key or api_key
    if not _api_key:
        _api_key = os.getenv("OPENAI_API_KEY")
    
    if user_api_key:
        logger.debug(f"Using user-specific API key for this request")
```

### 5. Azure OpenAI LLM 绑定 (`lightrag/llm/azure_openai.py`)

```python
async def azure_openai_complete_if_cache(..., **kwargs):
    # ... 现有代码 ...
    
    # 新增：API Key 处理 - 优先使用用户特定的 API key
    user_api_key = kwargs.pop("user_api_key", None)
    
    api_key = (
        user_api_key or
        api_key or 
        os.getenv("AZURE_OPENAI_API_KEY") or 
        os.getenv("LLM_BINDING_API_KEY")
    )
    
    if user_api_key:
        logger.debug(f"Using user-specific API key for this request")
    
    # 新增：支持查询模型切换（之前缺失）
    llm_query_model = kwargs.pop("llm_query_model", None)
    if llm_query_model:
        deployment = llm_query_model
        logger.debug(f"Using query model for user query: {deployment}")
```

## 使用示例

### cURL
```bash
curl -X POST http://localhost:8020/query \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-user-specific-key-12345" \
  -d '{
    "query": "What is machine learning?",
    "mode": "mix"
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8020/query",
    headers={
        "Content-Type": "application/json",
        "X-Openai-Key": "sk-user-specific-key-12345"
    },
    json={
        "query": "What is machine learning?",
        "mode": "mix"
    }
)
```

## API Key 优先级

系统按以下优先级使用 API Key：

1. **X-Openai-Key Header**（最高优先级）
2. **api_key 函数参数**
3. **OPENAI_API_KEY 环境变量**（回退选项）

## 验证方式

在服务器日志中查找以下消息：

```
INFO: Using user-specific API key from X-Openai-Key header
DEBUG: Using user-specific API key for this request
```

## 文件清单

### 修改的文件
1. `lightrag/base.py` - 添加 `user_api_key` 字段
2. `lightrag/api/routers/query_routes.py` - 添加 Header 参数支持
3. `lightrag/lightrag.py` - 传递 `user_api_key` 到 `global_config`
4. `lightrag/llm/openai.py` - 优先使用 `user_api_key`
5. `lightrag/llm/azure_openai.py` - 添加 `user_api_key` 支持

### 新增的文件
1. `docs/X-Openai-Key-Header.md` - 详细使用文档
2. `docs/IMPLEMENTATION_X_OPENAI_KEY.md` - 实现说明文档
3. `test_user_api_key.py` - 功能测试脚本
4. `docs/CHANGE_SUMMARY_CN.md` - 本文件（中文变更摘要）

## 兼容性

- **完全向后兼容**：如果不提供 `X-Openai-Key` Header，系统会使用默认的 API Key
- **无破坏性变更**：现有代码无需修改即可继续工作
- **透明升级**：新功能对现有用户透明，仅在需要时启用

## 测试建议

运行测试脚本验证功能：

```bash
python test_user_api_key.py
```

或手动测试：

1. 启动 LightRAG API 服务
2. 使用 curl 或其他工具发送带 `X-Openai-Key` Header 的请求
3. 检查服务器日志，确认使用了自定义 API Key
4. 验证查询功能正常工作

## 应用场景

1. **SaaS 应用**：为每个租户使用独立的 API Key
2. **分层计费**：不同用户等级使用不同的 API Key
3. **成本跟踪**：精确追踪每个用户的 token 消耗
4. **配额管理**：为不同用户设置不同的 API 配额
