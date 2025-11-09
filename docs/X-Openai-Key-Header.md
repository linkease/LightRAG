# X-Openai-Key Header 功能文档

## 概述

LightRAG API 现在支持通过 HTTP Header 传递用户特定的 OpenAI API Key，这使得多租户应用可以为不同用户使用独立的 API Key，从而实现：

- **独立计费**: 每个用户使用自己的 OpenAI API Key，token 使用量分别计费
- **用户隔离**: 不同用户的 API 配额和限制相互独立
- **灵活管理**: 可以为不同用户或访问层级配置不同的 API Key

## 使用方法

### HTTP Header 参数

在请求中添加 `X-Openai-Key` Header：

```http
POST /query HTTP/1.1
Host: your-lightrag-api.com
Content-Type: application/json
X-Openai-Key: sk-user-specific-key-12345

{
  "query": "What is machine learning?",
  "mode": "mix"
}
```

### 支持的端点

以下端点都支持 `X-Openai-Key` Header：

1. **POST /query** - 标准查询（非流式）
2. **POST /query/stream** - 流式查询
3. **POST /query/data** - 数据检索（仅返回结构化数据）

### API Key 优先级

API Key 的使用遵循以下优先级（从高到低）：

1. **X-Openai-Key Header**: 请求中的 Header（优先级最高）
2. **api_key 参数**: 传递给 LLM 函数的参数
3. **环境变量**: `OPENAI_API_KEY` 或 `LLM_BINDING_API_KEY`

## 代码示例

### Python 示例

```python
import requests
import json

# API 配置
API_URL = "http://localhost:8020"
USER_API_KEY = "sk-proj-user1-abc123..."

# 标准查询
def query_with_user_key(query_text, user_api_key):
    url = f"{API_URL}/query"
    headers = {
        "Content-Type": "application/json",
        "X-Openai-Key": user_api_key  # 用户特定的 API Key
    }
    payload = {
        "query": query_text,
        "mode": "mix",
        "include_references": True
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# 使用示例
result = query_with_user_key(
    "Explain neural networks",
    USER_API_KEY
)
print(result["response"])

# 流式查询
def stream_query_with_user_key(query_text, user_api_key):
    url = f"{API_URL}/query/stream"
    headers = {
        "Content-Type": "application/json",
        "X-Openai-Key": user_api_key
    }
    payload = {
        "query": query_text,
        "mode": "mix",
        "stream": True
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                print(data["response"], end="", flush=True)
            if "error" in data:
                print(f"Error: {data['error']}")

# 使用示例
stream_query_with_user_key(
    "What is deep learning?",
    USER_API_KEY
)
```

### JavaScript/TypeScript 示例

```typescript
// 标准查询
async function queryWithUserKey(
  queryText: string,
  userApiKey: string
): Promise<any> {
  const response = await fetch('http://localhost:8020/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Openai-Key': userApiKey  // 用户特定的 API Key
    },
    body: JSON.stringify({
      query: queryText,
      mode: 'mix',
      include_references: true
    })
  });
  
  return response.json();
}

// 使用示例
const result = await queryWithUserKey(
  'Explain neural networks',
  'sk-proj-user1-abc123...'
);
console.log(result.response);

// 流式查询
async function streamQueryWithUserKey(
  queryText: string,
  userApiKey: string
): Promise<void> {
  const response = await fetch('http://localhost:8020/query/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Openai-Key': userApiKey
    },
    body: JSON.stringify({
      query: queryText,
      mode: 'mix',
      stream: true
    })
  });
  
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;
    
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.trim()) {
        const data = JSON.parse(line);
        if (data.response) {
          process.stdout.write(data.response);
        }
        if (data.error) {
          console.error('Error:', data.error);
        }
      }
    }
  }
}

// 使用示例
await streamQueryWithUserKey(
  'What is deep learning?',
  'sk-proj-user1-abc123...'
);
```

### cURL 示例

```bash
# 标准查询
curl -X POST http://localhost:8020/query \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-proj-user1-abc123..." \
  -d '{
    "query": "What is machine learning?",
    "mode": "mix",
    "include_references": true
  }'

# 流式查询
curl -X POST http://localhost:8020/query/stream \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-proj-user1-abc123..." \
  -d '{
    "query": "Explain deep learning",
    "mode": "mix",
    "stream": true
  }'

# 数据检索
curl -X POST http://localhost:8020/query/data \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-proj-user1-abc123..." \
  -d '{
    "query": "Neural networks",
    "mode": "mix",
    "top_k": 5
  }'
```

## 多租户应用场景

### 场景 1: SaaS 应用

```python
# 为不同用户使用不同的 API Key
class LightRAGClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def query_for_user(self, user_id: str, query: str) -> dict:
        # 从数据库获取用户的 API Key
        user_api_key = self.get_user_api_key(user_id)
        
        headers = {
            "Content-Type": "application/json",
            "X-Openai-Key": user_api_key
        }
        
        payload = {
            "query": query,
            "mode": "mix"
        }
        
        response = requests.post(
            f"{self.api_url}/query",
            headers=headers,
            json=payload
        )
        
        return response.json()
    
    def get_user_api_key(self, user_id: str) -> str:
        # 从数据库或配置中获取用户的 API Key
        # 这里是示例实现
        return f"sk-proj-{user_id}-..."

# 使用示例
client = LightRAGClient("http://localhost:8020")

# 用户 A 的查询
result_a = client.query_for_user("user_a", "What is AI?")

# 用户 B 的查询
result_b = client.query_for_user("user_b", "Explain ML algorithms")
```

### 场景 2: 分层访问控制

```python
# 根据用户层级使用不同的 API Key
class TieredLightRAGClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.tier_keys = {
            "free": "sk-proj-free-tier-...",
            "pro": "sk-proj-pro-tier-...",
            "enterprise": "sk-proj-enterprise-tier-..."
        }
    
    def query_with_tier(self, tier: str, query: str) -> dict:
        api_key = self.tier_keys.get(tier)
        
        headers = {
            "Content-Type": "application/json",
            "X-Openai-Key": api_key
        }
        
        payload = {
            "query": query,
            "mode": "mix"
        }
        
        response = requests.post(
            f"{self.api_url}/query",
            headers=headers,
            json=payload
        )
        
        return response.json()

# 使用示例
client = TieredLightRAGClient("http://localhost:8020")

# 免费用户查询
result_free = client.query_with_tier("free", "Basic question?")

# 企业用户查询
result_enterprise = client.query_with_tier("enterprise", "Complex analysis?")
```

## 监控和日志

当使用自定义 API Key 时，LightRAG 会在日志中记录：

```
INFO: Using user-specific API key from X-Openai-Key header
DEBUG: Using user-specific API key for this request
```

可以通过这些日志消息来验证功能是否正常工作。

## 安全建议

1. **HTTPS**: 在生产环境中始终使用 HTTPS 来保护 API Key
2. **Key 管理**: 不要在客户端代码中硬编码 API Key
3. **权限控制**: 在后端服务器上管理用户的 API Key
4. **日志脱敏**: 确保日志中不会泄露完整的 API Key
5. **速率限制**: 考虑添加额外的速率限制保护

## 测试

使用提供的测试脚本验证功能：

```bash
# 运行测试脚本
python test_user_api_key.py
```

测试脚本会验证：
- `/query` 端点是否正确使用自定义 API Key
- `/query/stream` 端点是否正确使用自定义 API Key
- `/query/data` 端点是否正确使用自定义 API Key
- 不传递 Header 时是否使用默认 API Key

## 技术实现细节

### 参数传递流程

1. **API 层** (`query_routes.py`):
   - 从 HTTP Header 中提取 `X-Openai-Key`
   - 将其存储到 `QueryParam.user_api_key`

2. **LightRAG 核心** (`lightrag.py`):
   - 从 `QueryParam` 中提取 `user_api_key`
   - 将其传递到 `global_config["user_api_key"]`

3. **LLM 绑定** (`llm/openai.py`):
   - 从 kwargs 中提取 `user_api_key`
   - 优先使用 `user_api_key`（如果提供）
   - 否则使用默认的 API Key

### QueryParam 新字段

```python
@dataclass
class QueryParam:
    # ... 其他字段 ...
    
    user_api_key: str | None = None
    """User-specific API key for LLM requests.
    When provided, this API key will be used instead of the default LLM_BINDING_API_KEY
    for this specific query. This allows per-user API key tracking for token usage billing.
    """
```

## 故障排查

### 问题：API Key 没有生效

**检查项**:
1. 确认 Header 名称正确: `X-Openai-Key`
2. 检查日志中是否有 "Using user-specific API key" 消息
3. 验证 API Key 格式正确（通常以 `sk-` 开头）

### 问题：仍然使用默认 API Key

**可能原因**:
1. Header 名称拼写错误
2. 中间件或代理移除了自定义 Header
3. API 服务器版本过旧，不支持此功能

### 问题：认证失败

**检查项**:
1. 验证提供的 API Key 是否有效
2. 检查 OpenAI 账户是否有足够的额度
3. 确认 API Key 的权限设置

## 相关文档

- [LightRAG API 文档](../README.md)
- [查询参数配置](./UserQueryHeaderConfiguration.md)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
