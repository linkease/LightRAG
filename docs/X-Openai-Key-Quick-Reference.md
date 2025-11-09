# X-Openai-Key Header - 快速参考

## 一句话说明
通过 HTTP Header 为每个查询请求指定独立的 OpenAI API Key

## 基本用法

```bash
# 添加 X-Openai-Key Header
curl -X POST http://localhost:8020/query \
  -H "X-Openai-Key: sk-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question", "mode": "mix"}'
```

## 支持的端点

| 端点 | 支持状态 |
|------|---------|
| POST /query | ✅ 支持 |
| POST /query/stream | ✅ 支持 |
| POST /query/data | ✅ 支持 |

## API Key 优先级

```
X-Openai-Key Header (最高)
    ↓
api_key 参数
    ↓
OPENAI_API_KEY 环境变量 (默认)
```

## Python 示例

```python
import requests

response = requests.post(
    "http://localhost:8020/query",
    headers={"X-Openai-Key": "sk-your-key"},
    json={"query": "What is AI?", "mode": "mix"}
)
```

## JavaScript 示例

```javascript
const response = await fetch('http://localhost:8020/query', {
  method: 'POST',
  headers: {
    'X-Openai-Key': 'sk-your-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'What is AI?',
    mode: 'mix'
  })
});
```

## 验证方法

服务器日志中查找：
```
INFO: Using user-specific API key from X-Openai-Key header
```

## 常见用途

✅ 多租户应用 - 每个租户独立 API Key  
✅ 成本追踪 - 精确统计每个用户的 token 消耗  
✅ 分层服务 - 不同等级用户使用不同的 key  
✅ 配额管理 - 各用户配额独立管理  

## 安全提示

⚠️ 生产环境必须使用 HTTPS  
⚠️ 不要在客户端代码中硬编码 API Key  
⚠️ 在后端服务器上管理和分发 API Key  

## 相关文档

📖 [详细文档](./X-Openai-Key-Header.md)  
🧪 [测试脚本](../test_user_api_key.py)  
📝 [实现说明](./IMPLEMENTATION_X_OPENAI_KEY.md)  
📋 [变更摘要](./CHANGE_SUMMARY_CN.md)
