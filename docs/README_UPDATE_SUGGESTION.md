# README 更新建议

建议在主 README.md 中添加以下章节：

---

## Per-User API Key Support (多用户 API Key 支持)

LightRAG API 支持为每个查询请求使用独立的 OpenAI API Key，这对于多租户应用非常有用。

### 快速开始

在 HTTP 请求中添加 `X-Openai-Key` Header：

```bash
curl -X POST http://localhost:8020/query \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-your-user-specific-key" \
  -d '{
    "query": "Your question here",
    "mode": "mix"
  }'
```

### 支持的端点

- `POST /query` - 标准查询
- `POST /query/stream` - 流式查询  
- `POST /query/data` - 数据检索

### 使用场景

- **成本跟踪**: 为每个用户使用独立的 API Key，精确追踪 token 消耗
- **多租户应用**: 每个租户使用自己的 OpenAI 账户和配额
- **分层服务**: 不同用户等级使用不同的 API Key 配置

### 详细文档

查看完整文档：[X-Openai-Key Header 功能说明](docs/X-Openai-Key-Header.md)

---

## 建议插入位置

在 README.md 的 "API Usage" 或 "Advanced Features" 章节后面。

## 中文版本

在 README-zh.md 中添加相应的中文说明：

---

## 多用户 API Key 支持

LightRAG API 支持为每个查询请求使用独立的 OpenAI API Key，适用于多租户应用场景。

### 快速开始

在 HTTP 请求头中添加 `X-Openai-Key`：

```bash
curl -X POST http://localhost:8020/query \
  -H "Content-Type: application/json" \
  -H "X-Openai-Key: sk-你的用户专属密钥" \
  -d '{
    "query": "你的问题",
    "mode": "mix"
  }'
```

### 主要优势

- **独立计费**: 每个用户使用自己的 API Key，token 使用量单独统计
- **租户隔离**: 不同租户的 API 配额和限制相互独立
- **灵活管理**: 可为不同用户或访问层级配置不同的 API Key

### 详细文档

查看完整文档：[X-Openai-Key Header 功能说明](docs/X-Openai-Key-Header.md)

---
