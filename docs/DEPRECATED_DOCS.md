# 已废弃的文档说明

以下文档描述的是旧的实现方案（基于 `is_user_query` 和 `OPENAI_QUERY_KEY`），已经被新的模型区分方案取代。

## 已废弃的文档

1. **`QUICKSTART_HEADER.md`** - 描述使用 HTTP Header 和 `is_user_query` 的旧方案
2. **`QUICKSTART_HEADER_CN.md`** - 中文版的 HTTP Header 配置指南（旧方案）
3. **`UserQueryHeaderConfiguration.md`** - 用户查询 Header 配置（旧方案）
4. **`PATCH_SUMMARY.md`** - 基于 `is_user_query` 的补丁说明（旧方案）
5. **`IMPLEMENTATION_COMPLETE.md`** - 旧方案的实现完成说明

## 新的文档

请参考以下文档了解新的实现方案：

- **`ModelBasedRouting.md`** - 基于模型名称的路由方案（当前推荐）

## 主要变化

### 旧方案（已废弃）
```python
# 使用不同的 API Key
OPENAI_API_KEY=sk-xxx       # 内部使用
OPENAI_QUERY_KEY=sk-yyy     # 用户查询使用

# 使用 is_user_query 标志
param = QueryParam(
    is_user_query=True,      # 显式标记
    mode="hybrid"
)
```

### 新方案（当前）
```python
# 使用不同的模型名称
LLM_MODEL=qwen-local         # 内部使用（知识库构建）
LLM_QUERY_MODEL=gpt-4o       # 用户查询使用（API 调用）

# 通过 llm_query_model 隐式判断
param = QueryParam(
    llm_query_model="gpt-4o",  # 设置此参数即表示用户查询
    mode="hybrid"
)
```

## 优势

新方案相比旧方案的优势：

1. **更简洁** - 只需设置模型名称，不需要额外的标志
2. **更直观** - 有查询模型 = 用户查询，没有 = 内部调用
3. **更灵活** - 可以在运行时动态指定模型
4. **更易维护** - 减少一个配置参数和字段
5. **更好的扩展性** - 支持后端根据模型名称路由到不同实例

## 迁移指南

如果你正在使用旧方案，请按以下步骤迁移：

1. **移除环境变量**
   ```bash
   # 删除
   # OPENAI_QUERY_KEY=xxx
   ```

2. **添加新环境变量**
   ```bash
   # 添加
   LLM_MODEL=your-default-model
   LLM_QUERY_MODEL=your-query-model  # 可选
   ```

3. **更新代码**
   ```python
   # 旧代码
   param = QueryParam(mode="hybrid", is_user_query=True)
   
   # 新代码
   query_model = os.getenv("LLM_QUERY_MODEL")
   param = QueryParam(mode="hybrid", llm_query_model=query_model)
   ```

4. **API 自动处理**
   - `/query` 和 `/query/stream` 端点会自动从环境变量读取 `LLM_QUERY_MODEL`
   - 无需手动设置

## 技术支持

如有疑问，请参考：
- 新方案文档：`docs/ModelBasedRouting.md`
- 示例代码：`examples/lightrag_user_query_header_demo.py`
- 核心实现：`lightrag/llm/openai.py`
