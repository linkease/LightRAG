"""
示例：使用不同的模型处理用户查询和知识库构建

这个示例展示了如何在用户查询和系统内部调用时使用不同的 LLM 模型。
- 用户查询（API 调用）：使用 LLM_QUERY_MODEL（例如：云端大模型 gpt-4o）
- 内部调用（知识库构建）：使用 LLM_MODEL（例如：本地模型 qwen-local）

使用场景：
1. 相同的 API 服务地址，通过不同的模型名称路由到不同的模型实例
2. 成本优化：用户查询使用高性能云端模型，内部任务使用经济的本地模型
3. 性能优化：用户查询使用快速响应的模型，后台任务使用批处理优化的模型
4. 后端服务器根据模型名称路由到不同的实例（如阿里云 Qwen vs 本地 Qwen）

运行要求：
- 需要设置环境变量（通过 .env 文件或直接设置）
- OPENAI_API_KEY 或 LLM_BINDING_API_KEY：LLM API 密钥
- LLM_BINDING_HOST：LLM 服务地址（默认：https://api.openai.com/v1）
- LLM_MODEL：默认 LLM 模型名称（用于知识库构建，默认：gpt-4o-mini）
- LLM_QUERY_MODEL：用户查询专用模型名称（可选，如果不设置则使用 LLM_MODEL）
- EMBEDDING_BINDING_HOST：Embedding 服务地址（默认：http://localhost:11434）
- EMBEDDING_MODEL：Embedding 模型名称（默认：bge-m3:latest）
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./demoDir"


async def llm_model_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False,
    llm_query_model=None,  # 用户查询专用模型（如果设置，说明这是用户查询）
    **kwargs
) -> str:
    """
    自定义 LLM 函数，根据 llm_query_model 是否存在使用不同的模型
    
    Args:
        llm_query_model: 用户查询时使用的模型名称，如果为 None 则使用默认模型
    """
    # 决定使用哪个模型
    if llm_query_model:
        model = llm_query_model
        print(f"[用户查询] 使用查询模型: {model}")
    else:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        print(f"[内部调用] 使用默认模型: {model}")
    
    print(f"[DEBUG] keyword_extraction={keyword_extraction}")
    
    # 移除不应该传递给 openai_complete_if_cache 的参数
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("hashing_kv", None)  # 移除 hashing_kv，它由 LightRAG 内部管理
    
    result = await openai_complete_if_cache(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        llm_query_model=llm_query_model,
        keyword_extraction=keyword_extraction,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        **kwargs_copy,
    )
    print(f"[DEBUG] LLM 返回内容前 100 字: {str(result)[:100]}")
    return result


async def main():
    print("\n" + "="*80)
    print("LightRAG 用户查询模型配置示例")
    print("="*80)
    
    # 验证必要的环境变量
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        print("❌ 错误：未设置 OPENAI_API_KEY 或 LLM_BINDING_API_KEY")
        print("请在 .env 文件中配置或设置环境变量")
        return
    
    llm_model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    llm_query_model = os.getenv('LLM_QUERY_MODEL', None)
    
    print(f"✓ 默认 LLM 模型（知识库构建）: {llm_model}")
    print(f"✓ 查询 LLM 模型（用户查询）: {llm_query_model if llm_query_model else llm_model + ' (未设置，使用默认模型)'}")
    print(f"✓ LLM 服务: {os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1')}")
    print(f"✓ Embedding 模型: {os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')}")
    print(f"✓ Embedding 服务: {os.getenv('EMBEDDING_BINDING_HOST', 'http://localhost:11434')}")
    print(f"✓ 工作目录: {WORKING_DIR}")
    
    # 初始化 RAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
    
    print("\n[1/2] 初始化存储...")
    await rag.initialize_storages()
    print("[2/2] 初始化管道状态...")
    await initialize_pipeline_status()
    print("✓ 初始化完成\n")
    
    # 场景 1: 插入文档（内部调用，使用 LLM_MODEL）
    print("="*80)
    print("📝 场景 1: 插入文档（内部调用 - 知识库构建）")
    print("="*80)
    print("预期行为：")
    print(f"  - 使用默认模型: {llm_model}")
    print("  - 用于知识提取、实体识别等后台任务")
    print("  - 可以使用本地模型以降低成本")
    print("-"*80)
    
    test_content = (
        "人工智能（Artificial Intelligence, AI）是计算机科学的一个重要分支，"
        "致力于创建能够执行通常需要人类智能才能完成的任务的智能系统。"
        "这些任务包括视觉感知、语音识别、决策制定和语言翻译等。"
        "机器学习（Machine Learning, ML）是人工智能的核心子领域，"
        "它使计算机系统能够从数据和经验中自动学习和改进，而无需显式编程。"
        "深度学习（Deep Learning）则是机器学习的一个分支，"
        "使用多层神经网络来学习数据的复杂表示。"
    )
    
    print(f"\n插入文档内容预览：\n{test_content[:100]}...\n")
    await rag.ainsert(test_content)
    print("✓ 文档插入完成（已触发知识图谱构建）\n")
    
    # 场景 2: 用户查询（API 调用，使用 LLM_QUERY_MODEL）
    print("="*80)
    print("🔍 场景 2: 用户查询（API 调用 - 模拟用户请求）")
    print("="*80)
    print("预期行为：")
    if llm_query_model:
        print(f"  - 使用查询专用模型: {llm_query_model}")
        print("  - 适用于需要高质量响应的用户查询")
    else:
        print(f"  - 使用默认模型: {llm_model} (未配置 LLM_QUERY_MODEL)")
    print("  - 适用于 API 接口、Web 应用等面向用户的查询")
    print("-"*80)
    
    user_question = "什么是人工智能？它与机器学习有什么关系？"
    print(f"\n用户问题：{user_question}\n")
    
    param_user = QueryParam(
        mode="hybrid", 
        llm_query_model=llm_query_model,  # 设置查询专用模型（设置此参数即表示这是用户查询）
        response_type="Multiple Paragraphs"
    )
    
    result_user = await rag.aquery(user_question, param=param_user)
    print(f"查询结果：\n{'-'*80}")
    print(f"{str(result_user)[:300]}...")
    print(f"{'-'*80}\n")
    
    # 场景 3: 内部查询（不通过 API，使用默认模型）
    print("="*80)
    print("⚙️  场景 3: 程序内部查询（非 API 调用）")
    print("="*80)
    print("预期行为：")
    print(f"  - 使用默认模型: {llm_model}")
    print("  - 适用于程序内部逻辑、批处理任务等")
    print("  - 不设置 is_user_query 和 llm_query_model")
    print("-"*80)
    
    internal_question = "深度学习的主要特点是什么？"
    print(f"\n内部查询：{internal_question}\n")
    
    param_internal = QueryParam(
        mode="local",
        only_need_context=False
    )  # 不设置 llm_query_model，使用默认模型
    
    result_internal = await rag.aquery(internal_question, param=param_internal)
    print(f"查询结果：\n{'-'*80}")
    print(f"{str(result_internal)[:300]}...")
    print(f"{'-'*80}\n")
    
    # 总结
    print("="*80)
    print("✅ 示例运行完成！")
    print("="*80)
    print("\n💡 关键要点:")
    print("  1. ✓ 设置 LLM_MODEL 用于知识库构建（内部任务）")
    print("  2. ✓ 设置 LLM_QUERY_MODEL 用于用户查询（如果未设置则使用 LLM_MODEL）")
    print("  3. ✓ QueryParam 中设置 llm_query_model 即表示这是用户查询，将使用指定的模型")
    print("  4. ✓ 通过模型名称区分不同场景，后端可根据模型路由到不同实例")
    
    print("\n🎯 实际应用场景:")
    print("  • 成本优化：用户查询用云端高性能模型，内部任务用本地经济模型")
    print("  • 性能优化：不同场景使用不同优化的模型")
    print("  • 后端路由：通过模型名称路由到不同的服务实例")
    print("  • 灵活部署：相同 API 地址，不同模型名称对应不同后端")
    
    print("\n📝 配置示例（.env 文件）:")
    print("  LLM_MODEL=qwen-local              # 本地模型用于知识库构建")
    print("  LLM_QUERY_MODEL=gpt-4o            # 云端模型用于用户查询")
    print("  LLM_BINDING_HOST=http://localhost:8000/v1")
    print("  # 后端服务器根据模型名称路由：")
    print("  #   - qwen-local -> 本地 Qwen 实例")
    print("  #   - gpt-4o -> 云端 OpenAI 实例")
    
    print("\n📚 相关资源:")
    print("  • API 服务器: lightrag/api/lightrag_server.py")
    print("  • 查询路由: lightrag/api/routers/query.py")
    print("  • 核心实现: lightrag/llm/openai.py")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示：")
        print("  1. 检查 .env 配置是否正确")
        print("  2. 确认 Embedding 服务（如 Ollama）是否正在运行")
        print("  3. 确认 LLM API 密钥和服务地址是否正确")
