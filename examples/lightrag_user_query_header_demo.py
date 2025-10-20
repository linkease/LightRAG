"""
示例：根据 is_user_query 标志添加不同的 HTTP Header

这个示例展示了如何在用户查询和系统内部调用时使用不同的 HTTP Header。
- 用户查询（API 调用）：添加 "X-User-Query: true" Header
- 内部调用（知识库构建）：添加 "X-User-Query: false" Header

使用场景：
1. 相同的大模型服务器地址，但需要根据调用类型区分请求
2. 后端服务器根据 Header 路由到不同的模型实例（如阿里云 Qwen vs 本地 Qwen）
3. 计费或监控：区分用户查询和系统内部调用的资源使用

运行要求：
- 需要设置环境变量（通过 .env 文件或直接设置）
- OPENAI_API_KEY 或 LLM_BINDING_API_KEY：LLM API 密钥
- LLM_BINDING_HOST：LLM 服务地址（默认：https://api.openai.com/v1）
- LLM_MODEL：LLM 模型名称（默认：gpt-4o-mini）
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


async def llm_model_func_with_header(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False,
    is_user_query=False,  # 关键参数：区分用户查询和内部调用
    **kwargs
) -> str:
    """
    自定义 LLM 函数，根据 is_user_query 标志使用不同的 API KEY
    
    Args:
        is_user_query: True 表示用户查询（API 调用），False 表示内部调用
    """
    # Debug: 打印 is_user_query 和 OPENAI_QUERY_KEY
    print(f"[DEBUG] is_user_query={is_user_query}, keyword_extraction={keyword_extraction}")
    print(f"[DEBUG] OPENAI_QUERY_KEY={os.getenv('OPENAI_QUERY_KEY')}")
    print(f"[DEBUG] OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}")
    
    # 移除不应该传递给 openai_complete_if_cache 的参数
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("hashing_kv", None)  # 移除 hashing_kv，它由 LightRAG 内部管理
    
    result = await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        is_user_query=is_user_query,
        keyword_extraction=keyword_extraction,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        **kwargs_copy,
    )
    print(f"[DEBUG] LLM 返回内容前 100 字: {str(result)[:100]}")
    return result


async def main():
    print("\n" + "="*80)
    print("LightRAG 用户查询 Header 配置示例")
    print("="*80)
    
    # 验证必要的环境变量
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        print("❌ 错误：未设置 OPENAI_API_KEY 或 LLM_BINDING_API_KEY")
        print("请在 .env 文件中配置或设置环境变量")
        return
    
    print(f"✓ LLM 模型: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"✓ LLM 服务: {os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1')}")
    print(f"✓ Embedding 模型: {os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')}")
    print(f"✓ Embedding 服务: {os.getenv('EMBEDDING_BINDING_HOST', 'http://localhost:11434')}")
    print(f"✓ 工作目录: {WORKING_DIR}")
    print(f"[DEBUG] 当前环境 OPENAI_QUERY_KEY={os.getenv('OPENAI_QUERY_KEY')}")
    print(f"[DEBUG] 当前环境 OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}")
    
    # 初始化 RAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func_with_header,
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
    
    # 场景 1: 插入文档（内部调用，is_user_query=False）
    print("="*80)
    print("📝 场景 1: 插入文档（内部调用 - 知识库构建）")
    print("="*80)
    print("预期行为：")
    print("  - LLM 调用时添加 'X-User-Query: false' Header")
    print("  - 后端可根据此 Header 路由到内部处理模型")
    print("  - 适用于知识提取、实体识别等后台任务")
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
    
    # 场景 2: 用户查询（API 调用，is_user_query=True）
    print("="*80)
    print("🔍 场景 2: 用户查询（API 调用 - 模拟用户请求）")
    print("="*80)
    print("预期行为：")
    print("  - LLM 调用时添加 'X-User-Query: true' Header")
    print("  - 后端可根据此 Header 路由到用户服务模型")
    print("  - 适用于 API 接口、Web 应用等面向用户的查询")
    print("-"*80)
    
    user_question = "什么是人工智能？它与机器学习有什么关系？"
    print(f"\n用户问题：{user_question}\n")
    
    param_user = QueryParam(
        mode="hybrid", 
        is_user_query=True,  # 显式标记为用户查询
        response_type="Multiple Paragraphs"
    )
    
    result_user = await rag.aquery(user_question, param=param_user)
    print(f"查询结果：\n{'-'*80}")
    print(f"{str(result_user)[:300]}...")
    print(f"{'-'*80}\n")
    
    # 场景 3: 内部查询（不通过 API，is_user_query=False）
    print("="*80)
    print("⚙️  场景 3: 程序内部查询（非 API 调用）")
    print("="*80)
    print("预期行为：")
    print("  - LLM 调用时添加 'X-User-Query: false' Header")
    print("  - 后端可根据此 Header 路由到内部处理模型")
    print("  - 适用于程序内部逻辑、批处理任务等")
    print("-"*80)
    
    internal_question = "深度学习的主要特点是什么？"
    print(f"\n内部查询：{internal_question}\n")
    
    param_internal = QueryParam(
        mode="local",
        only_need_context=False
    )  # 不设置 is_user_query，默认为 False
    
    result_internal = await rag.aquery(internal_question, param=param_internal)
    print(f"查询结果：\n{'-'*80}")
    print(f"{str(result_internal)[:300]}...")
    print(f"{'-'*80}\n")
    
    # 总结
    print("="*80)
    print("✅ 示例运行完成！")
    print("="*80)
    print("\n💡 关键要点:")
    print("  1. ✓ 通过 API 路由（/query, /query/stream）的查询会自动设置 is_user_query=True")
    print("  2. ✓ 知识库构建（插入文档、实体提取等）自动使用 is_user_query=False")
    print("  3. ✓ 程序内部直接调用 aquery 时，默认 is_user_query=False")
    print("  4. ✓ 可以根据 Header 在后端服务器进行路由、计费或监控")
    
    print("\n🎯 实际应用场景:")
    print("  • 使用不同的模型处理用户查询（高性能）和内部任务（经济型）")
    print("  • 基于请求类型的计费和配额管理")
    print("  • 请求优先级控制（用户查询优先）")
    print("  • 监控和日志分析（区分用户行为和系统行为）")
    
    print("\n� 相关资源:")
    print("  • 完整文档: docs/UserQueryHeaderConfiguration.md")
    print("  • API 服务器: lightrag/api/lightrag_server.py")
    print("  • 查询路由: lightrag/api/routers/query.py")
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
