"""
tools/rag_tools.py — 本地知识库检索工具

将 rag/build_memory.py 中的检索流水线封装为 LangChain @tool，
供 Researcher Agent 注册并调用。

工具列表:
    search_local_papers(query, mode)  → 本地 ChromaDB 检索（HyDE + Reranker）
"""

from __future__ import annotations

from langchain_core.tools import tool

from config import (
    MODEL_RESEARCHER,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    RAG_TOP_K_FINAL,
)


def _get_llm_client():
    """延迟初始化 LLM 客户端，避免模块加载时连接 Ollama"""
    from openai import OpenAI
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


@tool
def search_local_papers(query: str, mode: str = "hyde_reranker") -> str:
    """
    在本地 PINN 论文知识库中检索与问题最相关的原文片段。

    当用户询问 PINN、物理信息神经网络、偏微分方程、损失函数构造、
    正逆问题、配点法等学术细节时，必须调用此工具获取原文依据。

    Args:
        query: 用户的问题或关键词
        mode:  检索模式
               - "hyde_reranker" （默认）: HyDE 查询扩写 + BGE Reranker 精排，精度最高
               - "hyde"          : 仅 HyDE 扩写，速度稍快
               - "direct"        : 直接向量检索，最快但精度最低

    Returns:
        格式化的原文片段字符串，含来源文件名
    """
    from rag.build_memory import retrieve_context

    use_hyde     = mode in ("hyde", "hyde_reranker")
    use_reranker = mode == "hyde_reranker"

    llm_client = _get_llm_client() if use_hyde else None

    context_str, metadatas = retrieve_context(
        query=query,
        top_k=RAG_TOP_K_FINAL,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        llm_client=llm_client,
        model=MODEL_RESEARCHER,
    )

    if not context_str.strip():
        return "本地知识库中未检索到相关内容，请尝试 search_arxiv 或 web_search。"

    # 以 metadatas 为权威列表驱动循环，避免 split 结果与 metadatas 数量不对齐
    chunks = context_str.split("\n\n---\n\n")
    n = min(len(chunks), len(metadatas))   # 取短者，防止 IndexError
    parts = []
    for i in range(n):
        source = metadatas[i].get("source", "unknown")
        parts.append(f"【片段 {i+1} | 来源: {source}】\n{chunks[i].strip()}")

    return "\n\n".join(parts)
