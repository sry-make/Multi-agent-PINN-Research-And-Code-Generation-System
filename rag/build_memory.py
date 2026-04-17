"""
RAG 模块迁移层 — 从 V1 兼容升级到 V2

变更:
    - 嵌入模型: all-MiniLM-L6-v2 (384-dim) → BAAI/bge-m3 (1024-dim, 多语言)
    - Reranker 设备: CPU → CUDA (RTX 4060 Ti)
    - ChromaDB Collection: pinn_papers → pinn_papers_v2
    - 新增: build_v2() 一键重建知识库
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL_NAME,
    EMBED_MODEL_SOURCE,
    EMBED_DEVICE,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K_COARSE,
    RAG_TOP_K_FINAL,
    PAPERS_DIR,
)

# ── ChromaDB 客户端 ────────────────────────────────────────────
_client_db: chromadb.PersistentClient | None = None
_collection = None


def _format_embedding_init_error(exc: Exception) -> str:
    """将嵌入模型初始化异常转成更可执行的诊断信息。"""
    root = exc
    while getattr(root, "__cause__", None) is not None:
        root = root.__cause__

    reason_lines: list[str] = []
    install_hint = ""

    if isinstance(root, ModuleNotFoundError):
        missing = getattr(root, "name", "") or "unknown"
        if missing == "torch":
            reason_lines.append("当前 conda 环境缺少 PyTorch，sentence-transformers 无法导入。")
            install_hint = (
                '\n建议安装: python -m pip install "torch>=2.2.0" '
                '--extra-index-url https://download.pytorch.org/whl/cpu'
            )
        elif missing == "sentence_transformers":
            reason_lines.append("当前环境缺少 sentence-transformers。")
            install_hint = (
                '\n建议安装: python -m pip install "sentence-transformers>=3.0.0"'
            )
        else:
            reason_lines.append(f"当前环境缺少 Python 依赖模块: {missing}")

    elif "sentence_transformers" in str(root):
        reason_lines.append(
            "sentence-transformers 依赖链初始化失败，请优先检查 torch / sentence-transformers 是否完整。"
        )

    return (
        "初始化嵌入模型失败。"
        f"\n- 配置模型: {EMBED_MODEL_NAME}"
        f"\n- 实际来源: {EMBED_MODEL_SOURCE}"
        f"\n- 实际设备: {EMBED_DEVICE}"
        "\n可能原因:"
        "\n1. 本地 Hugging Face 缓存不完整"
        "\n2. 当前环境无法联网下载模型"
        "\n3. 指定设备不可用"
        + (f"\n4. 依赖问题: {'; '.join(reason_lines)}" if reason_lines else "")
        + install_hint
        + f"\n原始错误: {exc}"
    )


def _get_collection():
    global _client_db, _collection
    if _collection is not None:
        return _collection

    _client_db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_SOURCE,
            device=EMBED_DEVICE,
        )
    except Exception as e:
        raise RuntimeError(_format_embedding_init_error(e)) from e
    _collection = _client_db.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )
    return _collection


# ── HyDE 查询重写 ──────────────────────────────────────────────

def rewrite_query_hyde(query: str, llm_client, model: str) -> str:
    """
    HyDE: 将短查询扩写为假设性学术段落，提升向量检索命中率。
    使用英文扩写（解决 V1 中文 HyDE 与英文嵌入空间不匹配问题）。
    """
    prompt = (
        "You are a PINN (Physics-Informed Neural Networks) expert. "
        "Expand the following short query into a detailed academic hypothesis "
        "containing rich terminology (collocation points, residual loss, PDE, etc.). "
        "Write ONLY the expanded text in English, 100-200 words, no extra explanation.\n\n"
        f"Query: {query}\nExpanded:"
    )
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ── 检索 ───────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    top_k: int = RAG_TOP_K_FINAL,
    use_hyde: bool = True,
    use_reranker: bool = True,
    llm_client=None,
    model: str = "",
) -> tuple[str, list[dict]]:
    """
    完整检索流水线: HyDE → 粗召回 → Reranker 精排

    Returns:
        (context_str, metadatas)
    """
    collection = _get_collection()

    # Step 1: HyDE 查询重写
    search_query = query
    if use_hyde and llm_client:
        search_query = rewrite_query_hyde(query, llm_client, model)

    # Step 2: 粗召回
    coarse_k = RAG_TOP_K_COARSE if use_reranker else top_k
    results = collection.query(query_texts=[search_query], n_results=coarse_k)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    # Step 3: Reranker 精排
    if use_reranker and docs:
        from rag.reranker import BGEReranker
        reranker = BGEReranker()
        docs, metas = reranker.rerank(query, docs, metas, top_k=top_k)
    else:
        docs  = docs[:top_k]
        metas = metas[:top_k]

    context_str = "\n\n---\n\n".join(docs)
    return context_str, metas


# ── 知识库构建 ─────────────────────────────────────────────────

def build_v2(papers_dir: str = PAPERS_DIR) -> None:
    """
    重建 V2 知识库（bge-m3 嵌入）。

    用法:
        python -m rag.build_memory
    """
    from pypdf import PdfReader

    collection = _get_collection()
    pdf_files = list(Path(papers_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  {papers_dir} 中未找到 PDF 文件")
        return

    print(f"📚 发现 {len(pdf_files)} 篇论文，开始构建 V2 知识库...")
    print(f"   嵌入模型: {EMBED_MODEL_NAME}")
    print(f"   模型来源: {EMBED_MODEL_SOURCE}")
    print(f"   运行设备: {EMBED_DEVICE}")

    for doc_id, pdf_path in enumerate(pdf_files):
        print(f"📄 [{doc_id+1}/{len(pdf_files)}] 解析: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        # 滑动窗口分块
        chunks, ids, metas = [], [], []
        for i, start in enumerate(range(0, len(text), RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP)):
            chunk = text[start : start + RAG_CHUNK_SIZE]
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
                ids.append(f"v2_doc{doc_id}_chunk{i}")
                metas.append({"source": pdf_path.name, "doc_id": doc_id})

        if chunks:
            # upsert 而非 add：重复运行时不会因 ID 冲突报错
            collection.upsert(documents=chunks, metadatas=metas, ids=ids)
            print(f"   ✅ 入库 {len(chunks)} 个片段")

    print(f"\n🎉 V2 知识库构建完成！共 {collection.count()} 条向量记录。")


if __name__ == "__main__":
    build_v2()
