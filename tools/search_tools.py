"""
tools/search_tools.py — 外部搜索工具

工具列表:
    search_arxiv(query, max_results)  → arXiv 实时论文搜索
    web_search(query, max_results)    → Web 搜索（查询改写 + rerank）
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from langchain_core.tools import tool

from config import (
    ARXIV_MAX_RESULTS,
    ARXIV_SORT_BY,
    MODEL_RESEARCHER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    SERPAPI_KEY,
    WEB_SEARCH_CANDIDATE_POOL,
    WEB_SEARCH_ENABLE_QUERY_REWRITE,
    WEB_SEARCH_ENABLE_RERANK,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_QUERY_REWRITE_MODEL,
)


_QUERY_REWRITE_PROMPT = """\
You optimize web search queries for a research agent.

Rewrite the user's request into one concise search-engine-friendly query.
Rules:
- Keep the technical intent unchanged.
- If the original query is Chinese, translate it to natural English search keywords.
- Preserve important entities such as PINN, PDE, PyTorch, GitHub, tutorial, documentation.
- Prefer compact keyword-style phrasing over long sentences.
- Output exactly one line, with no quotes, markdown, bullets, or explanation.
"""

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+|[\u4e00-\u9fff]+")
_REWRITE_PREFIX_RE = re.compile(
    r"^(rewritten query|search query|query)\s*[:：-]\s*",
    re.IGNORECASE,
)


def _get_llm_client():
    """延迟初始化 LLM 客户端，避免模块导入时发起连接。"""
    from openai import OpenAI

    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sanitize_rewritten_query(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""

    cleaned = cleaned.replace("```", " ").strip().strip("\"'`")
    cleaned = _REWRITE_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.splitlines()[0].strip()
    cleaned = cleaned.strip("\"'`")
    return _normalize_text(cleaned)


def _rewrite_web_query(query: str) -> str:
    """将用户请求压缩为更适合搜索引擎的查询语句。"""
    if not WEB_SEARCH_ENABLE_QUERY_REWRITE:
        return query

    normalized_query = _normalize_text(query)
    if not normalized_query:
        return query

    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model=WEB_SEARCH_QUERY_REWRITE_MODEL or MODEL_RESEARCHER,
            messages=[
                {"role": "system", "content": _QUERY_REWRITE_PROMPT},
                {"role": "user", "content": normalized_query},
            ],
            temperature=0.0,
            max_tokens=48,
        )
        raw = response.choices[0].message.content or ""
        rewritten = _sanitize_rewritten_query(raw)
        return rewritten or normalized_query
    except Exception:
        return normalized_query


def _build_search_result(
    *,
    title: str,
    body: str,
    href: str,
    provider: str,
    query_used: str,
) -> dict[str, str]:
    return {
        "title": _normalize_text(title) or "（无标题）",
        "body": _normalize_text(body),
        "href": str(href or "").strip(),
        "provider": provider,
        "query": _normalize_text(query_used),
    }


def _normalize_href(href: str) -> str:
    href = str(href or "").strip()
    if not href:
        return ""

    parsed = urlparse(href)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _dedupe_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """按链接优先去重，缺链接时退化为标题+摘要键。"""
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()

    for result in results:
        href_key = _normalize_href(result.get("href", ""))
        fallback_key = (
            f"{_normalize_text(result.get('title', '')).lower()}|"
            f"{_normalize_text(result.get('body', '')).lower()[:180]}"
        )
        key = href_key or fallback_key
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(result)

    return deduped


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(_normalize_text(text))
        if len(token.strip()) > 1
    }


def _source_quality_boost(href: str) -> float:
    host = urlparse(str(href or "")).netloc.lower()
    if not host:
        return 0.0
    if any(tag in host for tag in ("github.com", "readthedocs.io", "docs.")):
        return 0.35
    if "arxiv.org" in host:
        return 0.25
    if host.endswith(".edu") or host.endswith(".ac.uk"):
        return 0.15
    return 0.0


def _fallback_rerank_web_results(
    ranking_query: str,
    results: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    """当 CrossEncoder 不可用时，用轻量词项重叠做保底排序。"""
    query_tokens = _tokenize(ranking_query)

    def score(result: dict[str, str]) -> tuple[float, str]:
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href", "")
        doc_tokens = _tokenize(" ".join([title, body, href]))
        title_tokens = _tokenize(title)
        overlap = len(query_tokens & doc_tokens)
        title_overlap = len(query_tokens & title_tokens)
        return (
            overlap + (1.5 * title_overlap) + _source_quality_boost(href),
            href,
        )

    ranked = sorted(results, key=score, reverse=True)
    return ranked[:top_k]


def _build_rerank_document(result: dict[str, str]) -> str:
    parts = [
        result.get("title", ""),
        result.get("body", ""),
        result.get("href", ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def _rerank_web_results(
    ranking_query: str,
    results: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    if not results:
        return []

    top_k = min(top_k, len(results))
    if not WEB_SEARCH_ENABLE_RERANK or len(results) <= top_k:
        return results[:top_k]

    documents = [_build_rerank_document(result) for result in results]
    metadatas = [dict(result) for result in results]

    try:
        from rag.reranker import BGEReranker

        reranker = BGEReranker()
        _, ranked_metas = reranker.rerank(
            ranking_query,
            documents,
            metadatas,
            top_k=top_k,
        )
        return ranked_metas
    except Exception:
        return _fallback_rerank_web_results(ranking_query, results, top_k)


def _duckduckgo_search_raw(query: str, max_results: int) -> list[dict[str, str]]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    normalized: list[dict[str, str]] = []
    for row in results:
        normalized.append(
            _build_search_result(
                title=row.get("title", "（无标题）"),
                body=row.get("body", ""),
                href=row.get("href", ""),
                provider="DuckDuckGo",
                query_used=query,
            )
        )
    return normalized


def _serpapi_search_raw(query: str, max_results: int) -> list[dict[str, str]]:
    import httpx

    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results,
        "engine": "google",
    }
    resp = httpx.get("https://serpapi.com/search", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    normalized: list[dict[str, str]] = []
    for row in data.get("organic_results", [])[:max_results]:
        normalized.append(
            _build_search_result(
                title=row.get("title", "（无标题）"),
                body=row.get("snippet", ""),
                href=row.get("link", ""),
                provider="SerpAPI",
                query_used=query,
            )
        )
    return normalized


def _search_backend(query: str, max_results: int) -> list[dict[str, str]]:
    if SERPAPI_KEY:
        try:
            return _serpapi_search_raw(query, max_results)
        except Exception:
            return _duckduckgo_search_raw(query, max_results)
    return _duckduckgo_search_raw(query, max_results)


def _collect_web_search_candidates(
    queries: list[str],
    max_results: int,
) -> list[dict[str, str]]:
    unique_queries: list[str] = []
    for query in queries:
        normalized = _normalize_text(query)
        if normalized and normalized not in unique_queries:
            unique_queries.append(normalized)

    if not unique_queries:
        return []

    all_results: list[dict[str, str]] = []
    errors: list[str] = []

    for query in unique_queries:
        try:
            all_results.extend(_search_backend(query, max_results))
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    deduped = _dedupe_search_results(all_results)
    if deduped:
        return deduped

    if errors:
        raise RuntimeError("; ".join(errors[:2]))
    return []


def _format_web_search_results(
    original_query: str,
    rewritten_query: str,
    results: list[dict[str, str]],
) -> str:
    if not results:
        return f"未找到与 '{original_query}' 相关的网络结果。"

    parts: list[str] = []
    if rewritten_query and rewritten_query != _normalize_text(original_query):
        parts.append(f"【查询改写】{rewritten_query}")

    for i, result in enumerate(results, 1):
        title = result.get("title", "（无标题）")
        body = result.get("body", "")
        href = result.get("href", "")
        provider = result.get("provider", "Web")
        parts.append(
            f"[{i}] {title}\n"
            f"    来源: {provider}\n"
            f"    摘要: {body[:200]}...\n"
            f"    链接: {href}"
        )

    return "\n\n".join(parts)


@tool
def search_arxiv(query: str, max_results: int = ARXIV_MAX_RESULTS) -> str:
    """
    在 arXiv 上实时搜索最新学术论文。

    当需要查找本地知识库没有收录的最新文献，或验证某篇论文是否存在时调用。
    适用于 PINN、深度学习、偏微分方程求解等领域的文献调研。

    Args:
        query:       搜索关键词（建议用英文以获得最佳结果）
        max_results: 返回论文数量，默认 5，最多 10

    Returns:
        格式化的论文列表，含标题、作者、摘要、arXiv 链接
    """
    import arxiv

    max_results = min(max_results, 10)  # 硬上限，避免返回过多 token

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_by = sort_map.get(ARXIV_SORT_BY, arxiv.SortCriterion.Relevance)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
        results = list(client.results(search))
    except Exception as exc:
        return f"arXiv 搜索失败: {exc}\n建议检查网络连接或使用本地知识库。"

    if not results:
        return f"arXiv 上未找到与 '{query}' 相关的论文。"

    parts = []
    for i, paper in enumerate(results, 1):
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        year = paper.published.year if paper.published else "n.d."
        summary = paper.summary.replace("\n", " ")[:300]
        parts.append(
            f"[{i}] {paper.title}\n"
            f"    作者: {authors} ({year})\n"
            f"    摘要: {summary}...\n"
            f"    链接: {paper.entry_id}"
        )

    return "\n\n".join(parts)


@tool
def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """
    使用 Web 搜索获取最新信息或非 arXiv 来源的内容。

    升级点:
    - 先用 LLM 将用户问题改写为更适合搜索引擎的查询词
    - 同时保留原始查询与改写查询，扩大候选召回
    - 对候选结果去重后使用 reranker 重新排序

    Args:
        query:       搜索关键词
        max_results: 返回结果数，默认 5

    Returns:
        格式化的搜索结果列表，含标题、摘要、链接
    """
    max_results = min(max_results, 10)
    candidate_pool = min(10, max(max_results, WEB_SEARCH_CANDIDATE_POOL))

    rewritten_query = _rewrite_web_query(query)
    candidate_queries = [query]
    if rewritten_query and rewritten_query != _normalize_text(query):
        candidate_queries.append(rewritten_query)

    try:
        candidates = _collect_web_search_candidates(candidate_queries, candidate_pool)
    except Exception as exc:
        return f"Web 搜索失败: {exc}"

    if not candidates:
        return f"未找到与 '{query}' 相关的网络结果。"

    ranking_query = query
    if rewritten_query and rewritten_query != _normalize_text(query):
        ranking_query = f"{_normalize_text(query)}\n{rewritten_query}"

    ranked = _rerank_web_results(ranking_query, candidates, max_results)
    return _format_web_search_results(query, rewritten_query, ranked)
