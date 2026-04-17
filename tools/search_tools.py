"""
tools/search_tools.py — 外部搜索工具

工具列表:
    search_arxiv(query, max_results)  → arXiv 实时论文搜索
    web_search(query, max_results)    → DuckDuckGo 网络搜索（SerpAPI 可选）
"""

from __future__ import annotations

from langchain_core.tools import tool

from config import (
    ARXIV_MAX_RESULTS,
    ARXIV_SORT_BY,
    WEB_SEARCH_MAX_RESULTS,
    SERPAPI_KEY,
)


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
        "relevance":       arxiv.SortCriterion.Relevance,
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
    except Exception as e:
        return f"arXiv 搜索失败: {e}\n建议检查网络连接或使用本地知识库。"

    if not results:
        return f"arXiv 上未找到与 '{query}' 相关的论文。"

    parts = []
    for i, paper in enumerate(results, 1):
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        year    = paper.published.year if paper.published else "n.d."
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
    使用 DuckDuckGo 进行网络搜索，获取最新信息或非 arXiv 来源的内容。

    适用于查找教程、博客、GitHub 实现、技术文档等非论文资源。
    当 arXiv 和本地知识库都无法回答时作为补充。

    Args:
        query:       搜索关键词
        max_results: 返回结果数，默认 5

    Returns:
        格式化的搜索结果列表，含标题、摘要、链接
    """
    max_results = min(max_results, 10)

    # 优先尝试 SerpAPI（结果质量更高）
    if SERPAPI_KEY:
        return _serpapi_search(query, max_results)
    return _duckduckgo_search(query, max_results)


def _duckduckgo_search(query: str, max_results: int) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"DuckDuckGo 搜索失败: {e}"

    if not results:
        return f"未找到与 '{query}' 相关的网络结果。"

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "（无标题）")
        body  = r.get("body",  "")[:200]
        href  = r.get("href",  "")
        parts.append(f"[{i}] {title}\n    {body}...\n    {href}")

    return "\n\n".join(parts)


def _serpapi_search(query: str, max_results: int) -> str:
    try:
        import httpx
        params = {
            "q":       query,
            "api_key": SERPAPI_KEY,
            "num":     max_results,
            "engine":  "google",
        }
        resp = httpx.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # SerpAPI 失败时自动降级到 DuckDuckGo
        return _duckduckgo_search(query, max_results)

    organic = data.get("organic_results", [])
    if not organic:
        return f"未找到与 '{query}' 相关的结果。"

    parts = []
    for i, r in enumerate(organic[:max_results], 1):
        title   = r.get("title",   "（无标题）")
        snippet = r.get("snippet", "")[:200]
        link    = r.get("link",    "")
        parts.append(f"[{i}] {title}\n    {snippet}...\n    {link}")

    return "\n\n".join(parts)
