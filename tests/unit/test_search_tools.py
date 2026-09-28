from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.search_tools import (
    _dedupe_search_results,
    _rewrite_web_query,
    web_search,
)


pytestmark = pytest.mark.unit


def test_rewrite_web_query_sanitizes_llm_output() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='Search Query: "PINN loss balancing tutorial PyTorch GitHub"'
                )
            )
        ]
    )

    with patch("tools.search_tools._get_llm_client", return_value=mock_client):
        rewritten = _rewrite_web_query("帮我找一下 PINN 损失平衡教程")

    assert rewritten == "PINN loss balancing tutorial PyTorch GitHub"


def test_rewrite_web_query_falls_back_to_original_query_on_error() -> None:
    with patch("tools.search_tools._get_llm_client", side_effect=RuntimeError("llm down")):
        rewritten = _rewrite_web_query("PINN 官方文档")

    assert rewritten == "PINN 官方文档"


def test_dedupe_search_results_prefers_unique_links() -> None:
    deduped = _dedupe_search_results(
        [
            {
                "title": "PINN tutorial",
                "body": "A",
                "href": "https://github.com/org/repo",
                "provider": "DuckDuckGo",
                "query": "pinn tutorial",
            },
            {
                "title": "PINN tutorial duplicate",
                "body": "B",
                "href": "https://github.com/org/repo/",
                "provider": "SerpAPI",
                "query": "pinn tutorial pytorch",
            },
            {
                "title": "DeepXDE docs",
                "body": "Docs",
                "href": "https://deepxde.readthedocs.io/en/latest/",
                "provider": "DuckDuckGo",
                "query": "deepxde docs",
            },
        ]
    )

    assert len(deduped) == 2
    assert deduped[0]["href"] == "https://github.com/org/repo"
    assert deduped[1]["title"] == "DeepXDE docs"


def test_web_search_rewrites_and_reranks_candidates() -> None:
    raw_candidates = [
        {
            "title": "General PINN overview",
            "body": "Broad overview article.",
            "href": "https://example.com/pinn-overview",
            "provider": "DuckDuckGo",
            "query": "帮我找 PINN 教程",
        },
        {
            "title": "PyTorch PINN tutorial repo",
            "body": "Hands-on PINN training example with PyTorch.",
            "href": "https://github.com/example/pinn-tutorial",
            "provider": "SerpAPI",
            "query": "pinn pytorch tutorial github",
        },
    ]

    with patch(
        "tools.search_tools._rewrite_web_query",
        return_value="pinn pytorch tutorial github",
    ) as rewrite_mock:
        with patch(
            "tools.search_tools._collect_web_search_candidates",
            return_value=raw_candidates,
        ) as collect_mock:
            with patch(
                "tools.search_tools._rerank_web_results",
                return_value=[raw_candidates[1], raw_candidates[0]],
            ) as rerank_mock:
                rendered = web_search.invoke({"query": "帮我找 PINN 教程", "max_results": 2})

    rewrite_mock.assert_called_once_with("帮我找 PINN 教程")
    collect_mock.assert_called_once_with(
        ["帮我找 PINN 教程", "pinn pytorch tutorial github"],
        8,
    )
    rerank_mock.assert_called_once()
    assert "【查询改写】pinn pytorch tutorial github" in rendered
    assert rendered.index("[1] PyTorch PINN tutorial repo") < rendered.index("[2] General PINN overview")
