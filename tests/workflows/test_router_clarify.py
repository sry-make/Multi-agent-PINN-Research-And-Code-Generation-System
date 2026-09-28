from __future__ import annotations

from unittest.mock import patch

import pytest

from memory.session_manager import SessionManager
from orchestrator.graph import build_graph
from orchestrator.router import RouterDecision


pytestmark = pytest.mark.workflow


def test_low_confidence_query_returns_clarification_response(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    decision = RouterDecision(
        intent="clarify",
        source="llm_clarify",
        confidence=0.41,
        entropy=0.92,
        reason="目标不够明确",
        scores={
            "qa": 0.34,
            "survey": 0.22,
            "code": 0.24,
            "full_pipeline": 0.20,
        },
        needs_clarification=True,
        clarification_question="你是希望我先解释一下，还是直接写并运行代码？",
        missing_information=["是要解释概念还是执行任务"],
        default_intent="qa",
        defaulted=False,
    )

    with patch("memory.SessionManager", return_value=manager):
        with patch("memory.load_project_memory", return_value={}):
            with patch("memory.retrieve_experience_hints", return_value=[]):
                with patch("orchestrator.graph.resolve_intent", return_value=decision):
                    graph = build_graph()
                    result = graph.invoke(
                        {
                            "query": "帮我看一下 PINN",
                            "messages": [],
                            "session_id": "workflow-router-clarify",
                        },
                        config={"configurable": {"thread_id": "workflow-router-clarify"}},
                    )

    assert result["intent"] == "clarify"
    assert "## 路由澄清" in result["final_answer"]
    assert "你是希望我先解释一下，还是直接写并运行代码？" in result["final_answer"]
    assert "默认宽容路径: `qa`" in result["final_answer"]
