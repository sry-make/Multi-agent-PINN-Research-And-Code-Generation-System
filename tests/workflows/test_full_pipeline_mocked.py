from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.session_manager import SessionManager
from orchestrator.graph import build_graph


pytestmark = pytest.mark.workflow


def test_full_pipeline_mocked_produces_research_code_review_and_memory(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    appended_records: list[dict] = []

    def mock_researcher(state: dict) -> dict:
        query = str(state.get("query", ""))
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content="research complete"),
        ]
        return {
            "current_step": "researcher",
            "literature_report": (
                "PINN combines data constraints with physics loss and is suitable for low-data PDE tasks. "
                "[来源: mock_full_pipeline.pdf]"
            ),
            "design_proposal": (
                "Use a compact MLP, print training progress, and export train_log.txt plus loss.png."
            ),
            "retrieved_sources": [{"tool": "mock_search", "query": query}],
            "messages": messages,
        }

    def mock_coder(state: dict) -> dict:
        query = str(state.get("query", ""))
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content="coder complete"),
        ]
        train_log = tmp_path / "train_log.txt"
        train_log.write_text("epoch=0 loss=1.0e-1\n", encoding="utf-8")
        loss_png = tmp_path / "loss.png"
        loss_png.write_bytes(b"PNG MOCK\n")
        return {
            "current_step": "coder",
            "generated_code": "import torch\nprint('full pipeline demo')\n",
            "execution_stdout": "Epoch 0 | total_loss=1.0e-1 | physics_loss=1.0e-2",
            "execution_stderr": "",
            "execution_success": True,
            "artifact_paths": [str(train_log), str(loss_png)],
            "code_retry_count": int(state.get("code_retry_count", 0) or 0),
            "messages": messages,
        }

    def mock_examiner(state: dict) -> dict:
        retry_count = int(state.get("examiner_retry_count", 0) or 0)
        return {
            "current_step": "examiner",
            "academic_review": "[PASS] Research summary is evidence-grounded.",
            "code_review": "[PASS] Code executed successfully and exported artifacts.",
            "examiner_verdict": "PASS",
            "examiner_retry_count": retry_count + 1,
        }

    with patch("memory.SessionManager", return_value=manager):
        with patch("memory.load_project_memory", return_value={}):
            with patch("memory.retrieve_experience_hints", return_value=[]):
                with patch("memory.append_experience_record", side_effect=appended_records.append):
                    with patch("orchestrator.graph.detect_intent", return_value="full_pipeline"):
                        with patch("agents.researcher.run_researcher", side_effect=mock_researcher):
                            with patch("agents.coder.run_coder", side_effect=mock_coder):
                                with patch("agents.examiner.run_examiner", side_effect=mock_examiner):
                                    graph = build_graph()
                                    result = graph.invoke(
                                        {
                                            "query": "先综述再实现一个最小 PINN 演示",
                                            "messages": [],
                                            "session_id": "workflow-full-pipeline",
                                        },
                                        config={"configurable": {"thread_id": "workflow-full-pipeline"}},
                                    )

    summary = manager.load_summary("workflow-full-pipeline")
    assert result["examiner_verdict"] == "PASS"
    assert result["execution_success"] is True
    assert "## 文献综述" in result["final_answer"]
    assert "## 技术方案" in result["final_answer"]
    assert "## 代码实现" in result["final_answer"]
    assert "## 产物文件" in result["final_answer"]
    assert summary["recent_queries"][-1] == "先综述再实现一个最小 PINN 演示"
    assert appended_records
