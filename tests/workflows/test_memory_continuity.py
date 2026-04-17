from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.session_manager import SessionManager
from orchestrator.graph import build_graph


pytestmark = pytest.mark.workflow


def test_memory_continuity_updates_recent_queries_and_digest(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    appended_records: list[dict] = []

    def mock_coder(state: dict) -> dict:
        query = str(state.get("query", ""))
        retry_count = int(state.get("code_retry_count", 0) or 0)
        artifact_name = "train_log_followup.txt" if "继续" in query else "train_log_base.txt"
        artifact_path = tmp_path / artifact_name
        artifact_path.write_text(f"query={query}\n", encoding="utf-8")
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content=f"handled: {query}"),
        ]
        return {
            "current_step": "coder",
            "generated_code": f"print({query!r})\n",
            "execution_stdout": f"Handled query: {query}",
            "execution_stderr": "",
            "execution_success": True,
            "artifact_paths": [str(artifact_path)],
            "code_retry_count": retry_count,
            "messages": messages,
        }

    def mock_examiner(state: dict) -> dict:
        retry_count = int(state.get("examiner_retry_count", 0) or 0)
        return {
            "current_step": "examiner",
            "academic_review": "",
            "code_review": "[PASS] Memory-aware code run validated.",
            "examiner_verdict": "PASS",
            "examiner_retry_count": retry_count + 1,
        }

    with patch("memory.SessionManager", return_value=manager):
        with patch("memory.load_project_memory", return_value={}):
            with patch("memory.retrieve_experience_hints", return_value=[]):
                with patch("memory.append_experience_record", side_effect=appended_records.append):
                    with patch("orchestrator.graph.detect_intent", return_value="code"):
                        with patch("agents.coder.run_coder", side_effect=mock_coder):
                            with patch("agents.examiner.run_examiner", side_effect=mock_examiner):
                                graph = build_graph()
                                session_id = "workflow-memory"
                                first_query = "先写一个最小 PINN 示例"
                                second_query = "继续在上一轮基础上增加训练日志导出"

                                graph.invoke(
                                    {
                                        "query": first_query,
                                        "messages": [],
                                        "session_id": session_id,
                                    },
                                    config={"configurable": {"thread_id": session_id}},
                                )
                                second_result = graph.invoke(
                                    {
                                        "query": second_query,
                                        "messages": [],
                                        "session_id": session_id,
                                    },
                                    config={"configurable": {"thread_id": session_id}},
                                )

    summary = manager.load_summary("workflow-memory")
    assert second_result["examiner_verdict"] == "PASS"
    assert summary["recent_queries"][-2:] == [first_query, second_query]
    assert summary["compressed_turns"] >= 1
    assert summary["conversation_digest"]
    assert second_result["session_summary"]["session_id"] == "workflow-memory"
    assert appended_records
