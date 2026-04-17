from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.session_manager import SessionManager
from orchestrator.graph import build_graph


pytestmark = pytest.mark.workflow


def test_retry_flow_recovers_after_examiner_fail(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    appended_records: list[dict] = []

    def mock_coder(state: dict) -> dict:
        retry_count = int(state.get("code_retry_count", 0) or 0)
        query = str(state.get("query", ""))
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content=f"coder retry={retry_count}"),
        ]
        if retry_count == 0:
            return {
                "current_step": "coder",
                "generated_code": "def main(:\n    print('broken')\n",
                "execution_stdout": "",
                "execution_stderr": "SyntaxError: invalid syntax",
                "execution_success": False,
                "artifact_paths": [],
                "code_retry_count": retry_count,
                "messages": messages,
            }

        artifact_path = tmp_path / "train_log.txt"
        artifact_path.write_text("epoch=0 loss=1.0e-1\n", encoding="utf-8")
        return {
            "current_step": "coder",
            "generated_code": "print('fixed pinn run')\n",
            "execution_stdout": "Recovered from SyntaxError and produced train log.",
            "execution_stderr": "",
            "execution_success": True,
            "artifact_paths": [str(artifact_path)],
            "code_retry_count": retry_count,
            "messages": messages,
        }

    def mock_examiner(state: dict) -> dict:
        retry_count = int(state.get("examiner_retry_count", 0) or 0)
        success = bool(state.get("execution_success", False))
        return {
            "current_step": "examiner",
            "academic_review": "",
            "code_review": (
                "[FAIL] SyntaxError detected."
                if not success
                else "[PASS] Retry fix validated successfully."
            ),
            "examiner_verdict": "PASS" if success else "FAIL",
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
                                result = graph.invoke(
                                    {
                                        "query": "修复一个语法错误的 PINN 脚本并重试",
                                        "messages": [],
                                        "session_id": "workflow-retry",
                                    },
                                    config={"configurable": {"thread_id": "workflow-retry"}},
                                )

    summary = manager.load_summary("workflow-retry")
    assert result["examiner_verdict"] == "PASS"
    assert result["execution_success"] is True
    assert result["code_retry_count"] == 1
    assert result["examiner_retry_count"] == 2
    assert "产物文件" in result["final_answer"]
    assert "Recovered from SyntaxError" in result["execution_stdout"]
    assert summary["last_code_summary"]
    assert appended_records
