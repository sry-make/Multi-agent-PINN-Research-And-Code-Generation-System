from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.session_manager import SessionManager
from orchestrator.graph import build_graph, node_memory_read, node_memory_writeback


pytestmark = pytest.mark.integration


def test_build_graph_contains_memory_nodes() -> None:
    graph = build_graph()
    nodes = list(graph.nodes.keys())

    for name in (
        "__start__",
        "parse_intent",
        "memory_read",
        "researcher",
        "coder",
        "examiner",
        "synthesize",
        "memory_writeback",
    ):
        assert name in nodes


def test_node_memory_read_compresses_prior_messages(tmp_path) -> None:
    manager = SessionManager(tmp_path)

    with patch("memory.SessionManager", return_value=manager):
        result = node_memory_read(
            {
                "query": "继续上一次的 PINN 代码调试",
                "intent": "code",
                "session_id": "pytest-memory-read",
                "messages": [
                    HumanMessage(content="写一个最小 PINN"),
                    AIMessage(content="我会先生成代码并运行。"),
                ],
            }
        )

    compressed_messages = result.get("messages") or []
    assert compressed_messages
    assert getattr(compressed_messages[0], "id", "") == "__remove_all__"
    assert "conversation_digest" in (result.get("session_summary") or {})


def test_node_memory_writeback_persists_summary_and_experience(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    appended_records: list[dict] = []

    def _append(record: dict) -> None:
        appended_records.append(record)

    with patch("memory.SessionManager", return_value=manager):
        with patch("memory.append_experience_record", side_effect=_append):
            result = node_memory_writeback(
                {
                    "session_id": "pytest-memory-writeback",
                    "query": "修复 PINN 训练时报错",
                    "intent": "code",
                    "generated_code": "print('retry')",
                    "execution_success": False,
                    "execution_stdout": "",
                    "execution_stderr": "RuntimeError: shape mismatch",
                    "examiner_verdict": "FAIL",
                    "code_review": "检查张量 shape，并确保 loss.backward 前维度一致。",
                    "artifact_paths": ["outputs/demo/error_log.txt"],
                    "messages": [],
                }
            )

    summary = manager.load_summary("pytest-memory-writeback")
    assert result["session_summary"]["session_id"] == "pytest-memory-writeback"
    assert "shape mismatch" in summary["last_error_summary"]
    assert appended_records
    assert appended_records[0]["fingerprint"]
