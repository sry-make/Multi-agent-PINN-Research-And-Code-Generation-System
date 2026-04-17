from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.session_manager import (
    SessionManager,
    build_session_summary,
    compress_message_history,
    format_code_memory,
)


pytestmark = pytest.mark.unit


def test_build_session_summary_tracks_success_and_failure(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session_id = manager.reset_session(prefix="pytest-session")
    summary = manager.load_summary(session_id)

    successful = build_session_summary(
        summary,
        {
            "session_id": session_id,
            "query": "实现一个最小 PINN demo",
            "intent": "code",
            "generated_code": "print('ok')",
            "execution_success": True,
            "execution_stdout": "ok",
            "execution_stderr": "",
            "examiner_verdict": "PASS",
            "artifact_paths": ["outputs/demo/train_log.txt"],
        },
    )
    manager.save_summary(session_id, successful)
    reloaded = manager.load_summary(session_id)

    assert reloaded["last_intent"] == "code"
    assert "print('ok')" in reloaded["last_code_snippet"]
    assert "print('ok')" in reloaded["last_successful_code_snippet"]
    assert reloaded["last_successful_artifacts"] == ["outputs/demo/train_log.txt"]

    failed = build_session_summary(
        reloaded,
        {
            "session_id": session_id,
            "query": "继续修复上面的 PINN 代码",
            "intent": "code",
            "generated_code": "import torch\nprint('retry')",
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": "RuntimeError: shape mismatch",
            "examiner_verdict": "FAIL",
            "code_review": "[规则预检 FAIL] 代码执行失败",
            "messages": [],
        },
    )

    assert "shape mismatch" in failed["last_error_summary"]
    assert "shape mismatch" in failed["last_failure_error_summary"]
    assert "print('retry')" in failed["last_failed_code_snippet"]
    assert "print('ok')" in failed["last_successful_code_snippet"]

    code_memory = format_code_memory(failed)
    assert "Last successful baseline" in code_memory
    assert "Last failed attempt" in code_memory
    assert "RuntimeError: shape mismatch" in code_memory


def test_compress_message_history_rolls_messages_into_digest() -> None:
    _, compressed_summary, compressed = compress_message_history(
        [
            HumanMessage(content="先解释一下 PINN 的损失函数"),
            AIMessage(content="我先从 PDE residual 和边界条件讲起。"),
            HumanMessage(content="继续，顺便给我一个最小代码示例"),
            AIMessage(content="我会先检索文献，再给出实现建议。"),
        ],
        {"conversation_digest": "", "compressed_turns": 0},
    )

    assert compressed is True
    assert "Compressed 4 internal message(s)" in compressed_summary["conversation_digest"]
    assert compressed_summary["compressed_turns"] == 1
    assert compressed_summary["message_window_size"] == 0
