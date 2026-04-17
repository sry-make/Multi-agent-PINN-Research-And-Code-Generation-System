from __future__ import annotations

import pytest

from memory.experience_store import (
    append_experience_record,
    build_experience_fingerprint,
    build_experience_record,
    format_experience_hints,
    load_experience_records,
    retrieve_experience_hints,
)


pytestmark = pytest.mark.unit


def test_append_experience_record_merges_by_fingerprint(tmp_path) -> None:
    db_path = tmp_path / "experience.jsonl"

    first = append_experience_record(
        {
            "ts": "2026-04-17T00:00:00+00:00",
            "session_id": "s1",
            "intent": "code",
            "query": "embedding init failed on torch import",
            "success": False,
            "error_type": "missing_torch",
            "symptom": "sentence-transformers failed because torch was missing",
            "resolution_hint": "install torch cpu wheel",
            "tags": ["rag", "env"],
        },
        path=db_path,
    )
    second = append_experience_record(
        {
            "ts": "2026-04-17T01:00:00+00:00",
            "session_id": "s2",
            "intent": "code",
            "query": "embedding init failed on torch import again",
            "success": True,
            "error_type": "missing_torch",
            "symptom": "sentence-transformers failed because torch was missing",
            "resolution_hint": "install torch cpu wheel and retry",
            "tags": ["rag", "env", "fix"],
            "artifact_paths": ["outputs/demo/embedding_check.txt"],
        },
        path=db_path,
    )

    records = load_experience_records(path=db_path)
    assert len(records) == 1
    assert first["fingerprint"] == second["fingerprint"] == build_experience_fingerprint(second)
    assert records[0]["occurrence_count"] == 2
    assert records[0]["success_count"] == 1
    assert records[0]["failure_count"] == 1
    assert records[0]["experience_score"] >= 4


def test_retrieve_experience_hints_prefers_relevant_and_scored_records(tmp_path) -> None:
    db_path = tmp_path / "experience.jsonl"
    append_experience_record(
        {
            "intent": "code",
            "query": "torch embedding error",
            "success": True,
            "error_type": "missing_torch",
            "symptom": "embedding failed because torch was missing",
            "resolution_hint": "install torch cpu wheel",
            "occurrence_count": 2,
            "success_count": 2,
            "failure_count": 0,
        },
        path=db_path,
    )
    append_experience_record(
        {
            "intent": "qa",
            "query": "PINN overview",
            "success": True,
            "error_type": "successful_run",
            "symptom": "general qa response",
            "resolution_hint": "reuse researcher summary",
        },
        path=db_path,
    )

    hints = retrieve_experience_hints(
        "torch embedding init error",
        intent="code",
        limit=2,
        path=db_path,
    )

    assert hints
    assert hints[0]["intent"] == "code"
    assert hints[0]["error_type"] == "missing_torch"
    assert hints[0]["retrieval_score"] >= hints[0]["experience_score"]

    formatted = format_experience_hints(hints)
    assert "seen=" in formatted
    assert "score=" in formatted


def test_build_experience_record_contains_aggregatable_fields() -> None:
    record = build_experience_record(
        {
            "session_id": "tui-demo",
            "intent": "code",
            "query": "修复 PINN 训练时报错",
            "examiner_verdict": "FAIL",
            "execution_success": False,
            "execution_stderr": "RuntimeError: shape mismatch",
            "code_review": "检查张量 shape，并确保 loss.backward 前维度一致。",
            "artifact_paths": ["outputs/demo/error_log.txt"],
        }
    )

    assert record["fingerprint"]
    assert record["query_prefix"]
    assert record["occurrence_count"] == 1
    assert record["failure_count"] == 1
    assert record["success_count"] == 0
    assert record["experience_score"] >= 1
