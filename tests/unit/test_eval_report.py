from __future__ import annotations

import pytest

from eval.report import build_metrics


pytestmark = pytest.mark.unit


def test_build_metrics_includes_quality_and_engineering_panels() -> None:
    results = [
        {
            "case": {"id": "case-a", "category": "code"},
            "duration_ms": 120.0,
            "post_run_summary": {"session_id": "s1"},
            "final_state": {
                "execution_success": True,
                "artifact_paths": ["outputs/a.txt"],
            },
            "rubric": {
                "score": 84.0,
                "rule_score": 80.0,
                "passed": True,
                "mismatches": [],
            },
            "judge": {
                "attempted": True,
                "score": 92.0,
                "passed": True,
                "tokens_used": 40,
                "duration_ms": 18.0,
                "error": "",
            },
            "observability": {
                "cost": {
                    "total_tokens": 300,
                    "per_agent_tokens": {"Coder": 200, "Examiner": 100},
                    "per_model_tokens": {"mock-coder": 200, "mock-examiner": 100},
                },
                "trace": {
                    "llm_durations_ms": [30.0, 50.0],
                    "tool_durations_ms": [12.0],
                    "per_agent_llm_calls": {"Coder": 1, "Examiner": 1},
                },
                "workflow": {
                    "retry_triggered": False,
                    "retry_count": 0,
                    "examiner_loops": 1,
                    "query_turn_count": 1,
                    "compressed_turns": 0,
                },
            },
        },
        {
            "case": {"id": "case-b", "category": "workflow"},
            "duration_ms": 240.0,
            "post_run_summary": {"session_id": "s2", "compressed_turns": 1},
            "final_state": {
                "execution_success": True,
                "artifact_paths": [],
            },
            "rubric": {
                "score": 78.0,
                "rule_score": 75.0,
                "passed": True,
                "mismatches": [],
            },
            "judge": {
                "attempted": True,
                "score": 85.0,
                "passed": True,
                "tokens_used": 60,
                "duration_ms": 25.0,
                "error": "",
            },
            "observability": {
                "cost": {
                    "total_tokens": 500,
                    "per_agent_tokens": {"Coder": 300, "Examiner": 120, "Researcher": 80},
                    "per_model_tokens": {"mock-coder": 300, "mock-examiner": 120, "mock-researcher": 80},
                },
                "trace": {
                    "llm_durations_ms": [45.0, 60.0, 22.0],
                    "tool_durations_ms": [15.0, 20.0],
                    "per_agent_llm_calls": {"Coder": 1, "Examiner": 1, "Researcher": 1},
                },
                "workflow": {
                    "retry_triggered": True,
                    "retry_count": 1,
                    "examiner_loops": 2,
                    "query_turn_count": 2,
                    "compressed_turns": 1,
                },
            },
        },
    ]

    metrics = build_metrics(results, mode="mock")

    assert metrics["avg_score"] == 81.0
    assert metrics["avg_rule_score"] == 77.5
    assert metrics["panels"]["quality"]["judge_coverage_rate"] == 100.0
    assert metrics["panels"]["latency"]["p95_case_duration_ms"] >= 120.0
    assert metrics["panels"]["cost"]["total_agent_tokens"] == 800
    assert metrics["panels"]["reliability"]["retry_rate"] == 50.0
    assert metrics["slow_cases"][0]["id"] == "case-b"
