from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eval.judge import judge_case, resolve_judge_mode


pytestmark = pytest.mark.unit


def test_resolve_judge_mode_auto_uses_heuristic_for_mock() -> None:
    assert resolve_judge_mode("auto", run_mode="mock") == "heuristic"
    assert resolve_judge_mode("off", run_mode="live") == "off"
    assert resolve_judge_mode("llm", run_mode="mock") == "llm"


def test_judge_case_heuristic_returns_normalized_payload() -> None:
    case = {
        "id": "demo",
        "category": "code",
        "required_sections": ["代码实现", "运行结果"],
        "expect_execution_success": True,
    }
    result = {
        "queries": ["写一个最小 PINN 示例并运行"],
        "final_state": {
            "execution_success": True,
            "code_retry_count": 1,
            "final_answer": (
                "## 代码实现\n\n```python\nprint('ok')\n```\n\n"
                "## 运行结果\n\n训练成功，包含 [来源: mock.pdf]"
            ),
        },
        "observability": {
            "workflow": {"retry_count": 1},
            "cost": {"total_tokens": 320},
        },
    }
    rule_rubric = {
        "score": 90.0,
        "rule_score": 90.0,
        "hard_failures": [],
        "mismatches": [],
    }

    payload = judge_case(case, result, rule_rubric, run_mode="mock", judge_mode="heuristic")

    assert payload["mode"] == "heuristic"
    assert payload["attempted"] is True
    assert payload["score"] is not None
    assert payload["passed"] is True
    assert payload["reason"]
    assert isinstance(payload["dimensions"], dict)


def test_judge_case_llm_parses_openai_compatible_json_response() -> None:
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"score": 88, "passed": true, "confidence": "high", '
                    '"reason": "Clear and correct.", '
                    '"strengths": ["complete"], "issues": [], '
                    '"dimensions": {"task_completion": 90, "technical_correctness": 88, '
                    '"clarity": 86, "groundedness": 87}}'
                )
            )
        )
    ]
    fake_response.usage = MagicMock(total_tokens=123)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    case = {"id": "llm-demo", "category": "qa"}
    result = {
        "queries": ["解释 PINN"],
        "final_state": {
            "intent": "qa",
            "examiner_verdict": "PASS",
            "final_answer": "PINN answer with [来源: mock.pdf]",
        },
        "post_run_summary": {"recent_queries": ["解释 PINN"]},
        "observability": {"workflow": {}, "cost": {}},
    }
    rule_rubric = {"score": 90.0, "rule_score": 90.0, "hard_failures": [], "mismatches": []}

    with patch("eval.judge.OpenAI", return_value=fake_client):
        payload = judge_case(case, result, rule_rubric, run_mode="live", judge_mode="llm")

    assert payload["mode"] == "llm"
    assert payload["score"] == 88.0
    assert payload["passed"] is True
    assert payload["confidence"] == "high"
    assert payload["tokens_used"] == 123
    assert payload["dimensions"]["task_completion"] == 90
