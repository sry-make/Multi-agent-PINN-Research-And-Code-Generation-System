from __future__ import annotations

import pytest

from config import EVAL_JUDGE_WEIGHT
from eval.rubrics import finalize_case_score


pytestmark = pytest.mark.unit


def test_finalize_case_score_blends_rule_and_judge_score() -> None:
    rule_rubric = {
        "score": 80.0,
        "rule_score": 80.0,
        "passed": True,
        "hard_failures": [],
        "mismatches": [],
    }
    judge_result = {
        "mode": "heuristic",
        "score": 90.0,
        "passed": True,
        "error": "",
    }

    merged = finalize_case_score(rule_rubric, judge_result)

    assert merged["rule_score"] == 80.0
    assert merged["judge_score"] == 90.0
    assert merged["score"] == round(80.0 * (1.0 - EVAL_JUDGE_WEIGHT) + 90.0 * EVAL_JUDGE_WEIGHT, 2)
    assert merged["passed"] is True
    assert merged["judge_mode"] == "heuristic"


def test_finalize_case_score_keeps_rule_failures_hard() -> None:
    rule_rubric = {
        "score": 88.0,
        "rule_score": 88.0,
        "passed": False,
        "hard_failures": ["expected execution_success=True, got False"],
        "mismatches": ["expected execution_success=True, got False"],
    }
    judge_result = {
        "mode": "heuristic",
        "score": 95.0,
        "passed": True,
        "error": "",
    }

    merged = finalize_case_score(rule_rubric, judge_result)

    assert merged["passed"] is False
    assert merged["hard_failures"]
