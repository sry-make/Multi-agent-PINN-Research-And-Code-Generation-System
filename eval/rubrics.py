"""Rule-based scoring for fixed evaluation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import EVAL_JUDGE_PASS_THRESHOLD, EVAL_JUDGE_WEIGHT


PASS_THRESHOLD = 75.0


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pass", "passed"}
    return False


def _last_query(case: dict[str, Any]) -> str:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        return str(turns[-1]).strip()
    return str(case.get("query", "")).strip()


def _artifact_names(paths: list[Any]) -> list[str]:
    return [Path(str(path)).name for path in paths if str(path).strip()]


def _contains_all(text: str, expected: list[str]) -> tuple[int, list[str]]:
    haystack = text.lower()
    matched = [item for item in expected if str(item).lower() in haystack]
    return len(matched), matched


def _add_check(
    checks: list[dict[str, Any]],
    mismatches: list[str],
    *,
    name: str,
    condition: bool,
    max_score: int,
    expected: Any,
    actual: Any,
    hard: bool,
    detail: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(condition),
            "score": max_score if condition else 0,
            "max_score": max_score,
            "expected": expected,
            "actual": actual,
            "hard": hard,
            "detail": detail,
        }
    )
    if not condition:
        mismatches.append(detail or f"{name} mismatch")


def _add_ratio_check(
    checks: list[dict[str, Any]],
    mismatches: list[str],
    *,
    name: str,
    matched_count: int,
    total_count: int,
    max_score: int,
    expected: Any,
    actual: Any,
    detail: str,
) -> None:
    ratio = 1.0 if total_count <= 0 else matched_count / total_count
    score = round(max_score * ratio, 2)
    passed = matched_count == total_count
    checks.append(
        {
            "name": name,
            "passed": passed,
            "score": score,
            "max_score": max_score,
            "expected": expected,
            "actual": actual,
            "hard": False,
            "detail": detail,
        }
    )
    if not passed:
        mismatches.append(detail)


def score_case_result(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Score a case result against declarative expectations."""
    state = dict(result.get("final_state") or {})
    summary = dict(result.get("post_run_summary") or state.get("session_summary") or {})
    final_answer = _normalize_text(state.get("final_answer", ""))
    artifact_paths = list(state.get("artifact_paths") or [])
    checks: list[dict[str, Any]] = []
    mismatches: list[str] = []

    if result.get("error"):
        checks.append(
            {
                "name": "runner_error",
                "passed": False,
                "score": 0,
                "max_score": 100,
                "expected": "successful evaluation run",
                "actual": result["error"],
                "hard": True,
                "detail": result["error"],
            }
        )
        return {
            "score": 0.0,
            "rule_score": 0.0,
            "passed": False,
            "checks": checks,
            "hard_failures": [result["error"]],
            "mismatches": [result["error"]],
            "judge_score": None,
            "judge_passed": None,
            "judge_mode": "off",
            "judge_error": "",
            "judge_weight": 0.0,
            "score_breakdown": {
                "rule": 0.0,
                "judge": None,
                "overall": 0.0,
            },
        }

    expected_intent = case.get("expected_intent")
    if expected_intent:
        _add_check(
            checks,
            mismatches,
            name="intent_match",
            condition=str(state.get("intent", "")).strip() == str(expected_intent).strip(),
            max_score=15,
            expected=expected_intent,
            actual=state.get("intent", ""),
            hard=True,
            detail=f"expected intent={expected_intent}, got {state.get('intent', '')}",
        )

    expected_verdict = case.get("expected_verdict")
    if expected_verdict:
        _add_check(
            checks,
            mismatches,
            name="verdict_match",
            condition=str(state.get("examiner_verdict", "")).strip().upper() == str(expected_verdict).strip().upper(),
            max_score=15,
            expected=expected_verdict,
            actual=state.get("examiner_verdict", ""),
            hard=True,
            detail=(
                f"expected verdict={expected_verdict}, "
                f"got {state.get('examiner_verdict', '')}"
            ),
        )

    if "expect_execution_success" in case:
        expected_success = _to_bool(case.get("expect_execution_success"))
        actual_success = _to_bool(state.get("execution_success", False))
        _add_check(
            checks,
            mismatches,
            name="execution_success_match",
            condition=actual_success == expected_success,
            max_score=15,
            expected=expected_success,
            actual=actual_success,
            hard=True,
            detail=f"expected execution_success={expected_success}, got {actual_success}",
        )

    if "expect_artifacts" in case:
        expected_artifacts = _to_bool(case.get("expect_artifacts"))
        actual_artifacts = bool(artifact_paths)
        _add_check(
            checks,
            mismatches,
            name="artifact_expectation",
            condition=actual_artifacts == expected_artifacts,
            max_score=10,
            expected=expected_artifacts,
            actual=actual_artifacts,
            hard=True,
            detail=f"expected artifacts={expected_artifacts}, got {actual_artifacts}",
        )

    if "expect_retry" in case:
        expected_retry = _to_bool(case.get("expect_retry"))
        actual_retry = bool(
            int(state.get("code_retry_count", 0) or 0) > 0
            or int(state.get("examiner_retry_count", 0) or 0) > 1
        )
        _add_check(
            checks,
            mismatches,
            name="retry_expectation",
            condition=actual_retry == expected_retry,
            max_score=10,
            expected=expected_retry,
            actual=actual_retry,
            hard=True,
            detail=f"expected retry={expected_retry}, got {actual_retry}",
        )

    if case.get("expect_memory_writeback"):
        last_query = _last_query(case)
        recent_queries = [str(item) for item in summary.get("recent_queries") or [] if str(item).strip()]
        memory_ok = bool(summary.get("session_id")) and (last_query in recent_queries)
        _add_check(
            checks,
            mismatches,
            name="memory_writeback",
            condition=memory_ok,
            max_score=10,
            expected=f"session summary contains latest query: {last_query}",
            actual=recent_queries,
            hard=True,
            detail="session summary was not updated with the latest query",
        )

    if "expect_recent_queries_min" in case:
        expected_min = int(case.get("expect_recent_queries_min") or 0)
        recent_queries = [str(item) for item in summary.get("recent_queries") or [] if str(item).strip()]
        _add_check(
            checks,
            mismatches,
            name="recent_queries_min",
            condition=len(recent_queries) >= expected_min,
            max_score=10,
            expected=f">={expected_min}",
            actual=len(recent_queries),
            hard=True,
            detail=(
                f"expected at least {expected_min} recent queries in session summary, "
                f"got {len(recent_queries)}"
            ),
        )

    if "expect_experience_hints_min" in case:
        expected_min = int(case.get("expect_experience_hints_min") or 0)
        actual_count = len(list(state.get("experience_hints") or []))
        _add_check(
            checks,
            mismatches,
            name="experience_hint_retrieval",
            condition=actual_count >= expected_min,
            max_score=10,
            expected=f">={expected_min}",
            actual=actual_count,
            hard=True,
            detail=(
                f"expected at least {expected_min} experience hints, got {actual_count}"
            ),
        )

    required_sections = [str(item).strip() for item in case.get("required_sections") or [] if str(item).strip()]
    if required_sections:
        matched_count, matched_sections = _contains_all(final_answer, required_sections)
        _add_ratio_check(
            checks,
            mismatches,
            name="required_sections",
            matched_count=matched_count,
            total_count=len(required_sections),
            max_score=10,
            expected=required_sections,
            actual=matched_sections,
            detail=(
                "missing required sections: "
                + ", ".join(section for section in required_sections if section not in matched_sections)
            ),
        )

    required_keywords = [str(item).strip() for item in case.get("required_keywords") or [] if str(item).strip()]
    if required_keywords:
        matched_count, matched_keywords = _contains_all(final_answer, required_keywords)
        _add_ratio_check(
            checks,
            mismatches,
            name="required_keywords",
            matched_count=matched_count,
            total_count=len(required_keywords),
            max_score=10,
            expected=required_keywords,
            actual=matched_keywords,
            detail=(
                "missing required keywords: "
                + ", ".join(keyword for keyword in required_keywords if keyword not in matched_keywords)
            ),
        )

    required_artifact_names = [
        str(item).strip()
        for item in case.get("required_artifact_names") or []
        if str(item).strip()
    ]
    if required_artifact_names:
        artifact_names = _artifact_names(artifact_paths)
        matched = [
            name
            for name in required_artifact_names
            if name in artifact_names
        ]
        _add_ratio_check(
            checks,
            mismatches,
            name="required_artifact_names",
            matched_count=len(matched),
            total_count=len(required_artifact_names),
            max_score=10,
            expected=required_artifact_names,
            actual=artifact_names,
            detail=(
                "missing required artifacts: "
                + ", ".join(name for name in required_artifact_names if name not in artifact_names)
            ),
        )

    possible = sum(check["max_score"] for check in checks) or 1
    earned = sum(float(check["score"]) for check in checks)
    hard_failures = [check["detail"] for check in checks if check["hard"] and not check["passed"]]
    score = round(earned / possible * 100.0, 2)
    passed = not hard_failures and score >= PASS_THRESHOLD

    return {
        "score": score,
        "rule_score": score,
        "passed": passed,
        "checks": checks,
        "hard_failures": hard_failures,
        "mismatches": mismatches,
        "judge_score": None,
        "judge_passed": None,
        "judge_mode": "off",
        "judge_error": "",
        "judge_weight": 0.0,
        "score_breakdown": {
            "rule": score,
            "judge": None,
            "overall": score,
        },
    }


def finalize_case_score(
    rule_rubric: dict[str, Any],
    judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine rule-based score with optional second-layer judge score.

    Rule score remains the hard contract layer; judge score adds semantic quality
    when available. Any hard rule failure still blocks pass.
    """
    merged = dict(rule_rubric or {})
    rule_score = float(merged.get("rule_score", merged.get("score", 0.0)) or 0.0)
    hard_failures = list(merged.get("hard_failures") or [])

    judge_payload = dict(judge_result or {})
    judge_score_raw = judge_payload.get("score")
    judge_valid = (
        judge_score_raw is not None
        and not str(judge_payload.get("error", "")).strip()
        and judge_payload.get("passed") is not None
    )
    judge_score = float(judge_score_raw or 0.0) if judge_valid else None
    judge_weight = EVAL_JUDGE_WEIGHT if judge_valid else 0.0

    if judge_valid:
        overall_score = round(
            rule_score * (1.0 - judge_weight) + judge_score * judge_weight,
            2,
        )
    else:
        overall_score = round(rule_score, 2)

    passed = not hard_failures and overall_score >= PASS_THRESHOLD
    judge_passed = judge_payload.get("passed") if judge_valid else None
    if judge_valid and judge_passed is False:
        passed = False
    if judge_valid and overall_score < EVAL_JUDGE_PASS_THRESHOLD:
        passed = False

    merged.update(
        {
            "score": overall_score,
            "rule_score": round(rule_score, 2),
            "passed": passed,
            "judge_score": round(judge_score, 2) if judge_score is not None else None,
            "judge_passed": judge_passed,
            "judge_mode": str(judge_payload.get("mode", "off")).strip() or "off",
            "judge_error": str(judge_payload.get("error", "")).strip(),
            "judge_weight": judge_weight,
            "score_breakdown": {
                "rule": round(rule_score, 2),
                "judge": round(judge_score, 2) if judge_score is not None else None,
                "overall": overall_score,
            },
        }
    )
    return merged
