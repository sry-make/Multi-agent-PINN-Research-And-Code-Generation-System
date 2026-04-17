"""Second-layer evaluation with heuristic or LLM-as-judge backends."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from config import (
    EVAL_JUDGE_ENABLED,
    EVAL_JUDGE_MAX_TOKENS,
    EVAL_JUDGE_MODEL,
    EVAL_JUDGE_MODE,
    EVAL_JUDGE_PASS_THRESHOLD,
    EVAL_JUDGE_TIMEOUT_SEC,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clip_text(value: Any, limit: int = 5000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_string_list(values: list[Any], limit: int = 3) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized[:limit]


def resolve_judge_mode(
    requested_mode: str | None = None,
    *,
    run_mode: str = "mock",
) -> str:
    """Resolve requested judge mode into a concrete backend."""
    mode = str(requested_mode or EVAL_JUDGE_MODE or "auto").strip().lower()
    if mode in {"", "none"}:
        mode = "auto"

    if mode == "off":
        return "off"

    if mode == "auto":
        if not EVAL_JUDGE_ENABLED:
            return "off"
        return "heuristic" if run_mode == "mock" else "llm"

    if mode in {"heuristic", "llm"}:
        return mode

    return "off"


def _extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}

    stripped = text.strip()
    candidates = [stripped]

    fenced = re.findall(r"```json\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced[-1].strip())

    block = _JSON_BLOCK_RE.search(stripped)
    if block:
        candidates.append(block.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    return {}


def _normalize_dimensions(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, int] = {}
    for key in ("task_completion", "technical_correctness", "clarity", "groundedness"):
        try:
            normalized[key] = int(raw.get(key, 0) or 0)
        except Exception:
            normalized[key] = 0
    return normalized


def _default_judge_result(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "attempted": mode != "off",
        "available": mode != "off",
        "model": EVAL_JUDGE_MODEL if mode == "llm" else "(heuristic-judge)",
        "score": None,
        "passed": None,
        "confidence": "",
        "strengths": [],
        "issues": [],
        "reason": "",
        "dimensions": {},
        "tokens_used": 0,
        "duration_ms": 0.0,
        "error": "",
        "raw_response": "",
    }


def _build_judge_prompt(
    case: dict[str, Any],
    result: dict[str, Any],
    rule_rubric: dict[str, Any],
) -> str:
    state = dict(result.get("final_state") or {})
    summary = dict(result.get("post_run_summary") or {})
    observability = dict(result.get("observability") or {})

    payload = {
        "case_id": case.get("id", ""),
        "category": case.get("category", ""),
        "queries": result.get("queries", []),
        "expected_intent": case.get("expected_intent", ""),
        "expected_verdict": case.get("expected_verdict", ""),
        "required_sections": case.get("required_sections", []),
        "required_keywords": case.get("required_keywords", []),
        "rule_score": rule_rubric.get("score", 0.0),
        "rule_mismatches": rule_rubric.get("mismatches", []),
        "hard_failures": rule_rubric.get("hard_failures", []),
        "observed_intent": state.get("intent", ""),
        "observed_verdict": state.get("examiner_verdict", ""),
        "execution_success": state.get("execution_success", False),
        "artifact_paths": state.get("artifact_paths", []),
        "academic_review": _clip_text(state.get("academic_review", ""), 1200),
        "code_review": _clip_text(state.get("code_review", ""), 1200),
        "execution_stdout": _clip_text(state.get("execution_stdout", ""), 1500),
        "execution_stderr": _clip_text(state.get("execution_stderr", ""), 1000),
        "final_answer": _clip_text(state.get("final_answer", ""), 7000),
        "recent_queries": summary.get("recent_queries", []),
        "compressed_turns": summary.get("compressed_turns", 0),
        "retry_count": dict(observability.get("workflow") or {}).get("retry_count", 0),
        "agent_tokens": dict(observability.get("cost") or {}).get("total_tokens", 0),
    }

    instructions = {
        "task": (
            "Evaluate whether the agent output satisfies the case intent, "
            "technical correctness, groundedness, and answer clarity."
        ),
        "requirements": [
            "Use only the provided case expectations and observed output.",
            "Do not invent missing evidence.",
            "Penalize empty, generic, or obviously incomplete answers.",
            "Keep rationale concise and interview-friendly.",
            "Return strict JSON only.",
        ],
        "response_schema": {
            "score": "number 0-100",
            "passed": "boolean",
            "confidence": "low | medium | high",
            "reason": "short string",
            "strengths": ["string", "string"],
            "issues": ["string", "string"],
            "dimensions": {
                "task_completion": "0-100",
                "technical_correctness": "0-100",
                "clarity": "0-100",
                "groundedness": "0-100",
            },
        },
    }

    return (
        "You are a strict evaluator for a resume-grade multi-agent system.\n"
        "Read the case, expected outcomes, and actual result, then judge quality.\n\n"
        f"Instructions:\n{json.dumps(instructions, ensure_ascii=False, indent=2)}\n\n"
        f"Case bundle:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _heuristic_judge(
    case: dict[str, Any],
    result: dict[str, Any],
    rule_rubric: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    state = dict(result.get("final_state") or {})
    final_answer = str(state.get("final_answer", "") or "")
    query_text = " ".join(str(item) for item in result.get("queries") or [])
    score = float(rule_rubric.get("score", 0.0))
    strengths: list[str] = []
    issues: list[str] = []

    if "[来源:" in final_answer:
        strengths.append("Answer keeps at least one explicit source marker.")
    elif case.get("category") in {"qa", "survey", "full_pipeline"}:
        score -= 8
        issues.append("Research-style response lacks explicit source grounding.")

    if "## 代码实现" in final_answer and case.get("category") in {"code", "full_pipeline", "workflow"}:
        strengths.append("Output exposes code implementation directly for review.")
    elif case.get("category") in {"code", "full_pipeline", "workflow"}:
        score -= 12
        issues.append("Code-oriented task does not clearly show the implementation section.")

    if "## 运行结果" in final_answer and state.get("execution_success"):
        strengths.append("Run results are surfaced instead of hiding execution details.")
    elif case.get("expect_execution_success") and not state.get("execution_success"):
        score -= 10
        issues.append("Execution success was expected but the run did not succeed.")

    if len(final_answer.strip()) < 180:
        score -= 8
        issues.append("Final answer is too short and may not be interview-grade.")
    else:
        strengths.append("Final answer has enough substance for demo review.")

    if int(state.get("code_retry_count", 0) or 0) > 0 and "修复" in query_text:
        strengths.append("Retry loop is visible, which helps demo the agent workflow.")

    if rule_rubric.get("hard_failures"):
        issues.extend(rule_rubric.get("hard_failures")[:2])

    score = max(0.0, min(100.0, round(score, 2)))
    passed = not rule_rubric.get("hard_failures") and score >= EVAL_JUDGE_PASS_THRESHOLD

    return {
        **_default_judge_result(mode),
        "score": score,
        "passed": passed,
        "confidence": "medium",
        "reason": (
            "Heuristic judge based on answer completeness, grounding, execution visibility, "
            "and alignment with task type."
        ),
        "strengths": _normalize_string_list(strengths, limit=3),
        "issues": _normalize_string_list(issues, limit=3),
        "dimensions": {
            "task_completion": int(score),
            "technical_correctness": int(max(0.0, min(100.0, score - (6 if issues else 0)))),
            "clarity": int(max(0.0, min(100.0, score))),
            "groundedness": int(max(0.0, min(100.0, score - (10 if "[来源:" not in final_answer else 0)))),
        },
    }


def _llm_judge(
    case: dict[str, Any],
    result: dict[str, Any],
    rule_rubric: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    from observability.tracer import timer

    prompt = _build_judge_prompt(case, result, rule_rubric)
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=EVAL_JUDGE_TIMEOUT_SEC,
    )
    response_text = ""
    tokens_used = 0

    with timer() as t:
        response = client.chat.completions.create(
            model=EVAL_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=EVAL_JUDGE_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge for a multi-agent system. "
                        "Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

    response_text = str(response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens_used = int(getattr(usage, "total_tokens", 0) or 0)

    payload = _extract_json_payload(response_text)
    if not payload:
        raise ValueError("judge did not return valid JSON")

    score = max(0.0, min(100.0, float(payload.get("score", 0.0) or 0.0)))
    passed = bool(payload.get("passed", False))
    confidence = str(payload.get("confidence", "")).strip().lower()
    confidence = confidence if confidence in {"low", "medium", "high"} else "medium"

    return {
        **_default_judge_result(mode),
        "score": round(score, 2),
        "passed": passed,
        "confidence": confidence,
        "reason": " ".join(str(payload.get("reason", "")).split()).strip(),
        "strengths": _normalize_string_list(list(payload.get("strengths") or []), limit=3),
        "issues": _normalize_string_list(list(payload.get("issues") or []), limit=3),
        "dimensions": _normalize_dimensions(payload.get("dimensions")),
        "tokens_used": tokens_used,
        "duration_ms": round(t.ms, 2),
        "raw_response": _clip_text(response_text, 4000),
    }


def judge_case(
    case: dict[str, Any],
    result: dict[str, Any],
    rule_rubric: dict[str, Any],
    *,
    run_mode: str = "mock",
    judge_mode: str | None = None,
) -> dict[str, Any]:
    """Run second-layer evaluation and return a normalized judge payload."""
    mode = resolve_judge_mode(judge_mode, run_mode=run_mode)
    base = _default_judge_result(mode)
    if mode == "off":
        base["attempted"] = False
        base["available"] = False
        return base

    try:
        if mode == "heuristic":
            return _heuristic_judge(case, result, rule_rubric, mode=mode)
        if mode == "llm":
            return _llm_judge(case, result, rule_rubric, mode=mode)
        base["attempted"] = False
        base["available"] = False
        return base
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["available"] = False
        return base
