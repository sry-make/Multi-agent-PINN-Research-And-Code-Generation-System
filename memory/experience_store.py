"""Experience memory read/write helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EXPERIENCE_DB_PATH

EXPERIENCE_HINT_LIMIT = 3
QUERY_PREFIX_TOKEN_LIMIT = 6
QUERY_PREFIX_CHAR_LIMIT = 80


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pass", "passed"}
    return False


def _normalize_string_list(values: list[Any], limit: int = 8) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = _trim_text(str(value), 160)
        if text and text not in normalized:
            normalized.append(text)
    return normalized[-limit:]


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[0-9a-zA-Z_\-\+\.\u4e00-\u9fff]+", str(text).lower()))
    if tokens:
        return tokens
    fallback = _trim_text(str(text).lower(), 32)
    return {fallback} if fallback else set()


def _query_prefix(text: str) -> str:
    tokens = list(
        re.findall(r"[0-9a-zA-Z_\-\+\.\u4e00-\u9fff]+", str(text).lower())
    )
    if tokens:
        prefix = " ".join(tokens[:QUERY_PREFIX_TOKEN_LIMIT])
        return _trim_text(prefix, QUERY_PREFIX_CHAR_LIMIT)
    return _trim_text(str(text).lower(), QUERY_PREFIX_CHAR_LIMIT)


def build_experience_fingerprint(record: dict[str, Any]) -> str:
    """Build a lightweight fingerprint for deduping similar experience entries."""
    intent = _trim_text(str(record.get("intent", "")).strip().lower(), 40) or "unknown"
    error_type = _trim_text(str(record.get("error_type", "")).strip().lower(), 80) or "generic"
    query = str(record.get("query_prefix", "")).strip() or str(record.get("query", "")).strip()
    query_prefix = _query_prefix(query) or "generic-query"
    return f"{intent}::{error_type}::{query_prefix}"


def _compute_experience_score(record: dict[str, Any]) -> int:
    occurrence_count = max(1, int(record.get("occurrence_count", 1) or 1))
    success_count = max(0, int(record.get("success_count", 0) or 0))
    failure_count = max(0, int(record.get("failure_count", 0) or 0))
    has_resolution = bool(str(record.get("resolution_hint", "")).strip())
    has_artifacts = bool(list(record.get("artifact_paths") or []))
    verdict = str(record.get("examiner_verdict", "")).strip().upper()

    score = 1
    score += min(occurrence_count, 5)
    score += min(success_count, 3)
    if success_count and success_count >= failure_count:
        score += 2
    if has_resolution:
        score += 1
    if has_artifacts:
        score += 1
    if verdict == "PASS":
        score += 1
    return score


def _normalize_experience_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record or {})
    ts = str(normalized.get("ts", "")).strip() or _utc_now()

    success = _as_bool(normalized.get("success", False))
    occurrence_count = max(1, int(normalized.get("occurrence_count", normalized.get("count", 1)) or 1))
    success_count = int(normalized.get("success_count", 0) or 0)
    failure_count = int(normalized.get("failure_count", 0) or 0)
    if success_count == 0 and failure_count == 0:
        success_count = occurrence_count if success else 0
        failure_count = 0 if success else occurrence_count

    query = _trim_text(str(normalized.get("query", "")), 300)
    query_prefix = str(normalized.get("query_prefix", "")).strip() or _query_prefix(query)

    normalized_record = {
        "ts": ts,
        "first_seen_at": str(normalized.get("first_seen_at", "")).strip() or ts,
        "last_seen_at": str(normalized.get("last_seen_at", "")).strip() or ts,
        "session_id": str(normalized.get("session_id", "")).strip(),
        "intent": str(normalized.get("intent", "")).strip() or "unknown",
        "query": query,
        "query_prefix": query_prefix,
        "success": success,
        "examiner_verdict": str(normalized.get("examiner_verdict", "")).strip().upper(),
        "execution_success": _as_bool(normalized.get("execution_success", False)),
        "error_type": str(normalized.get("error_type", "")).strip() or "generic_failure",
        "symptom": _trim_text(str(normalized.get("symptom", "")), 240),
        "resolution_hint": _trim_text(str(normalized.get("resolution_hint", "")), 240),
        "artifact_paths": [
            str(path).strip()
            for path in list(normalized.get("artifact_paths") or [])[-4:]
            if str(path).strip()
        ],
        "tags": _normalize_string_list(list(normalized.get("tags") or []), limit=10),
        "occurrence_count": occurrence_count,
        "success_count": max(0, success_count),
        "failure_count": max(0, failure_count),
    }

    normalized_record["fingerprint"] = (
        str(normalized.get("fingerprint", "")).strip()
        or build_experience_fingerprint(normalized_record)
    )
    normalized_record["experience_score"] = int(
        normalized.get("experience_score", 0) or _compute_experience_score(normalized_record)
    )
    return normalized_record


def _merge_experience_record(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    current = _normalize_experience_record(existing)
    update = _normalize_experience_record(incoming)

    merged = dict(current)
    merged["ts"] = update["ts"]
    merged["last_seen_at"] = update["last_seen_at"] or update["ts"]
    merged["first_seen_at"] = min(
        str(current.get("first_seen_at", "")).strip() or update["first_seen_at"],
        str(update.get("first_seen_at", "")).strip() or current["first_seen_at"],
    )
    merged["occurrence_count"] = int(current.get("occurrence_count", 1)) + int(update.get("occurrence_count", 1))
    merged["success_count"] = int(current.get("success_count", 0)) + int(update.get("success_count", 0))
    merged["failure_count"] = int(current.get("failure_count", 0)) + int(update.get("failure_count", 0))

    for key in (
        "session_id",
        "intent",
        "query",
        "query_prefix",
        "examiner_verdict",
        "error_type",
        "symptom",
        "resolution_hint",
    ):
        if str(update.get(key, "")).strip():
            merged[key] = update[key]

    if update.get("artifact_paths"):
        merged["artifact_paths"] = list(update["artifact_paths"])

    merged["tags"] = _normalize_string_list(
        list(current.get("tags") or []) + list(update.get("tags") or []),
        limit=12,
    )
    merged["success"] = bool(update.get("success", current.get("success", False)))
    merged["execution_success"] = bool(
        update.get("execution_success", current.get("execution_success", False))
    )
    merged["fingerprint"] = current.get("fingerprint") or update.get("fingerprint") or build_experience_fingerprint(merged)
    merged["experience_score"] = _compute_experience_score(merged)
    return merged


def _write_experience_records(
    records: list[dict[str, Any]],
    path: str | Path = EXPERIENCE_DB_PATH,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_records = [_normalize_experience_record(record) for record in records if record]
    with open(file_path, "w", encoding="utf-8") as f:
        for record in normalized_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_experience_record(
    record: dict[str, Any],
    path: str | Path = EXPERIENCE_DB_PATH,
) -> dict[str, Any]:
    """Append or merge an experience record based on its fingerprint."""
    if not record:
        return {}

    normalized = _normalize_experience_record(record)
    records = load_experience_records(path=path)

    for index, existing in enumerate(records):
        if existing.get("fingerprint") == normalized["fingerprint"]:
            records[index] = _merge_experience_record(existing, normalized)
            _write_experience_records(records, path=path)
            return records[index]

    records.append(normalized)
    _write_experience_records(records, path=path)
    return normalized


def load_experience_records(path: str | Path = EXPERIENCE_DB_PATH) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    ordered_records: list[dict[str, Any]] = []
    index_by_fingerprint: dict[str, int] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_record, dict):
                continue

            normalized = _normalize_experience_record(raw_record)
            fingerprint = normalized["fingerprint"]
            if fingerprint in index_by_fingerprint:
                record_index = index_by_fingerprint[fingerprint]
                ordered_records[record_index] = _merge_experience_record(
                    ordered_records[record_index],
                    normalized,
                )
            else:
                index_by_fingerprint[fingerprint] = len(ordered_records)
                ordered_records.append(normalized)

    return ordered_records


def _derive_error_type(state: dict[str, Any]) -> str:
    stderr = str(state.get("execution_stderr", "")).lower()
    code_review = str(state.get("code_review", "")).lower()
    academic_review = str(state.get("academic_review", "")).lower()
    combined = "\n".join([stderr, code_review, academic_review])

    if "torch" in combined and "missing" in combined:
        return "missing_torch"
    if "modulenotfounderror" in combined:
        return "missing_dependency"
    if "syntaxerror" in combined or "indentationerror" in combined:
        return "syntax_error"
    if "timeout" in combined:
        return "timeout"
    if str(state.get("examiner_verdict", "")).upper() == "FAIL":
        return "examiner_fail"
    if state.get("execution_success", False):
        return "successful_run"
    return "generic_failure"


def build_experience_record(state: dict[str, Any]) -> dict[str, Any]:
    """Build a compact reusable experience record from the latest run."""
    query = _trim_text(str(state.get("query", "")), 300)
    if not query:
        return {}

    intent = str(state.get("intent", "")).strip() or "unknown"
    verdict = str(state.get("examiner_verdict", "")).strip().upper()
    execution_success = bool(state.get("execution_success", False))

    is_code_like = intent in {"code", "full_pipeline"}
    has_failure = verdict == "FAIL" or (is_code_like and not execution_success)
    has_meaningful_review = bool(
        str(state.get("code_review", "")).strip()
        or str(state.get("academic_review", "")).strip()
    )
    if not (is_code_like or has_failure or has_meaningful_review):
        return {}

    symptom = (
        _trim_text(str(state.get("execution_stderr", "")), 240)
        or _trim_text(str(state.get("code_review", "")), 240)
        or _trim_text(str(state.get("academic_review", "")), 240)
        or _trim_text(str(state.get("execution_stdout", "")), 240)
    )
    resolution_hint = (
        _trim_text(str(state.get("code_review", "")), 240)
        or _trim_text(str(state.get("academic_review", "")), 240)
        or _trim_text(str(state.get("design_proposal", "")), 240)
        or _trim_text(str(state.get("literature_report", "")), 240)
    )

    tags = [intent, _derive_error_type(state)]
    if execution_success:
        tags.append("execution_success")
    if verdict == "FAIL":
        tags.append("examiner_fail")
    if state.get("artifact_paths"):
        tags.append("has_artifacts")

    record = {
        "ts": _utc_now(),
        "session_id": str(state.get("session_id", "")).strip(),
        "intent": intent,
        "query": query,
        "query_prefix": _query_prefix(query),
        "success": verdict == "PASS" if verdict else execution_success,
        "examiner_verdict": verdict or "",
        "execution_success": execution_success,
        "error_type": _derive_error_type(state),
        "symptom": symptom,
        "resolution_hint": resolution_hint,
        "artifact_paths": [str(path) for path in list(state.get("artifact_paths") or [])[-4:]],
        "tags": sorted(set(tag for tag in tags if tag)),
        "occurrence_count": 1,
        "success_count": 1 if (verdict == "PASS" if verdict else execution_success) else 0,
        "failure_count": 0 if (verdict == "PASS" if verdict else execution_success) else 1,
    }
    return _normalize_experience_record(record)


def retrieve_experience_hints(
    query: str,
    intent: str = "",
    limit: int = EXPERIENCE_HINT_LIMIT,
    path: str | Path = EXPERIENCE_DB_PATH,
) -> list[dict[str, Any]]:
    records = load_experience_records(path=path)
    if not records:
        return []

    query_tokens = _tokenize(query)
    query_prefix = _query_prefix(query)
    scored: list[tuple[int, int, int, dict[str, Any]]] = []

    for index, record in enumerate(records):
        haystack = " ".join(
            str(record.get(key, ""))
            for key in ("query", "query_prefix", "symptom", "resolution_hint", "error_type", "intent")
        ).lower()
        lexical_overlap = sum(1 for token in query_tokens if token and token in haystack)
        score = lexical_overlap * 2

        if intent and str(record.get("intent", "")).strip() == intent:
            score += 3
        if query_prefix and query_prefix == str(record.get("query_prefix", "")).strip():
            score += 3
        if query_prefix and query_prefix in haystack:
            score += 1

        score += min(int(record.get("occurrence_count", 1) or 1), 4)
        score += min(int(record.get("experience_score", 0) or 0), 6)

        if str(record.get("resolution_hint", "")).strip():
            score += 1
        if list(record.get("artifact_paths") or []):
            score += 1

        if score > 0:
            enriched = dict(record)
            enriched["retrieval_score"] = score
            scored.append(
                (
                    score,
                    int(record.get("experience_score", 0) or 0),
                    index,
                    enriched,
                )
            )

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [record for _, _, _, record in scored[:limit]]


def format_experience_hints(hints: list[dict[str, Any]]) -> str:
    if not hints:
        return ""

    lines: list[str] = []
    for hint in hints[:EXPERIENCE_HINT_LIMIT]:
        error_type = str(hint.get("error_type", "record")).strip()
        symptom = _trim_text(str(hint.get("symptom", "")), 120)
        resolution = _trim_text(str(hint.get("resolution_hint", "")), 120)
        occurrence = max(1, int(hint.get("occurrence_count", 1) or 1))
        score = int(hint.get("experience_score", 0) or 0)
        parts = [f"- {error_type} (seen={occurrence}x, score={score})"]
        if symptom:
            parts.append(f"symptom: {symptom}")
        if resolution:
            parts.append(f"resolution: {resolution}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
