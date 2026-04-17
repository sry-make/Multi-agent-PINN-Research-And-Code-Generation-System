"""Project-level long-term memory helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROJECT_MEMORY_PATH

PROJECT_MEMORY_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(text: str, limit: int = 220) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _normalize_string_list(values: list[Any], limit: int = 8) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = _trim_text(str(value), 180)
        if text and text not in normalized:
            normalized.append(text)
    return normalized[:limit]


def default_project_memory() -> dict[str, Any]:
    now = _utc_now()
    return {
        "memory_version": PROJECT_MEMORY_VERSION,
        "updated_at": now,
        "project_name": "PINN Agent V2",
        "goal": (
            "Build a resume-friendly multi-agent research and coding assistant "
            "focused on PINNs, with clear workflow visibility and practical demos."
        ),
        "architecture_rules": [
            "Use LangGraph for workflow orchestration.",
            "Use Textual TUI as the main interface.",
            "Keep code execution sandboxed and artifact export visible.",
            "Prefer explainable engineering decisions over premature platform complexity.",
        ],
        "tech_stack": [
            "Python",
            "LangGraph",
            "LangChain",
            "Textual",
            "ChromaDB",
            "Docker",
        ],
        "current_priorities": [
            "Stabilize short-term session memory.",
            "Keep the demo path easy to understand.",
            "Show multi-agent collaboration in a structured way.",
        ],
        "known_risks": [
            "Sandbox boundaries must remain consistent across code and shell tools.",
            "Long-term project facts and experience memory are still evolving.",
            "Heavy local model prompts can stall the TUI if outputs grow too large.",
        ],
        "coding_preferences": [
            "Prefer lightweight local-first persistence.",
            "Favor structured outputs that are easy to review in the TUI.",
            "Keep engineering tradeoffs explicit for interview storytelling.",
        ],
        "decisions": [
            {
                "id": "workflow-langgraph",
                "summary": "Use LangGraph as the workflow orchestrator.",
                "rationale": "The project needs explicit SOP routing, retries, and memory read/write nodes.",
                "status": "accepted",
            },
            {
                "id": "ui-textual-tui",
                "summary": "Use Textual TUI as the primary user interface.",
                "rationale": "It is easier to demo in terminals and better fits trace-heavy agent workflows than a browser-first MVP.",
                "status": "accepted",
            },
        ],
        "rejected_options": [
            {
                "option": "Use Streamlit as the primary interface.",
                "reason": "The TUI makes session state, trace logs, and artifacts easier to present in interviews and demos.",
            },
            {
                "option": "Add multi-backend vector databases before the MVP stabilizes.",
                "reason": "That would add platform complexity without improving the resume-focused learning path.",
            },
        ],
    }


def _normalize_decisions(items: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for index, item in enumerate(items):
        if isinstance(item, dict):
            summary = _trim_text(str(item.get("summary", "")), 180)
            rationale = _trim_text(str(item.get("rationale", "")), 220)
            status = _trim_text(str(item.get("status", "")), 40) or "accepted"
            identifier = _trim_text(str(item.get("id", "")), 80) or f"decision-{index + 1}"
        else:
            summary = _trim_text(str(item), 180)
            rationale = ""
            status = "accepted"
            identifier = f"decision-{index + 1}"

        if not summary:
            continue

        payload = {
            "id": identifier,
            "summary": summary,
            "rationale": rationale,
            "status": status,
        }
        if payload not in normalized:
            normalized.append(payload)

    return normalized[:6]


def _normalize_rejected_options(items: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for item in items:
        if isinstance(item, dict):
            option = _trim_text(str(item.get("option", "")), 180)
            reason = _trim_text(str(item.get("reason", "")), 220)
        else:
            option = _trim_text(str(item), 180)
            reason = ""

        if not option:
            continue

        payload = {
            "option": option,
            "reason": reason,
        }
        if payload not in normalized:
            normalized.append(payload)

    return normalized[:6]


def normalize_project_memory(project_memory: dict[str, Any] | None) -> dict[str, Any]:
    base = default_project_memory()
    incoming = deepcopy(project_memory or {})
    if isinstance(incoming, dict):
        base.update(incoming)

    base["memory_version"] = int(base.get("memory_version", PROJECT_MEMORY_VERSION) or PROJECT_MEMORY_VERSION)
    base["updated_at"] = str(base.get("updated_at", "")).strip() or _utc_now()
    base["project_name"] = _trim_text(str(base.get("project_name", "")), 120) or "PINN Agent V2"
    base["goal"] = _trim_text(str(base.get("goal", "")), 260)
    base["architecture_rules"] = _normalize_string_list(list(base.get("architecture_rules") or []), limit=6)
    base["tech_stack"] = _normalize_string_list(list(base.get("tech_stack") or []), limit=8)
    base["current_priorities"] = _normalize_string_list(list(base.get("current_priorities") or []), limit=6)
    base["known_risks"] = _normalize_string_list(list(base.get("known_risks") or []), limit=6)
    base["coding_preferences"] = _normalize_string_list(list(base.get("coding_preferences") or []), limit=6)
    base["decisions"] = _normalize_decisions(list(base.get("decisions") or []))
    base["rejected_options"] = _normalize_rejected_options(list(base.get("rejected_options") or []))
    return base


def load_project_memory(path: str | Path = PROJECT_MEMORY_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return normalize_project_memory({})

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return normalize_project_memory({})

    if not isinstance(data, dict):
        return normalize_project_memory({})

    return normalize_project_memory(data)


def save_project_memory(
    project_memory: dict[str, Any],
    path: str | Path = PROJECT_MEMORY_PATH,
) -> Path:
    payload = normalize_project_memory(project_memory)
    payload["updated_at"] = _utc_now()

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return file_path


def record_project_decision(
    project_memory: dict[str, Any],
    summary: str,
    rationale: str = "",
    *,
    decision_id: str = "",
    status: str = "accepted",
) -> dict[str, Any]:
    payload = normalize_project_memory(project_memory)
    summary = _trim_text(summary, 180)
    if not summary:
        return payload

    decision = {
        "id": _trim_text(decision_id, 80) or f"decision-{len(payload['decisions']) + 1}",
        "summary": summary,
        "rationale": _trim_text(rationale, 220),
        "status": _trim_text(status, 40) or "accepted",
    }

    existing_decisions = [item for item in payload.get("decisions", []) if item.get("id") != decision["id"]]
    existing_decisions.append(decision)
    payload["decisions"] = _normalize_decisions(existing_decisions)
    payload["updated_at"] = _utc_now()
    return payload


def record_rejected_option(
    project_memory: dict[str, Any],
    option: str,
    reason: str = "",
) -> dict[str, Any]:
    payload = normalize_project_memory(project_memory)
    option = _trim_text(option, 180)
    if not option:
        return payload

    rejected = {
        "option": option,
        "reason": _trim_text(reason, 220),
    }
    payload["rejected_options"] = _normalize_rejected_options(
        list(payload.get("rejected_options") or []) + [rejected]
    )
    payload["updated_at"] = _utc_now()
    return payload


def format_project_memory(project_memory: dict[str, Any]) -> str:
    if not project_memory:
        return ""

    memory = normalize_project_memory(project_memory)
    lines: list[str] = [
        f"- Project memory version: {memory['memory_version']}",
        f"- Project updated at: {memory['updated_at']}",
    ]

    goal = str(memory.get("goal", "")).strip()
    if goal:
        lines.append(f"- Project goal: {goal}")

    for label, key, limit in (
        ("Architecture rules", "architecture_rules", 4),
        ("Current priorities", "current_priorities", 3),
        ("Known risks", "known_risks", 3),
        ("Coding preferences", "coding_preferences", 3),
    ):
        items = [str(item).strip() for item in memory.get(key) or [] if str(item).strip()]
        if items:
            lines.append(f"- {label}: {' | '.join(items[:limit])}")

    decisions = [
        item
        for item in memory.get("decisions") or []
        if isinstance(item, dict) and str(item.get("summary", "")).strip()
    ]
    if decisions:
        summarized = []
        for item in decisions[:3]:
            summary = str(item.get("summary", "")).strip()
            status = str(item.get("status", "")).strip()
            summarized.append(f"{summary} ({status})" if status else summary)
        lines.append(f"- Key decisions: {' | '.join(summarized)}")

    rejected_options = [
        item
        for item in memory.get("rejected_options") or []
        if isinstance(item, dict) and str(item.get("option", "")).strip()
    ]
    if rejected_options:
        summarized = [str(item.get("option", "")).strip() for item in rejected_options[:3]]
        lines.append(f"- Rejected options: {' | '.join(summarized)}")

    return "\n".join(lines)
