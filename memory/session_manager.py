"""Session-scoped memory persistence helpers."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import SESSION_MEMORY_DIR

MAX_RECENT_QUERIES = 6
SUMMARY_TEXT_LIMIT = 500
MAX_ARTIFACTS = 6
MAX_DIGEST_LINES = 8
CONVERSATION_DIGEST_LIMIT = 1200
CODE_SNIPPET_CHAR_LIMIT = 900
CODE_SNIPPET_LINE_LIMIT = 24


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(text: str, limit: int = SUMMARY_TEXT_LIMIT) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _trim_tail_text(text: str, limit: int) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return "..." + normalized[-(limit - 3) :].lstrip()


def _normalize_string_list(values: list[Any], limit: int = MAX_RECENT_QUERIES) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = _trim_text(str(value), 160)
        if text and text not in normalized:
            normalized.append(text)
    return normalized[-limit:]


def _build_code_summary(state: dict[str, Any], previous: dict[str, Any]) -> str:
    code = str(state.get("generated_code", "")).strip()
    if not code:
        return str(previous.get("last_code_summary", ""))

    status = "success" if state.get("execution_success", False) else "failed"
    stdout = _trim_text(str(state.get("execution_stdout", "")), 180)
    stderr = _trim_text(str(state.get("execution_stderr", "")), 180)
    artifacts = list(state.get("artifact_paths") or [])

    parts = [f"Execution {status}."]
    if stdout:
        parts.append(f"stdout: {stdout}")
    if stderr:
        parts.append(f"stderr: {stderr}")
    if artifacts:
        artifact_names = ", ".join(Path(path).name for path in artifacts[-3:])
        parts.append(f"artifacts: {artifact_names}")
    return " ".join(parts)


def _build_examiner_summary(state: dict[str, Any], previous: dict[str, Any]) -> str:
    verdict = str(state.get("examiner_verdict", "")).strip()
    academic = _trim_text(str(state.get("academic_review", "")), 220)
    code_review = _trim_text(str(state.get("code_review", "")), 220)

    parts = []
    if verdict:
        parts.append(f"Verdict: {verdict}.")
    if academic:
        parts.append(f"Academic: {academic}")
    if code_review:
        parts.append(f"Code: {code_review}")

    if parts:
        return " ".join(parts)
    return str(previous.get("last_examiner_summary", ""))


def default_session_summary(session_id: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "user_goal": "",
        "constraints": [],
        "recent_queries": [],
        "last_intent": "",
        "last_research_summary": "",
        "last_code_summary": "",
        "last_examiner_summary": "",
        "last_artifacts": [],
        "open_todos": [],
        "conversation_digest": "",
        "compressed_turns": 0,
        "message_window_size": 0,
        "last_history_compression_at": "",
        "last_code_snippet": "",
        "last_error_summary": "",
        "last_successful_code_snippet": "",
        "last_failed_code_snippet": "",
        "last_successful_artifacts": [],
        "last_failure_error_summary": "",
    }


def build_session_summary(
    previous_summary: dict[str, Any] | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Merge the latest run state into a persistent session summary."""
    previous = deepcopy(previous_summary or {})
    session_id = str(
        state.get("session_id")
        or previous.get("session_id")
        or ""
    ).strip()
    summary = default_session_summary(session_id)
    summary.update(previous)
    summary["session_id"] = session_id
    summary["updated_at"] = _utc_now()
    summary.setdefault("created_at", summary["updated_at"])

    query = str(state.get("query", "")).strip()
    recent_queries = list(summary.get("recent_queries") or [])
    if query:
        recent_queries.append(query)
    summary["recent_queries"] = _normalize_string_list(recent_queries)

    if query and not str(summary.get("user_goal", "")).strip():
        summary["user_goal"] = _trim_text(query, 200)

    if state.get("intent"):
        summary["last_intent"] = str(state["intent"])

    summary["message_window_size"] = len(list(state.get("messages") or []))

    literature_report = str(state.get("literature_report", "")).strip()
    if literature_report:
        summary["last_research_summary"] = _trim_text(literature_report)

    summary["last_code_summary"] = _build_code_summary(state, summary)
    summary["last_examiner_summary"] = _build_examiner_summary(state, summary)
    summary["last_code_snippet"] = _build_code_snippet(state, summary)
    summary["last_error_summary"] = _build_error_summary(state, summary)
    summary["last_successful_code_snippet"] = _build_successful_code_snippet(state, summary)
    summary["last_failed_code_snippet"] = _build_failed_code_snippet(state, summary)
    summary["last_successful_artifacts"] = _build_successful_artifacts(state, summary)
    summary["last_failure_error_summary"] = _build_failure_error_summary(state, summary)

    artifacts = list(state.get("artifact_paths") or [])
    if artifacts:
        summary["last_artifacts"] = [str(path) for path in artifacts[-MAX_ARTIFACTS:]]

    open_todos: list[str] = []
    if str(state.get("examiner_verdict", "")).upper() == "FAIL":
        open_todos.append("Address the latest examiner findings before the next run.")
    if not state.get("execution_success", True) and str(state.get("execution_stderr", "")).strip():
        open_todos.append(
            "Fix the latest execution failure: "
            + _trim_text(str(state.get("execution_stderr", "")), 180)
        )
    summary["open_todos"] = _normalize_string_list(open_todos, limit=4)

    constraints = list(summary.get("constraints") or [])
    summary["constraints"] = _normalize_string_list(constraints, limit=8)
    return summary


def format_session_summary(summary: dict[str, Any]) -> str:
    """Render session memory into a compact prompt-friendly block."""
    if not summary:
        return ""

    lines: list[str] = []
    user_goal = str(summary.get("user_goal", "")).strip()
    if user_goal:
        lines.append(f"- User goal: {user_goal}")

    recent_queries = [str(item) for item in summary.get("recent_queries") or [] if str(item).strip()]
    if recent_queries:
        joined_queries = " | ".join(recent_queries[-3:])
        lines.append(f"- Recent queries: {joined_queries}")

    last_research = str(summary.get("last_research_summary", "")).strip()
    if last_research:
        lines.append(f"- Last research summary: {last_research}")

    last_code = str(summary.get("last_code_summary", "")).strip()
    if last_code:
        lines.append(f"- Last code summary: {last_code}")

    last_examiner = str(summary.get("last_examiner_summary", "")).strip()
    if last_examiner:
        lines.append(f"- Last examiner summary: {last_examiner}")

    open_todos = [str(item) for item in summary.get("open_todos") or [] if str(item).strip()]
    if open_todos:
        lines.append(f"- Open TODOs: {' | '.join(open_todos[:3])}")

    conversation_digest = str(summary.get("conversation_digest", "")).strip()
    if conversation_digest:
        lines.append(f"- Compressed history: {conversation_digest}")

    return "\n".join(lines)


def format_code_memory(summary: dict[str, Any]) -> str:
    """Render code-centric short-term memory for coder/examiner agents."""
    if not summary:
        return ""

    lines: list[str] = []

    last_code_summary = str(summary.get("last_code_summary", "")).strip()
    if last_code_summary:
        lines.append(f"- Last code run: {last_code_summary}")

    last_error_summary = str(summary.get("last_error_summary", "")).strip()
    if last_error_summary:
        lines.append(f"- Last real error: {last_error_summary}")

    snippet = str(summary.get("last_code_snippet", "")).strip()
    if snippet:
        lines.append("- Last code snippet:")
        lines.append("```python")
        lines.append(snippet)
        lines.append("```")

    successful_snippet = str(summary.get("last_successful_code_snippet", "")).strip()
    if successful_snippet:
        lines.append("- Last successful baseline:")
        lines.append("```python")
        lines.append(successful_snippet)
        lines.append("```")

    failed_snippet = str(summary.get("last_failed_code_snippet", "")).strip()
    if failed_snippet:
        lines.append("- Last failed attempt:")
        lines.append("```python")
        lines.append(failed_snippet)
        lines.append("```")

    last_failure_error = str(summary.get("last_failure_error_summary", "")).strip()
    if last_failure_error:
        lines.append(f"- Last failure summary: {last_failure_error}")

    successful_artifacts = [
        str(item).strip()
        for item in summary.get("last_successful_artifacts") or []
        if str(item).strip()
    ]
    if successful_artifacts:
        artifact_names = ", ".join(
            Path(path).name for path in successful_artifacts[-4:]
        )
        lines.append(f"- Last successful artifacts: {artifact_names}")

    artifacts = [str(item).strip() for item in summary.get("last_artifacts") or [] if str(item).strip()]
    if artifacts:
        artifact_names = ", ".join(Path(path).name for path in artifacts[-4:])
        lines.append(f"- Last artifacts: {artifact_names}")

    return "\n".join(lines)


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    return message.__class__.__name__.replace("Message", "").lower()


def _build_code_snippet(state: dict[str, Any], previous: dict[str, Any]) -> str:
    code = str(state.get("generated_code", "")).strip()
    if not code:
        return str(previous.get("last_code_snippet", ""))

    lines = code.splitlines()
    if len(lines) > CODE_SNIPPET_LINE_LIMIT:
        lines = ["# ... older code omitted ..."] + lines[-CODE_SNIPPET_LINE_LIMIT:]
    snippet = "\n".join(lines).strip()
    if len(snippet) > CODE_SNIPPET_CHAR_LIMIT:
        snippet = "# ... older code omitted ...\n" + snippet[-(CODE_SNIPPET_CHAR_LIMIT - 29) :].lstrip()
    return snippet


def _build_successful_code_snippet(
    state: dict[str, Any],
    previous: dict[str, Any],
) -> str:
    code = str(state.get("generated_code", "")).strip()
    if code and bool(state.get("execution_success", False)):
        return _build_code_snippet(state, previous)
    return str(previous.get("last_successful_code_snippet", ""))


def _build_failed_code_snippet(
    state: dict[str, Any],
    previous: dict[str, Any],
) -> str:
    code = str(state.get("generated_code", "")).strip()
    verdict = str(state.get("examiner_verdict", "")).upper()
    if code and (
        not bool(state.get("execution_success", False))
        or verdict == "FAIL"
    ):
        return _build_code_snippet(state, previous)
    return str(previous.get("last_failed_code_snippet", ""))


def _build_successful_artifacts(
    state: dict[str, Any],
    previous: dict[str, Any],
) -> list[str]:
    artifacts = [str(path) for path in list(state.get("artifact_paths") or []) if str(path).strip()]
    if artifacts and bool(state.get("execution_success", False)):
        return artifacts[-MAX_ARTIFACTS:]
    return list(previous.get("last_successful_artifacts") or [])


def _build_failure_error_summary(
    state: dict[str, Any],
    previous: dict[str, Any],
) -> str:
    stderr = str(state.get("execution_stderr", "")).strip()
    if stderr and not bool(state.get("execution_success", False)):
        return _trim_text(stderr, 220)

    code_review = str(state.get("code_review", "")).strip()
    if code_review and str(state.get("examiner_verdict", "")).upper() == "FAIL":
        return _trim_text(code_review, 220)

    return str(previous.get("last_failure_error_summary", ""))


def _build_error_summary(state: dict[str, Any], previous: dict[str, Any]) -> str:
    stderr = str(state.get("execution_stderr", "")).strip()
    if stderr and not bool(state.get("execution_success", False)):
        return _trim_text(stderr, 220)

    code_review = str(state.get("code_review", "")).strip()
    if code_review and str(state.get("examiner_verdict", "")).upper() == "FAIL":
        return _trim_text(code_review, 220)

    return str(previous.get("last_error_summary", ""))


def _message_content_to_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_block = item.get("text") or item.get("content") or item.get("type")
                if text_block:
                    parts.append(str(text_block))
        text = " ".join(parts)
    else:
        text = str(content)

    text = " ".join(text.split())
    if text:
        return text

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return f"requested {len(tool_calls)} tool call(s)"

    tool_name = getattr(message, "name", None)
    if tool_name:
        return f"result from tool {tool_name}"

    return ""


def _build_history_digest_lines(messages: list[BaseMessage]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = _message_role(message)
        if role == "system":
            continue

        text = _trim_text(_message_content_to_text(message), 140)
        if not text:
            continue

        line = f"{role}: {text}"
        if line not in lines:
            lines.append(line)

    if len(lines) <= MAX_DIGEST_LINES:
        return lines

    head = lines[: MAX_DIGEST_LINES // 2]
    tail = lines[-(MAX_DIGEST_LINES // 2) :]
    omitted = len(lines) - len(head) - len(tail)
    return head + [f"... {omitted} older internal message(s) omitted ..."] + tail


def compress_message_history(
    messages: list[BaseMessage] | None,
    session_summary: dict[str, Any] | None,
) -> tuple[list[BaseMessage], dict[str, Any], bool]:
    """
    Compress prior-turn raw LangGraph messages into session summary text.

    Design choice:
    - At the start of a new user turn, old internal messages are no longer
      passed forward verbatim.
    - Instead, they are summarized into `conversation_digest` and the raw
      message window is cleared.
    """
    history = [msg for msg in list(messages or []) if isinstance(msg, BaseMessage)]
    summary = deepcopy(session_summary or {})

    if not history:
        summary.setdefault("conversation_digest", "")
        summary["message_window_size"] = 0
        return [], summary, False

    digest_lines = _build_history_digest_lines(history)
    previous_digest = str(summary.get("conversation_digest", "")).strip()

    digest_parts: list[str] = []
    if previous_digest:
        digest_parts.append(previous_digest)
    digest_parts.append(
        f"Compressed {len(history)} internal message(s) from the previous turn."
    )
    digest_parts.extend(digest_lines)

    summary["conversation_digest"] = _trim_tail_text(
        "\n".join(part for part in digest_parts if part),
        CONVERSATION_DIGEST_LIMIT,
    )
    summary["compressed_turns"] = int(summary.get("compressed_turns", 0)) + 1
    summary["message_window_size"] = 0
    summary["last_history_compression_at"] = _utc_now()
    return [], summary, True


class SessionManager:
    """Manage session identifiers and summary files."""

    def __init__(self, session_dir: str | Path = SESSION_MEMORY_DIR):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def create_session_id(self, prefix: str = "session") -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"

    def default_summary(self, session_id: str) -> dict[str, Any]:
        return default_session_summary(session_id)

    def get_session_path(self, session_id: str) -> Path:
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        return self.session_dir / f"{safe_session_id}.json"

    def ensure_session(self, session_id: str | None = None, prefix: str = "session") -> str:
        if session_id and str(session_id).strip():
            return str(session_id).strip()
        return self.reset_session(prefix=prefix)

    def reset_session(self, prefix: str = "session") -> str:
        session_id = self.create_session_id(prefix=prefix)
        self.save_summary(session_id, self.default_summary(session_id))
        return session_id

    def load_summary(self, session_id: str) -> dict[str, Any]:
        path = self.get_session_path(session_id)
        if not path.exists():
            return self.default_summary(session_id)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return self.default_summary(session_id)

        if not isinstance(data, dict):
            return self.default_summary(session_id)

        summary = self.default_summary(session_id)
        summary.update(data)
        summary["session_id"] = session_id
        return summary

    def save_summary(self, session_id: str, summary: dict[str, Any]) -> Path:
        payload = deepcopy(summary or {})
        payload["session_id"] = session_id
        payload.setdefault("created_at", _utc_now())
        payload["updated_at"] = _utc_now()

        path = self.get_session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
