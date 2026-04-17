"""
Tracer — 每步 Prompt / Tool 调用 / CoT 追踪

将每次 Agent 调用记录为一条 JSONL，供离线分析和 TUI Debug 面板展示。
格式兼容 OpenTelemetry Span 概念（无需引入重型 OTEL 依赖）。
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import TRACE_ENABLED, TRACE_LOG_DIR, LANGSMITH_API_KEY


class Tracer:
    """轻量级 JSONL 追踪器（单例模式）"""

    _instance: Tracer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.enabled = TRACE_ENABLED
        if not self.enabled:
            return

        log_dir = Path(TRACE_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._log_path = log_dir / f"trace_{date_str}.jsonl"
        self._session_id = str(uuid.uuid4())[:8]

        # 可选：LangSmith 上报
        self._langsmith = None
        if LANGSMITH_API_KEY:
            try:
                from langsmith import Client
                self._langsmith = Client(api_key=LANGSMITH_API_KEY)
            except ImportError:
                pass

    # ── 公共 API ──────────────────────────────────────────────

    def log_llm_call(
        self,
        agent: str,
        prompt: str,
        response: str,
        model: str,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录一次 LLM 调用"""
        self._write({
            "type":        "llm_call",
            "agent":       agent,
            "model":       model,
            "prompt":      prompt[:2000],        # 截断超长 prompt
            "response":    response[:2000],
            "tokens_used": tokens_used,
            "duration_ms": round(duration_ms, 2),
            **(metadata or {}),
        })

    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        duration_ms: float = 0.0,
    ) -> None:
        """记录一次 Tool 调用"""
        self._write({
            "type":        "tool_call",
            "agent":       agent,
            "tool":        tool_name,
            "input":       tool_input,
            "output":      str(tool_output)[:1000],
            "duration_ms": round(duration_ms, 2),
        })

    def log_state_transition(
        self,
        from_step: str,
        to_step: str,
        intent: str = "",
    ) -> None:
        """记录 SOP 状态转移"""
        self._write({
            "type":      "state_transition",
            "from_step": from_step,
            "to_step":   to_step,
            "intent":    intent,
        })

    def log_examiner_verdict(
        self,
        verdict: str,
        review: str,
        retry_count: int,
    ) -> None:
        """记录 Examiner 审查结果"""
        self._write({
            "type":        "examiner_verdict",
            "verdict":     verdict,
            "review":      review[:500],
            "retry_count": retry_count,
        })

    @property
    def session_id(self) -> str:
        """当前 tracer 会话 ID。"""
        return getattr(self, "_session_id", "")

    @property
    def log_path(self) -> Path | None:
        """当前 JSONL 日志路径。"""
        return getattr(self, "_log_path", None)

    def read_session_records(self, start_index: int = 0) -> list[dict[str, Any]]:
        """
        读取当前 tracer 会话的 JSONL 记录。

        Args:
            start_index: 跳过前 N 条已消费记录，适合 TUI 轮询增量读取

        Returns:
            当前 session 的记录列表
        """
        if not self.enabled or not self.session_id:
            return []

        path = self.log_path
        if path is None or not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("session_id") == self.session_id:
                    records.append(record)

        return records[start_index:]

    def read_session_records_from_offset(
        self,
        start_offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """
        从日志文件偏移量开始增量读取当前 session 的记录。

        适合 TUI 轮询场景，避免每次都重扫整份 JSONL 文件。
        """
        if not self.enabled or not self.session_id:
            return start_offset, []

        path = self.log_path
        if path is None or not path.exists():
            return start_offset, []

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            f.seek(start_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("session_id") == self.session_id:
                    records.append(record)
            next_offset = f.tell()

        return next_offset, records

    # ── 内部写入 ──────────────────────────────────────────────

    def _write(self, data: dict[str, Any]) -> None:
        if not self.enabled:
            return

        record = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            **data,
        }

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 便捷计时上下文管理器 ──────────────────────────────────────

class timer:
    """用法: with timer() as t: ...; print(t.ms)"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._start) * 1000

    @property
    def seconds(self) -> float:
        return self.ms / 1000


# 全局单例
tracer = Tracer()
