"""
Token 成本控制器

职责:
    1. 统计每次 LLM 调用的 Token 消耗
    2. 检查是否超出会话/单次预算
    3. 提供降级策略建议
    4. 向 TUI 状态栏推送实时数据
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from config import (
    TOKEN_BUDGET_PER_SESSION,
    TOKEN_BUDGET_PER_CALL,
    TOKEN_WARN_THRESHOLD,
    TOKEN_OVERBUDGET_STRATEGY,
)


class CostTracker:
    """单例 Token 成本追踪器"""

    _instance: CostTracker | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset()
        return cls._instance

    def _reset(self):
        self._session_tokens = 0
        self._call_history: list[dict] = []   # [{agent, tokens, model}, ...]

    # ── 公共 API ──────────────────────────────────────────────

    def record(self, agent: str, model: str, tokens: int) -> None:
        """记录一次调用的 Token 消耗"""
        self._session_tokens += tokens
        self._call_history.append({
            "agent":  agent,
            "model":  model,
            "tokens": tokens,
        })

    def check_call_budget(self, estimated_tokens: int) -> bool:
        """
        检查单次调用预算。
        Returns:
            True  → 预算充足，可以调用
            False → 超出单次上限，应降级
        """
        return estimated_tokens <= TOKEN_BUDGET_PER_CALL

    def check_session_budget(self) -> tuple[bool, str]:
        """
        检查会话累计预算。
        Returns:
            (within_budget, strategy)
            - within_budget: True → 仍在预算内
            - strategy: 超出时的降级策略字符串
        """
        if self._session_tokens >= TOKEN_BUDGET_PER_SESSION:
            return False, TOKEN_OVERBUDGET_STRATEGY
        return True, ""

    def is_warning(self) -> bool:
        """是否已达到预警阈值（TUI 状态栏变色提示）"""
        return self._session_tokens >= TOKEN_BUDGET_PER_SESSION * TOKEN_WARN_THRESHOLD

    def reset_session(self) -> None:
        """开始新会话时重置"""
        self._reset()

    # ── 状态读取（供 TUI 状态栏使用）────────────────────────

    @property
    def session_tokens(self) -> int:
        return self._session_tokens

    @property
    def session_budget(self) -> int:
        return TOKEN_BUDGET_PER_SESSION

    @property
    def usage_percent(self) -> float:
        return self._session_tokens / TOKEN_BUDGET_PER_SESSION * 100

    @property
    def summary(self) -> str:
        """TUI 状态栏显示的单行摘要"""
        warn = " ⚠️" if self.is_warning() else ""
        return f"Tokens: {self._session_tokens:,} / {TOKEN_BUDGET_PER_SESSION:,}{warn}"

    def per_agent_breakdown(self) -> dict[str, int]:
        """各 Agent Token 消耗分布"""
        breakdown: dict[str, int] = {}
        for record in self._call_history:
            breakdown[record["agent"]] = breakdown.get(record["agent"], 0) + record["tokens"]
        return breakdown

    def per_model_breakdown(self) -> dict[str, int]:
        """各模型 Token 消耗分布"""
        breakdown: dict[str, int] = {}
        for record in self._call_history:
            model = str(record.get("model", "")).strip() or "unknown"
            breakdown[model] = breakdown.get(model, 0) + int(record.get("tokens", 0) or 0)
        return breakdown

    def get_call_history(self) -> list[dict[str, Any]]:
        """返回调用明细的只读快照。"""
        return [deepcopy(record) for record in self._call_history]

    def snapshot(self) -> dict[str, Any]:
        """导出当前会话的成本快照，供 eval / 报表聚合。"""
        call_count = len(self._call_history)
        return {
            "total_tokens": self._session_tokens,
            "call_count": call_count,
            "avg_tokens_per_call": round(self._session_tokens / call_count, 2) if call_count else 0.0,
            "per_agent_tokens": self.per_agent_breakdown(),
            "per_model_tokens": self.per_model_breakdown(),
            "call_history": self.get_call_history(),
        }


# 全局单例
cost_tracker = CostTracker()
