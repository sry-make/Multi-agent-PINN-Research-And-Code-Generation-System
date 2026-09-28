from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from orchestrator.router import (
    _rule_based_intent,
    detect_intent,
    looks_like_contextual_followup,
    resolve_intent,
    route_by_intent,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("帮我写一段 PINN 代码", "code"),
        ("综述最新 PINN 进展", "survey"),
        ("PINN 的损失函数是什么", "qa"),
        ("写一个 PINN 代码并综述其理论基础", "full_pipeline"),
    ],
)
def test_rule_based_intent(query: str, expected: str) -> None:
    assert _rule_based_intent(query) == expected


def test_detect_intent_falls_back_to_rule_based_logic() -> None:
    with patch(
        "orchestrator.router._llm.chat.completions.create",
        side_effect=RuntimeError("router llm unavailable"),
    ):
        assert detect_intent("帮我写一段 PINN 代码") == "code"


def test_resolve_intent_uses_rule_router_for_explicit_patterns() -> None:
    decision = resolve_intent("先综述 PINN 边界损失，再写一段最小代码")

    assert decision.intent == "full_pipeline"
    assert decision.source == "rule"
    assert decision.confidence >= 0.95
    assert decision.needs_clarification is False


def test_resolve_intent_enters_clarify_on_flat_llm_distribution() -> None:
    payload = {
        "intent": "qa",
        "scores": {
            "qa": 0.33,
            "survey": 0.27,
            "code": 0.21,
            "full_pipeline": 0.19,
        },
        "reason": "用户只说了帮我看一下，目标不够明确。",
        "missing_information": ["是想解释概念还是直接写代码"],
        "clarification_question": "你是希望我先解释一下，还是直接写并运行代码？",
    }
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=str(payload).replace("'", '"')))]
    )

    with patch("orchestrator.router._llm.chat.completions.create", return_value=mock_response):
        decision = resolve_intent("帮我看一下 PINN")

    assert decision.intent == "clarify"
    assert decision.source == "llm_clarify"
    assert decision.needs_clarification is True
    assert "写并运行代码" in decision.clarification_question
    assert decision.default_intent == "qa"


def test_looks_like_contextual_followup_detects_short_or_contextual_queries() -> None:
    assert looks_like_contextual_followup("继续上次那个") is True
    assert looks_like_contextual_followup("这个呢") is True
    assert looks_like_contextual_followup("请综述 PINN 最新进展") is False


def test_route_by_intent_returns_expected_node() -> None:
    assert route_by_intent({"intent": "clarify"}) == "clarify"
    assert route_by_intent({"intent": "qa"}) == "researcher"
    assert route_by_intent({"intent": "survey"}) == "researcher"
    assert route_by_intent({"intent": "code"}) == "coder"
    assert route_by_intent({"intent": "full_pipeline"}) == "researcher"
    assert route_by_intent({}) == "researcher"
