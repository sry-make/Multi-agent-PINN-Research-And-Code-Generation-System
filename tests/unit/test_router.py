from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestrator.router import _rule_based_intent, detect_intent, route_by_intent


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


def test_route_by_intent_returns_expected_node() -> None:
    assert route_by_intent({"intent": "qa"}) == "researcher"
    assert route_by_intent({"intent": "survey"}) == "researcher"
    assert route_by_intent({"intent": "code"}) == "coder"
    assert route_by_intent({"intent": "full_pipeline"}) == "researcher"
    assert route_by_intent({}) == "researcher"
