from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.examiner import _rule_check_academic, _rule_check_code, run_examiner


pytestmark = pytest.mark.unit


def test_rule_check_academic_requires_substance_and_citation() -> None:
    ok_short, _ = _rule_check_academic("这是一个简短的回答")
    assert ok_short is False

    ok_valid, reason = _rule_check_academic(
        "这是一篇详细的文献综述，涵盖了 PINN 损失函数设计的核心方法，"
        "包含 [来源: paper.pdf] 的引用内容。"
    )
    assert ok_valid is True
    assert "通过" in reason


def test_rule_check_code_rejects_dangerous_or_failing_code() -> None:
    ok_empty, _ = _rule_check_code("", False, "")
    assert ok_empty is False

    ok_dangerous, reason_dangerous = _rule_check_code(
        "import os\nos.system('rm -rf /')",
        False,
        "",
    )
    assert ok_dangerous is False
    assert "危险操作" in reason_dangerous

    ok_runtime, reason_runtime = _rule_check_code(
        "import torch\nprint('ok')",
        False,
        "RuntimeError: shape mismatch",
    )
    assert ok_runtime is False
    assert "shape mismatch" in reason_runtime


def test_run_examiner_success_path_uses_fast_review() -> None:
    state = {
        "intent": "code",
        "current_step": "coder",
        "examiner_retry_count": 0,
        "generated_code": "import torch\nprint('ok')",
        "execution_success": True,
        "execution_stdout": "ok",
        "execution_stderr": "",
    }

    with patch(
        "agents.examiner._llm_review_code",
        side_effect=AssertionError("成功代码默认不应触发深度代码审查"),
    ):
        result = run_examiner(state)

    assert result["examiner_verdict"] == "PASS"
    assert "快速审查 PASS" in result["code_review"]


def test_run_examiner_empty_code_result_fails() -> None:
    state = {
        "intent": "code",
        "current_step": "coder",
        "examiner_retry_count": 0,
        "generated_code": "",
        "execution_success": False,
        "execution_stdout": "",
        "execution_stderr": "",
    }

    result = run_examiner(state)

    assert result["examiner_verdict"] == "FAIL"
    assert "未生成有效代码" in result["code_review"]
