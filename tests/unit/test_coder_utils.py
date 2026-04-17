from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.coder import _extract_code_block, _parse_execution_result, _react_loop


pytestmark = pytest.mark.unit


def test_extract_code_block_unwraps_execute_python_payload() -> None:
    tool_json = (
        '{"name": "execute_python", "arguments": '
        '{"code": "import torch\\nprint(1)"}}'
    )
    assert _extract_code_block(tool_json) == "import torch\nprint(1)"

    fenced_code = "```python\nimport torch\nprint('ok')\n```"
    assert "import torch" in _extract_code_block(fenced_code)


def test_parse_execution_result_extracts_stdout_stderr_and_artifacts() -> None:
    success, stdout, stderr, artifacts = _parse_execution_result(
        "[执行成功]\n"
        "[stdout]\nhello\n"
        "[stderr]\nwarning\n"
        "[artifacts]\noutputs/demo/result.png\noutputs/demo/train.log"
    )

    assert success is True
    assert stdout == "hello"
    assert stderr == "warning"
    assert artifacts == ["outputs/demo/result.png", "outputs/demo/train.log"]


def test_react_loop_keeps_executed_code_as_final_version(llm_response_factory) -> None:
    response_with_tool = llm_response_factory(
        content="",
        tool_calls=[
            {
                "name": "execute_python",
                "args": {"code": "print('sandbox ok')"},
                "id": "call-1",
            }
        ],
    )
    response_summary = llm_response_factory(
        content='{"name": "execute_python", "arguments": {"code": "print(\\"wrong wrapper\\")"}}',
        tool_calls=[],
    )

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=[response_with_tool, response_summary])

    mock_tool = MagicMock()
    mock_tool.invoke = MagicMock(
        return_value=(
            "[执行成功]\n"
            "[stdout]\nsandbox ok\n"
            "[artifacts]\n"
            "/home/kyrie/PINN_AGENT_PROJECT_V2/outputs/sandbox_runs/run_demo/result.png"
        )
    )

    with patch("agents.coder._build_llm", return_value=mock_llm):
        with patch.dict("agents.coder._TOOL_MAP", {"execute_python": mock_tool}, clear=False):
            code, stdout, stderr, success, artifacts, _ = _react_loop(
                [HumanMessage(content="写一个最小 Python 示例并运行")]
            )

    assert code == "print('sandbox ok')"
    assert "sandbox ok" in stdout
    assert stderr == ""
    assert success is True
    assert artifacts and artifacts[0].endswith("result.png")
