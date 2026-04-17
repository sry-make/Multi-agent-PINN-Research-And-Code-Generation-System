from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def llm_response_factory():
    def _make(
        *,
        content: str = "",
        tool_calls: list[dict] | None = None,
        total_tokens: int = 10,
    ) -> MagicMock:
        response = MagicMock()
        response.content = content
        response.tool_calls = tool_calls or []
        response.usage_metadata = {"total_tokens": total_tokens}
        return response

    return _make
