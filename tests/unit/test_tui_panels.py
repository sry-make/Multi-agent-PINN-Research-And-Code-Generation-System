from __future__ import annotations

import pytest

from tui.app import ArtifactPanel, MemoryStatusPanel


pytestmark = pytest.mark.unit


def test_memory_status_panel_builds_compact_content() -> None:
    panel = MemoryStatusPanel(
        "tui-20260417_113000-demo1234",
        {
            "compressed_turns": 2,
            "recent_queries": ["先写 PINN demo", "继续修复训练日志"],
            "last_failure_error_summary": "RuntimeError: shape mismatch",
            "last_successful_artifacts": [
                "/home/kyrie/PINN_AGENT_PROJECT_V2/outputs/demo/train_log.txt"
            ],
        },
    )

    content = panel._build_content()
    assert "会话    demo1234" in content
    assert "压缩    2 次" in content
    assert "继续修复训练日志" in content
    assert "shape mismatch" in content
    assert "train_log.txt" in content


def test_artifact_panel_shows_current_run_outputs() -> None:
    panel = ArtifactPanel()
    panel.set_artifacts(
        [
            "/home/kyrie/PINN_AGENT_PROJECT_V2/outputs/sandbox_runs/run_demo/result.png",
            "/home/kyrie/PINN_AGENT_PROJECT_V2/outputs/sandbox_runs/run_demo/train.log",
        ]
    )

    content = panel._build_content()
    assert "本次产物: 2 个" in content
    assert "result.png" in content
    assert "train.log" in content
