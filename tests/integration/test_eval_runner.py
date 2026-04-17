from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.runner import run_evaluation


pytestmark = pytest.mark.integration


def test_eval_runner_mock_generates_judge_and_engineering_panels(tmp_path) -> None:
    output_dir = tmp_path / "eval_run"

    result = run_evaluation(
        mode="mock",
        case_ids=["code_minimal_pinn"],
        output_dir=output_dir,
        judge_mode="heuristic",
    )

    metrics_path = Path(result["metrics_path"])
    summary_path = Path(result["summary_path"])
    results_path = Path(result["results_path"])

    assert metrics_path.exists()
    assert summary_path.exists()
    assert results_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    results_lines = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert metrics["panels"]["quality"]["judge_coverage_rate"] == 100.0
    assert metrics["panels"]["cost"]["total_agent_tokens"] > 0
    assert metrics["panels"]["latency"]["avg_case_duration_ms"] > 0
    assert metrics["panels"]["reliability"]["execution_success_rate"] == 100.0
    assert "## Quality Panel" in summary
    assert "## Engineering Panel" in summary
    assert results_lines[0]["judge"]["mode"] == "heuristic"
    assert results_lines[0]["observability"]["cost"]["total_tokens"] > 0
