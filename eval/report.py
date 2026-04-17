"""Report generation for evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 2)


def build_metrics(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for item in results if item.get("rubric", {}).get("passed"))
    failed_cases = total_cases - passed_cases
    avg_score = round(
        sum(float(item.get("rubric", {}).get("score", 0.0)) for item in results) / total_cases,
        2,
    ) if total_cases else 0.0
    avg_rule_score = round(
        sum(float(item.get("rubric", {}).get("rule_score", item.get("rubric", {}).get("score", 0.0))) for item in results) / total_cases,
        2,
    ) if total_cases else 0.0
    avg_duration_ms = round(
        sum(float(item.get("duration_ms", 0.0)) for item in results) / total_cases,
        2,
    ) if total_cases else 0.0

    category_metrics: dict[str, dict[str, Any]] = {}
    execution_successes = 0
    execution_attempt_cases = 0
    artifact_cases = 0
    memory_writebacks = 0
    retry_cases = 0
    retry_counts: list[float] = []
    examiner_loops: list[float] = []
    query_turn_counts: list[float] = []
    compressed_turn_counts: list[float] = []
    case_durations: list[float] = []
    llm_latencies: list[float] = []
    tool_latencies: list[float] = []
    case_tokens: list[float] = []
    judge_scores: list[float] = []
    judge_tokens: list[float] = []
    judge_durations: list[float] = []
    judge_attempts = 0
    judge_successes = 0
    judge_passes = 0
    per_agent_tokens: dict[str, int] = {}
    per_model_tokens: dict[str, int] = {}
    per_agent_llm_calls: dict[str, int] = {}

    for item in results:
        case = dict(item.get("case") or {})
        rubric = dict(item.get("rubric") or {})
        state = dict(item.get("final_state") or {})
        summary = dict(item.get("post_run_summary") or {})
        observability = dict(item.get("observability") or {})
        trace = dict(observability.get("trace") or {})
        cost = dict(observability.get("cost") or {})
        workflow = dict(observability.get("workflow") or {})
        judge = dict(item.get("judge") or {})
        category = str(case.get("category", "uncategorized")).strip() or "uncategorized"

        bucket = category_metrics.setdefault(
            category,
            {
                "total_cases": 0,
                "passed_cases": 0,
                "avg_score": 0.0,
                "avg_rule_score": 0.0,
                "scores": [],
                "rule_scores": [],
            },
        )
        bucket["total_cases"] += 1
        bucket["passed_cases"] += 1 if rubric.get("passed") else 0
        bucket["scores"].append(float(rubric.get("score", 0.0)))
        bucket["rule_scores"].append(float(rubric.get("rule_score", rubric.get("score", 0.0))))

        execution_attempted = bool(
            state.get("generated_code")
            or "expect_execution_success" in case
            or str(case.get("category", "")).strip() in {"code", "full_pipeline", "workflow", "safety", "memory"}
        )
        execution_attempt_cases += 1 if execution_attempted else 0
        execution_successes += 1 if execution_attempted and state.get("execution_success") else 0
        artifact_cases += 1 if list(state.get("artifact_paths") or []) else 0
        memory_writebacks += 1 if summary.get("session_id") else 0

        retry_triggered = bool(workflow.get("retry_triggered", False))
        retry_cases += 1 if retry_triggered else 0
        retry_counts.append(float(workflow.get("retry_count", 0) or 0))
        examiner_loops.append(float(workflow.get("examiner_loops", 0) or 0))
        query_turn_counts.append(float(workflow.get("query_turn_count", 0) or 0))
        compressed_turn_counts.append(float(workflow.get("compressed_turns", 0) or 0))
        case_durations.append(float(item.get("duration_ms", 0.0) or 0.0))
        llm_latencies.extend(float(value) for value in trace.get("llm_durations_ms", []) or [])
        tool_latencies.extend(float(value) for value in trace.get("tool_durations_ms", []) or [])
        case_tokens.append(float(cost.get("total_tokens", 0) or 0))

        for agent, tokens in dict(cost.get("per_agent_tokens") or {}).items():
            per_agent_tokens[agent] = per_agent_tokens.get(agent, 0) + int(tokens or 0)
        for model, tokens in dict(cost.get("per_model_tokens") or {}).items():
            per_model_tokens[model] = per_model_tokens.get(model, 0) + int(tokens or 0)
        for agent, calls in dict(trace.get("per_agent_llm_calls") or {}).items():
            per_agent_llm_calls[agent] = per_agent_llm_calls.get(agent, 0) + int(calls or 0)

        if judge.get("attempted"):
            judge_attempts += 1
        if judge.get("score") is not None and not judge.get("error"):
            judge_successes += 1
            judge_scores.append(float(judge.get("score", 0.0) or 0.0))
            judge_tokens.append(float(judge.get("tokens_used", 0) or 0))
            judge_durations.append(float(judge.get("duration_ms", 0.0) or 0.0))
            judge_passes += 1 if judge.get("passed") else 0

    for bucket in category_metrics.values():
        scores = bucket.pop("scores", [])
        rule_scores = bucket.pop("rule_scores", [])
        bucket["avg_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        bucket["avg_rule_score"] = round(sum(rule_scores) / len(rule_scores), 2) if rule_scores else 0.0

    low_score_cases = [
        {
            "id": item.get("case", {}).get("id", ""),
            "score": item.get("rubric", {}).get("score", 0.0),
            "mismatches": item.get("rubric", {}).get("mismatches", []),
        }
        for item in results
        if not item.get("rubric", {}).get("passed")
    ]

    slow_cases = sorted(
        [
            {
                "id": item.get("case", {}).get("id", ""),
                "duration_ms": float(item.get("duration_ms", 0.0) or 0.0),
                "tokens": float(dict(item.get("observability", {}).get("cost", {})).get("total_tokens", 0) or 0),
            }
            for item in results
        ],
        key=lambda item: item["duration_ms"],
        reverse=True,
    )[:5]

    expensive_cases = sorted(
        [
            {
                "id": item.get("case", {}).get("id", ""),
                "tokens": float(dict(item.get("observability", {}).get("cost", {})).get("total_tokens", 0) or 0),
                "duration_ms": float(item.get("duration_ms", 0.0) or 0.0),
            }
            for item in results
        ],
        key=lambda item: item["tokens"],
        reverse=True,
    )[:5]

    total_agent_tokens = int(sum(case_tokens))
    total_llm_calls = sum(per_agent_llm_calls.values())
    total_judge_tokens = int(sum(judge_tokens))

    panels = {
        "quality": {
            "avg_rule_score": avg_rule_score,
            "avg_overall_score": avg_score,
            "avg_judge_score": _avg(judge_scores),
            "judge_coverage_rate": round((judge_successes / total_cases) * 100.0, 2) if total_cases else 0.0,
            "judge_attempt_rate": round((judge_attempts / total_cases) * 100.0, 2) if total_cases else 0.0,
            "judge_pass_rate": round((judge_passes / judge_successes) * 100.0, 2) if judge_successes else 0.0,
            "judge_error_rate": round(((judge_attempts - judge_successes) / judge_attempts) * 100.0, 2) if judge_attempts else 0.0,
        },
        "latency": {
            "avg_case_duration_ms": avg_duration_ms,
            "p50_case_duration_ms": _percentile(case_durations, 0.50),
            "p95_case_duration_ms": _percentile(case_durations, 0.95),
            "max_case_duration_ms": round(max(case_durations), 2) if case_durations else 0.0,
            "avg_llm_latency_ms": _avg(llm_latencies),
            "p95_llm_latency_ms": _percentile(llm_latencies, 0.95),
            "avg_tool_latency_ms": _avg(tool_latencies),
            "p95_tool_latency_ms": _percentile(tool_latencies, 0.95),
            "avg_judge_duration_ms": _avg(judge_durations),
        },
        "cost": {
            "total_agent_tokens": total_agent_tokens,
            "avg_agent_tokens_per_case": _avg(case_tokens),
            "p95_agent_tokens_per_case": _percentile(case_tokens, 0.95),
            "total_llm_calls": total_llm_calls,
            "avg_llm_calls_per_case": round(total_llm_calls / total_cases, 2) if total_cases else 0.0,
            "avg_tokens_per_llm_call": round(total_agent_tokens / total_llm_calls, 2) if total_llm_calls else 0.0,
            "total_judge_tokens": total_judge_tokens,
            "avg_judge_tokens": _avg(judge_tokens),
            "per_agent_tokens": per_agent_tokens,
            "per_model_tokens": per_model_tokens,
            "per_agent_llm_calls": per_agent_llm_calls,
        },
        "reliability": {
            "execution_attempt_cases": execution_attempt_cases,
            "execution_success_rate": round((execution_successes / execution_attempt_cases) * 100.0, 2) if execution_attempt_cases else 0.0,
            "artifact_case_rate": round((artifact_cases / total_cases) * 100.0, 2) if total_cases else 0.0,
            "memory_writeback_rate": round((memory_writebacks / total_cases) * 100.0, 2) if total_cases else 0.0,
            "retry_rate": round((retry_cases / total_cases) * 100.0, 2) if total_cases else 0.0,
            "avg_retry_count": _avg(retry_counts),
            "avg_examiner_loops": _avg(examiner_loops),
            "avg_query_turn_count": _avg(query_turn_counts),
            "avg_compressed_turns": _avg(compressed_turn_counts),
        },
    }

    return {
        "generated_at": _utc_now(),
        "mode": mode,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round((passed_cases / total_cases) * 100.0, 2) if total_cases else 0.0,
        "avg_score": avg_score,
        "avg_rule_score": avg_rule_score,
        "avg_duration_ms": avg_duration_ms,
        "execution_success_rate": round((execution_successes / execution_attempt_cases) * 100.0, 2) if execution_attempt_cases else 0.0,
        "execution_attempt_cases": execution_attempt_cases,
        "artifact_case_rate": round((artifact_cases / total_cases) * 100.0, 2) if total_cases else 0.0,
        "memory_writeback_rate": round((memory_writebacks / total_cases) * 100.0, 2) if total_cases else 0.0,
        "panels": panels,
        "categories": category_metrics,
        "low_score_cases": low_score_cases,
        "slow_cases": slow_cases,
        "expensive_cases": expensive_cases,
    }


def write_eval_report(
    run_dir: str | Path,
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    metrics_path = run_path / "metrics.json"
    summary_path = run_path / "summary.md"

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    quality = dict(metrics.get("panels", {}).get("quality", {}))
    latency = dict(metrics.get("panels", {}).get("latency", {}))
    cost = dict(metrics.get("panels", {}).get("cost", {}))
    reliability = dict(metrics.get("panels", {}).get("reliability", {}))

    lines: list[str] = [
        "# Eval Summary",
        "",
        f"- Mode: `{metrics.get('mode', '')}`",
        f"- Total cases: `{metrics.get('total_cases', 0)}`",
        f"- Passed cases: `{metrics.get('passed_cases', 0)}`",
        f"- Failed cases: `{metrics.get('failed_cases', 0)}`",
        f"- Pass rate: `{metrics.get('pass_rate', 0.0)}%`",
        f"- Average overall score: `{metrics.get('avg_score', 0.0)}`",
        f"- Average rule score: `{metrics.get('avg_rule_score', 0.0)}`",
        "",
        "## Quality Panel",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Avg overall score | {quality.get('avg_overall_score', 0.0)} |",
        f"| Avg rule score | {quality.get('avg_rule_score', 0.0)} |",
        f"| Avg judge score | {quality.get('avg_judge_score', 0.0)} |",
        f"| Judge coverage rate | {quality.get('judge_coverage_rate', 0.0)}% |",
        f"| Judge pass rate | {quality.get('judge_pass_rate', 0.0)}% |",
        f"| Judge error rate | {quality.get('judge_error_rate', 0.0)}% |",
        "",
        "## Engineering Panel",
        "",
        "### Latency",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Avg case duration | {latency.get('avg_case_duration_ms', 0.0)} ms |",
        f"| P50 case duration | {latency.get('p50_case_duration_ms', 0.0)} ms |",
        f"| P95 case duration | {latency.get('p95_case_duration_ms', 0.0)} ms |",
        f"| Avg LLM latency | {latency.get('avg_llm_latency_ms', 0.0)} ms |",
        f"| P95 LLM latency | {latency.get('p95_llm_latency_ms', 0.0)} ms |",
        f"| Avg tool latency | {latency.get('avg_tool_latency_ms', 0.0)} ms |",
        f"| P95 tool latency | {latency.get('p95_tool_latency_ms', 0.0)} ms |",
        f"| Avg judge latency | {latency.get('avg_judge_duration_ms', 0.0)} ms |",
        "",
        "### Cost",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total agent tokens | {cost.get('total_agent_tokens', 0)} |",
        f"| Avg tokens / case | {cost.get('avg_agent_tokens_per_case', 0.0)} |",
        f"| P95 tokens / case | {cost.get('p95_agent_tokens_per_case', 0.0)} |",
        f"| Total LLM calls | {cost.get('total_llm_calls', 0)} |",
        f"| Avg LLM calls / case | {cost.get('avg_llm_calls_per_case', 0.0)} |",
        f"| Avg tokens / LLM call | {cost.get('avg_tokens_per_llm_call', 0.0)} |",
        f"| Total judge tokens | {cost.get('total_judge_tokens', 0)} |",
        "",
        "### Reliability",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Execution attempt cases | {reliability.get('execution_attempt_cases', 0)} |",
        f"| Execution success rate | {reliability.get('execution_success_rate', 0.0)}% |",
        f"| Artifact case rate | {reliability.get('artifact_case_rate', 0.0)}% |",
        f"| Memory writeback rate | {reliability.get('memory_writeback_rate', 0.0)}% |",
        f"| Retry rate | {reliability.get('retry_rate', 0.0)}% |",
        f"| Avg retry count | {reliability.get('avg_retry_count', 0.0)} |",
        f"| Avg examiner loops | {reliability.get('avg_examiner_loops', 0.0)} |",
        f"| Avg query turns | {reliability.get('avg_query_turn_count', 0.0)} |",
        f"| Avg compressed turns | {reliability.get('avg_compressed_turns', 0.0)} |",
        "",
        "## Cost Breakdown",
        "",
        "| Agent | Tokens | LLM Calls |",
        "| --- | ---: | ---: |",
    ]

    per_agent_tokens = dict(cost.get("per_agent_tokens", {}))
    per_agent_llm_calls = dict(cost.get("per_agent_llm_calls", {}))
    agent_names = sorted(set(per_agent_tokens) | set(per_agent_llm_calls))
    for agent in agent_names:
        lines.append(
            f"| {agent} | {per_agent_tokens.get(agent, 0)} | {per_agent_llm_calls.get(agent, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case ID | Category | Rule | Judge | Overall | Passed | Intent | Verdict | Retry | Tokens | Duration |",
            "| --- | --- | ---: | ---: | ---: | :---: | --- | --- | :---: | ---: | ---: |",
        ]
    )

    for item in results:
        case = dict(item.get("case") or {})
        state = dict(item.get("final_state") or {})
        rubric = dict(item.get("rubric") or {})
        judge = dict(item.get("judge") or {})
        tokens = dict(item.get("observability", {}).get("cost", {})).get("total_tokens", 0)
        retry_flag = (
            int(state.get("code_retry_count", 0) or 0) > 0
            or int(state.get("examiner_retry_count", 0) or 0) > 1
        )
        judge_score = judge.get("score")
        judge_display = (
            f"{float(judge_score):.2f}" if judge_score is not None else ("err" if judge.get("error") else "-")
        )
        lines.append(
            "| {case_id} | {category} | {rule:.2f} | {judge} | {overall:.2f} | {passed} | {intent} | {verdict} | {retry} | {tokens} | {duration:.2f} ms |".format(
                case_id=case.get("id", ""),
                category=case.get("category", ""),
                rule=float(rubric.get("rule_score", rubric.get("score", 0.0))),
                judge=judge_display,
                overall=float(rubric.get("score", 0.0)),
                passed="yes" if rubric.get("passed") else "no",
                intent=state.get("intent", ""),
                verdict=state.get("examiner_verdict", ""),
                retry="yes" if retry_flag else "no",
                tokens=int(tokens or 0),
                duration=float(item.get("duration_ms", 0.0) or 0.0),
            )
        )

    slow_cases = list(metrics.get("slow_cases") or [])
    if slow_cases:
        lines.extend(
            [
                "",
                "## Slowest Cases",
                "",
                "| Case ID | Duration | Tokens |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in slow_cases:
            lines.append(
                f"| {item.get('id', '')} | {float(item.get('duration_ms', 0.0) or 0.0):.2f} ms | {int(item.get('tokens', 0) or 0)} |"
            )

    expensive_cases = list(metrics.get("expensive_cases") or [])
    if expensive_cases:
        lines.extend(
            [
                "",
                "## Most Expensive Cases",
                "",
                "| Case ID | Tokens | Duration |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in expensive_cases:
            lines.append(
                f"| {item.get('id', '')} | {int(item.get('tokens', 0) or 0)} | {float(item.get('duration_ms', 0.0) or 0.0):.2f} ms |"
            )

    low_score_cases = list(metrics.get("low_score_cases") or [])
    if low_score_cases:
        lines.extend(
            [
                "",
                "## Follow-ups",
                "",
            ]
        )
        for item in low_score_cases:
            case_id = item.get("id", "")
            mismatches = item.get("mismatches") or ["No mismatch details recorded."]
            lines.append(f"### {case_id}")
            for mismatch in mismatches:
                lines.append(f"- {mismatch}")
            lines.append("")

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return metrics_path, summary_path
