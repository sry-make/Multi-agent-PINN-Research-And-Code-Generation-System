"""Run fixed evaluation cases in mock or live mode."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from config import EVAL_DIR, EVAL_JUDGE_MODE
from eval.judge import judge_case
from eval.report import build_metrics, write_eval_report
from eval.rubrics import finalize_case_score, score_case_result
from memory.experience_store import append_experience_record, retrieve_experience_hints
from memory.project_store import default_project_memory, load_project_memory, save_project_memory
from memory.session_manager import SessionManager
from observability.cost_tracker import cost_tracker
from observability.tracer import tracer
from orchestrator.graph import build_graph
from orchestrator.router import _rule_based_intent


DEFAULT_CASES_PATH = Path(EVAL_DIR) / "cases.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"cases file not found: {file_path}")

    cases: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            continue
        cases.append(payload)
    return cases


def load_cases(
    path: str | Path = DEFAULT_CASES_PATH,
    *,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    cases = _load_jsonl(path)
    if case_ids:
        wanted = {item.strip() for item in case_ids if item and item.strip()}
        cases = [case for case in cases if str(case.get("id", "")).strip() in wanted]
    if limit and limit > 0:
        cases = cases[:limit]
    return cases


def _case_turns(case: dict[str, Any]) -> list[str]:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        return [str(item).strip() for item in turns if str(item).strip()]

    query = str(case.get("query", "")).strip()
    return [query] if query else []


def _case_slug(case: dict[str, Any]) -> str:
    return str(case.get("id", "case")).strip() or "case"


def _default_session_id(case: dict[str, Any]) -> str:
    return f"eval-{_case_slug(case)}"


def _clip_text(text: Any, limit: int = 4000) -> str:
    normalized = str(text or "")
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _sanitize_state(state: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(state or {})
    messages = list(sanitized.pop("messages", []) or [])
    sanitized["message_count"] = len(messages)
    sanitized["message_types"] = [type(message).__name__ for message in messages[-6:]]

    for key in (
        "literature_report",
        "design_proposal",
        "generated_code",
        "execution_stdout",
        "execution_stderr",
        "academic_review",
        "code_review",
        "final_answer",
    ):
        if key in sanitized:
            sanitized[key] = _clip_text(sanitized.get(key, ""))

    if "experience_hints" in sanitized:
        sanitized["experience_hints"] = list(sanitized.get("experience_hints") or [])[:5]

    return sanitized


def _materialize_artifacts(case_dir: Path, artifact_names: list[str]) -> list[str]:
    artifact_dir = case_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for name in artifact_names:
        path = artifact_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            path.write_bytes(b"PNG MOCK\n")
        else:
            path.write_text(f"mock artifact generated for {name}\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def _current_trace_offset() -> int:
    path = tracer.log_path
    if path is None or not path.exists():
        return 0
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


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


def _write_trace_records(case_dir: Path, records: list[dict[str, Any]]) -> str:
    if not records:
        return ""

    path = case_dir / "trace_records.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def _collect_trace_snapshot(start_offset: int, *, case_dir: Path) -> dict[str, Any]:
    if not tracer.enabled:
        return {
            "trace_enabled": False,
            "trace_artifact_path": "",
            "record_count": 0,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "state_transition_count": 0,
            "examiner_verdict_count": 0,
            "llm_durations_ms": [],
            "tool_durations_ms": [],
            "total_llm_latency_ms": 0.0,
            "avg_llm_latency_ms": 0.0,
            "p95_llm_latency_ms": 0.0,
            "total_tool_latency_ms": 0.0,
            "avg_tool_latency_ms": 0.0,
            "p95_tool_latency_ms": 0.0,
            "per_agent_llm_calls": {},
            "per_tool_calls": {},
        }

    _, records = tracer.read_session_records_from_offset(start_offset)
    llm_records = [record for record in records if record.get("type") == "llm_call"]
    tool_records = [record for record in records if record.get("type") == "tool_call"]
    state_records = [record for record in records if record.get("type") == "state_transition"]
    verdict_records = [record for record in records if record.get("type") == "examiner_verdict"]

    llm_durations = [float(record.get("duration_ms", 0.0) or 0.0) for record in llm_records]
    tool_durations = [float(record.get("duration_ms", 0.0) or 0.0) for record in tool_records]

    per_agent_llm_calls: dict[str, int] = {}
    for record in llm_records:
        agent = str(record.get("agent", "")).strip() or "unknown"
        per_agent_llm_calls[agent] = per_agent_llm_calls.get(agent, 0) + 1

    per_tool_calls: dict[str, int] = {}
    for record in tool_records:
        tool_name = str(record.get("tool", "")).strip() or "unknown"
        per_tool_calls[tool_name] = per_tool_calls.get(tool_name, 0) + 1

    return {
        "trace_enabled": True,
        "trace_artifact_path": _write_trace_records(case_dir, records),
        "record_count": len(records),
        "llm_call_count": len(llm_records),
        "tool_call_count": len(tool_records),
        "state_transition_count": len(state_records),
        "examiner_verdict_count": len(verdict_records),
        "llm_durations_ms": [round(value, 2) for value in llm_durations],
        "tool_durations_ms": [round(value, 2) for value in tool_durations],
        "total_llm_latency_ms": round(sum(llm_durations), 2),
        "avg_llm_latency_ms": round(sum(llm_durations) / len(llm_durations), 2) if llm_durations else 0.0,
        "p95_llm_latency_ms": _percentile(llm_durations, 0.95),
        "total_tool_latency_ms": round(sum(tool_durations), 2),
        "avg_tool_latency_ms": round(sum(tool_durations) / len(tool_durations), 2) if tool_durations else 0.0,
        "p95_tool_latency_ms": _percentile(tool_durations, 0.95),
        "per_agent_llm_calls": per_agent_llm_calls,
        "per_tool_calls": per_tool_calls,
    }


def _collect_observability(
    *,
    turns: list[str],
    turn_results: list[dict[str, Any]],
    final_state: dict[str, Any],
    post_run_summary: dict[str, Any],
    trace_snapshot: dict[str, Any],
) -> dict[str, Any]:
    cost_snapshot = cost_tracker.snapshot()
    retry_count = int(final_state.get("code_retry_count", 0) or 0)
    examiner_loops = int(final_state.get("examiner_retry_count", 0) or 0)
    artifact_paths = list(final_state.get("artifact_paths") or [])

    return {
        "cost": cost_snapshot,
        "trace": trace_snapshot,
        "workflow": {
            "query_turn_count": len(turns),
            "turn_results": turn_results,
            "retry_triggered": bool(retry_count > 0 or examiner_loops > 1),
            "retry_count": retry_count,
            "examiner_loops": examiner_loops,
            "artifact_count": len(artifact_paths),
            "artifact_generated": bool(artifact_paths),
            "memory_writeback": bool(post_run_summary.get("session_id")),
            "compressed_turns": int(post_run_summary.get("compressed_turns", 0) or 0),
        },
    }


def _mock_report(case_id: str, query: str) -> str:
    if case_id == "survey_solid_mechanics":
        return (
            "PINN methods in solid mechanics combine governing-equation residuals "
            "with boundary constraints for elasticity and inverse identification. "
            "[来源: mock_solid_mechanics_review.pdf] They are attractive for low-data "
            "regimes, but training instability and boundary-condition balancing remain key limits."
        )
    if case_id == "survey_loss_balancing":
        return (
            "PINN loss balancing typically combines boundary loss and physics loss, "
            "often with adaptive weights to avoid one term dominating optimization. "
            "[来源: mock_loss_balancing_review.pdf] Common strategies include manual weights, "
            "gradient normalization, and curriculum-style residual scheduling."
        )
    return (
        f"PINN overview for query: {query}. The explanation grounds the role of "
        "physics loss in constraining the solution manifold and cites a mock source "
        "for reproducibility. [来源: mock_pinn_reference.pdf]"
    )


def _mock_design() -> str:
    return (
        "Use a small fully connected network with Tanh activations, combine boundary "
        "loss with physics loss, print training progress every few epochs, and export "
        "a train_log.txt plus loss.png for demo visibility."
    )


def _mock_code(profile: str, query: str, hints: list[dict[str, Any]] | None = None) -> str:
    base = [
        "import torch",
        "",
        "def main():",
        "    physics_loss = torch.tensor(0.01)",
        "    data_loss = torch.tensor(0.02)",
        "    total_loss = physics_loss + data_loss",
        "    print('training step 0 | total_loss=', float(total_loss))",
    ]

    if profile == "experience_reuse":
        resolution = ""
        if hints:
            resolution = str(hints[0].get("resolution_hint", "")).strip()
        base.extend(
            [
                "    # shape mismatch fix path",
                "    pred = torch.zeros(4, 1)",
                "    target = torch.zeros(4, 1)",
                "    print('fixed shape mismatch by aligning tensor dimensions')",
                f"    print({resolution!r})",
            ]
        )
    elif profile == "artifact_pass":
        base.append("    print('export train_log.txt and result.txt to host-visible artifacts')")
    elif profile == "memory_followup":
        base.append("    print('extended training log export based on previous turn context')")
    else:
        base.append("    print('PINN minimal example with physical loss term')")

    base.extend(["", "if __name__ == '__main__':", "    main()"])
    return "\n".join(base)


def _build_memory_patches(
    *,
    session_dir: Path,
    project_memory_path: Path,
    experience_db_path: Path,
) -> list[Any]:
    def _session_manager_factory(*_args, **_kwargs) -> SessionManager:
        return SessionManager(session_dir)

    return [
        patch("memory.SessionManager", side_effect=_session_manager_factory),
        patch(
            "memory.load_project_memory",
            side_effect=lambda *_args, **_kwargs: load_project_memory(project_memory_path),
        ),
        patch(
            "memory.retrieve_experience_hints",
            side_effect=lambda query, intent="", limit=3: retrieve_experience_hints(
                query,
                intent=intent,
                limit=limit,
                path=experience_db_path,
            ),
        ),
        patch(
            "memory.append_experience_record",
            side_effect=lambda record: append_experience_record(record, path=experience_db_path),
        ),
    ]


def _build_mock_patches(case: dict[str, Any], case_dir: Path) -> list[Any]:
    profile = str(case.get("mock_profile", "")).strip() or "qa_pass"
    expected_intent = str(case.get("expected_intent", "")).strip()
    mock_artifacts = [str(item).strip() for item in case.get("mock_artifacts") or [] if str(item).strip()]

    def _detect_intent(query: str) -> str:
        return expected_intent or _rule_based_intent(query)

    def _mock_researcher(state: dict[str, Any]) -> dict[str, Any]:
        query = str(state.get("query", "")).strip()
        tracer.log_state_transition(
            from_step=state.get("current_step", "memory_read"),
            to_step="researcher",
            intent=state.get("intent", ""),
        )
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content="Research step completed."),
        ]
        report = _mock_report(_case_slug(case), query)
        design = _mock_design() if state.get("intent") == "full_pipeline" else ""

        cost_tracker.record("Researcher", "mock-researcher", 180)
        tracer.log_llm_call(
            agent="Researcher",
            prompt=query,
            response=report,
            model="mock-researcher",
            tokens_used=180,
            duration_ms=42.0,
            metadata={"mock": True},
        )
        tracer.log_tool_call(
            agent="Researcher",
            tool_name="search_local_papers",
            tool_input={"query": query},
            tool_output="[mock] local papers result",
            duration_ms=12.0,
        )

        return {
            "current_step": "researcher",
            "literature_report": report,
            "design_proposal": design,
            "retrieved_sources": [{"tool": "mock_search", "query": query}],
            "messages": messages,
        }

    def _mock_coder(state: dict[str, Any]) -> dict[str, Any]:
        query = str(state.get("query", "")).strip()
        retry_count = int(state.get("code_retry_count", 0) or 0)
        hints = list(state.get("experience_hints") or [])
        tracer.log_state_transition(
            from_step=state.get("current_step", "researcher"),
            to_step="coder",
            intent=state.get("intent", ""),
        )
        messages = list(state.get("messages") or []) + [
            HumanMessage(content=query),
            AIMessage(content=f"Coder step completed on retry={retry_count}."),
        ]
        coder_tokens = 260 if retry_count == 0 else 190
        cost_tracker.record("Coder", "mock-coder", coder_tokens)

        if profile == "retry_recovery" and retry_count == 0:
            tracer.log_llm_call(
                agent="Coder",
                prompt=query,
                response="Generated a buggy draft for retry simulation.",
                model="mock-coder",
                tokens_used=coder_tokens,
                duration_ms=58.0,
                metadata={"mock": True, "retry_count": retry_count},
            )
            tracer.log_tool_call(
                agent="Coder",
                tool_name="execute_python",
                tool_input={"code": "def main(: ..."},
                tool_output="SyntaxError: invalid syntax",
                duration_ms=25.0,
            )
            return {
                "current_step": "coder",
                "generated_code": "def main(:\n    print('broken')\n",
                "execution_stdout": "",
                "execution_stderr": "SyntaxError: invalid syntax",
                "execution_success": False,
                "artifact_paths": [],
                "code_retry_count": retry_count,
                "messages": messages,
            }

        if profile == "dangerous_code_fail":
            tracer.log_llm_call(
                agent="Coder",
                prompt=query,
                response="Generated dangerous code for safety rejection case.",
                model="mock-coder",
                tokens_used=coder_tokens,
                duration_ms=40.0,
                metadata={"mock": True},
            )
            return {
                "current_step": "coder",
                "generated_code": "import os\nos.system('rm -rf /workspace')\n",
                "execution_stdout": "",
                "execution_stderr": "",
                "execution_success": False,
                "artifact_paths": [],
                "code_retry_count": retry_count,
                "messages": messages,
            }

        generated_code = _mock_code(profile, query, hints)
        artifacts = _materialize_artifacts(case_dir, mock_artifacts)
        stdout_lines = [
            "Epoch 0 | total_loss=1.0e-01 | physics_loss=1.0e-02",
            "Epoch 10 | total_loss=1.0e-03 | physics_loss=1.0e-04",
        ]
        if profile == "artifact_pass":
            stdout_lines.append("Artifacts exported to host-visible files.")
        if profile == "memory_followup":
            stdout_lines.append("Detailed training log export added based on previous turn context.")
        if profile == "experience_reuse":
            stdout_lines.append("Fixed shape mismatch by aligning prediction and target dimensions.")
        if profile == "retry_recovery" and retry_count > 0:
            stdout_lines.append("Recovered from SyntaxError and reran successfully.")

        tracer.log_llm_call(
            agent="Coder",
            prompt=query,
            response="Produced a runnable PINN-style script and exported artifacts.",
            model="mock-coder",
            tokens_used=coder_tokens,
            duration_ms=76.0 if retry_count == 0 else 63.0,
            metadata={"mock": True, "retry_count": retry_count},
        )
        tracer.log_tool_call(
            agent="Coder",
            tool_name="execute_python",
            tool_input={"code": generated_code},
            tool_output="\n".join(stdout_lines),
            duration_ms=33.0,
        )

        return {
            "current_step": "coder",
            "generated_code": generated_code,
            "execution_stdout": "\n".join(stdout_lines),
            "execution_stderr": "",
            "execution_success": True,
            "artifact_paths": artifacts,
            "code_retry_count": retry_count,
            "messages": messages,
        }

    def _mock_examiner(state: dict[str, Any]) -> dict[str, Any]:
        retry_count = int(state.get("examiner_retry_count", 0) or 0)
        tracer.log_state_transition(
            from_step=state.get("current_step", "coder"),
            to_step="examiner",
            intent=state.get("intent", ""),
        )
        academic_review = ""
        code_review = ""
        verdict = "PASS"

        report = str(state.get("literature_report", "")).strip()
        code = str(state.get("generated_code", "")).strip()
        stderr = str(state.get("execution_stderr", "")).strip()

        if report:
            if "[来源:" in report:
                academic_review = "[PASS] Academic content is grounded in cited evidence."
            else:
                academic_review = "[FAIL] Academic content lacks citations."
                verdict = "FAIL"

        if code:
            if "os.system" in code or "rm -rf" in code:
                code_review = "[规则预检 FAIL] 检测到危险操作，拒绝通过。 [FAIL]"
                verdict = "FAIL"
            elif not state.get("execution_success", False):
                code_review = f"[规则预检 FAIL] Detected runtime failure: {stderr} [FAIL]"
                verdict = "FAIL"
            else:
                extra = ""
                if profile == "retry_recovery" and int(state.get("code_retry_count", 0) or 0) > 0:
                    extra = " 已针对上一轮 SyntaxError 完成修复。"
                if profile == "experience_reuse" and state.get("experience_hints"):
                    extra += " 已复用历史经验中的维度对齐建议。"
                code_review = (
                    "[快速审查 PASS] 代码执行成功，包含物理损失项，无危险操作。"
                    f"{extra} [PASS]"
                )

        cost_tracker.record("Examiner", "mock-examiner", 90)
        tracer.log_llm_call(
            agent="Examiner",
            prompt="mock review",
            response=(academic_review or "") + "\n" + (code_review or ""),
            model="mock-examiner",
            tokens_used=90,
            duration_ms=29.0,
            metadata={"mock": True},
        )
        tracer.log_examiner_verdict(
            verdict=verdict,
            review=f"{academic_review} | {code_review}",
            retry_count=retry_count,
        )

        return {
            "current_step": "examiner",
            "academic_review": academic_review,
            "code_review": code_review,
            "examiner_verdict": verdict,
            "examiner_retry_count": retry_count + 1,
        }

    return [
        patch("orchestrator.graph.detect_intent", side_effect=_detect_intent),
        patch("agents.researcher.run_researcher", side_effect=_mock_researcher),
        patch("agents.coder.run_coder", side_effect=_mock_coder),
        patch("agents.examiner.run_examiner", side_effect=_mock_examiner),
    ]


def run_case(
    case: dict[str, Any],
    *,
    mode: str,
    run_dir: Path,
    judge_mode: str | None = None,
) -> dict[str, Any]:
    case_id = _case_slug(case)
    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    turns = _case_turns(case)
    if not turns:
        raise ValueError(f"case `{case_id}` has no query/turns")

    session_id = str(case.get("session_id", "")).strip() or _default_session_id(case)
    session_dir = case_dir / "sessions"
    project_memory_path = case_dir / "project_memory.json"
    experience_db_path = case_dir / "experience_db.jsonl"
    manager = SessionManager(session_dir)

    save_project_memory(default_project_memory(), path=project_memory_path)

    seed_summary = dict(case.get("seed_session_summary") or {})
    if seed_summary:
        seeded = manager.default_summary(session_id)
        seeded.update(seed_summary)
        manager.save_summary(session_id, seeded)

    for record in case.get("seed_experience_records") or []:
        append_experience_record(
            {
                "session_id": session_id,
                **dict(record),
            },
            path=experience_db_path,
        )

    graph = build_graph()
    turn_results: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    error = ""
    started_at = _utc_now()
    started_perf = time.perf_counter()
    cost_tracker.reset_session()
    trace_start_offset = _current_trace_offset()

    patches = _build_memory_patches(
        session_dir=session_dir,
        project_memory_path=project_memory_path,
        experience_db_path=experience_db_path,
    )
    if mode == "mock":
        patches.extend(_build_mock_patches(case, case_dir))

    with ExitStack() as stack:
        for ctx in patches:
            stack.enter_context(ctx)

        try:
            for query in turns:
                final_state = graph.invoke(
                    {"query": query, "messages": [], "session_id": session_id},
                    config={"configurable": {"thread_id": session_id}},
                )
                turn_results.append(
                    {
                        "query": query,
                        "intent": final_state.get("intent", ""),
                        "examiner_verdict": final_state.get("examiner_verdict", ""),
                        "execution_success": bool(final_state.get("execution_success", False)),
                        "artifact_count": len(list(final_state.get("artifact_paths") or [])),
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    duration_ms = round((time.perf_counter() - started_perf) * 1000, 2)
    post_run_summary = manager.load_summary(session_id)
    if final_state:
        final_state["session_summary"] = post_run_summary

    trace_snapshot = _collect_trace_snapshot(trace_start_offset, case_dir=case_dir)
    observability = _collect_observability(
        turns=turns,
        turn_results=turn_results,
        final_state=final_state,
        post_run_summary=post_run_summary,
        trace_snapshot=trace_snapshot,
    )

    result = {
        "case": case,
        "mode": mode,
        "session_id": session_id,
        "queries": turns,
        "turn_results": turn_results,
        "started_at": started_at,
        "duration_ms": duration_ms,
        "post_run_summary": post_run_summary,
        "final_state": _sanitize_state(final_state),
        "observability": observability,
        "error": error,
    }

    rule_rubric = score_case_result(case, result)
    judge_result = judge_case(
        case,
        result,
        rule_rubric,
        run_mode=mode,
        judge_mode=judge_mode,
    )
    result["judge"] = judge_result
    result["rubric"] = finalize_case_score(rule_rubric, judge_result)

    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def run_evaluation(
    *,
    mode: str,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    judge_mode: str | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path, case_ids=case_ids, limit=limit)
    if not cases:
        raise ValueError("no cases selected")

    if output_dir:
        run_dir = Path(output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(EVAL_DIR) / "runs" / f"run_{timestamp}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_case(case, mode=mode, run_dir=run_dir, judge_mode=judge_mode)
        for case in cases
    ]
    results_path = run_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = build_metrics(results, mode=mode)
    metrics_path, summary_path = write_eval_report(run_dir, results, metrics)
    return {
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed eval cases for PINN Agent V2.")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--case-id", action="append", default=[], help="Run only the selected case id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected cases.")
    parser.add_argument("--output-dir", default="", help="Optional custom output directory.")
    parser.add_argument(
        "--judge-mode",
        choices=["auto", "off", "heuristic", "llm"],
        default=EVAL_JUDGE_MODE,
        help="Second-layer judge backend. auto: mock->heuristic, live->llm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        mode=args.mode,
        cases_path=args.cases_path,
        case_ids=args.case_id or None,
        limit=args.limit or None,
        output_dir=args.output_dir or None,
        judge_mode=args.judge_mode,
    )
    metrics = result["metrics"]
    quality = dict(metrics.get("panels", {}).get("quality", {}))
    print(f"Eval run dir: {result['run_dir']}")
    print(f"Results JSONL: {result['results_path']}")
    print(f"Metrics JSON: {result['metrics_path']}")
    print(f"Summary MD: {result['summary_path']}")
    print(
        "Pass rate: {passed}/{total} ({rate}%), avg score={score}, judge coverage={judge_rate}%".format(
            passed=metrics.get("passed_cases", 0),
            total=metrics.get("total_cases", 0),
            rate=metrics.get("pass_rate", 0.0),
            score=metrics.get("avg_score", 0.0),
            judge_rate=quality.get("judge_coverage_rate", 0.0),
        )
    )


if __name__ == "__main__":
    main()
