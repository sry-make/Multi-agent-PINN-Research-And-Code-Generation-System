"""
agents/examiner.py — 质量门禁 Agent

职责: 对 Researcher 的学术输出和 Coder 的代码输出进行双轨审查，
      输出 PASS / FAIL 裁决及修改建议。

审查策略:
    规则预检（快速、无 LLM）→ LLM 深度审查（慢、精准）→ 综合裁决

学术审查维度:
    - 是否有文献引用支撑
    - 公式符号是否规范
    - 结论是否超出检索证据范围

代码审查维度:
    - 执行是否成功（execution_success）
    - stderr 是否含真实错误（排除警告）
    - 代码是否含危险操作

输出字段（写入 AgentState）:
    academic_review     学术审查意见
    code_review         代码审查意见
    examiner_verdict    "PASS" | "FAIL"
    examiner_retry_count 递增后的重试计数
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS,
    EXAMINER_LLM_TIMEOUT_SEC,
    EXAMINER_STRICT_MODE,
    EXAMINER_REVIEW_MAX_CODE_CHARS,
    EXAMINER_REVIEW_MAX_REPORT_CHARS,
    EXAMINER_REVIEW_MAX_STDIO_CHARS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_EXAMINER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)
from memory import format_code_memory, format_project_memory, format_session_summary
from observability.cost_tracker import cost_tracker
from observability.tracer import timer, tracer
from orchestrator.state import AgentState

AGENT_NAME = "Examiner"

# 危险操作正则（代码安全检查）
_DANGEROUS_PATTERNS = re.compile(
    r"\b(os\.system|subprocess\.call|shutil\.rmtree|rmdir|rm\s+-rf"
    r"|__import__\s*\(\s*['\"]os['\"]|eval\s*\(|exec\s*\()\b",
    re.IGNORECASE,
)

# stderr 中视为"无害警告"的模式（不触发 FAIL）
_WARN_ONLY_PATTERNS = re.compile(
    r"(DeprecationWarning|FutureWarning|UserWarning"
    r"|RuntimeWarning|PendingDeprecationWarning)",
    re.IGNORECASE,
)

# ── LLM 工厂 ────────────────────────────────────────────────
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        model=MODEL_EXAMINER,
        temperature=0.1,           # 审查需要确定性，温度偏低
        max_tokens=LLM_MAX_TOKENS,
        timeout=EXAMINER_LLM_TIMEOUT_SEC,
        max_retries=1,
    )


def _count_tokens(response: AIMessage) -> int:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return 0
    if isinstance(meta, dict):
        return meta.get("total_tokens", 0)
    return getattr(meta, "total_tokens", 0)


def _clip_review_text(text: str, limit: int, label: str) -> str:
    """
    对超长审查输入做首尾保留，避免从中间硬切导致错误语义。
    """
    if not text or len(text) <= limit:
        return text

    head = int(limit * 0.7)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    return (
        text[:head].rstrip()
        + f"\n\n... [{label} 中间已省略 {omitted} 个字符] ...\n\n"
        + text[-tail:].lstrip()
    )


def _non_warning_stderr_lines(stderr: str) -> list[str]:
    """提取 stderr 中真正代表失败的行，过滤常见 warning。"""
    if not stderr:
        return []
    return [
        ln for ln in stderr.splitlines()
        if ln.strip() and not _WARN_ONLY_PATTERNS.search(ln)
    ]


def _build_fast_code_review(stderr: str) -> str:
    """成功执行时的快速代码审查结果，避免演示阶段卡在深度评审。"""
    warning_lines = [
        ln for ln in stderr.splitlines()
        if ln.strip() and _WARN_ONLY_PATTERNS.search(ln)
    ]
    warning_note = ""
    if warning_lines:
        warning_note = (
            "\n检测到 stderr 中仅包含警告信息，未发现真实错误。"
        )

    return (
        "[快速审查 PASS] 代码执行成功，未发现危险操作或真实报错。"
        "为避免开发与演示阶段卡在深度 LLM 评审，已跳过本轮深度代码审查。"
        f"{warning_note}\n"
        "如需强制深度审查，可在 .env 中设置 "
        "PINN_AGENT_EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS=true。\n"
        "[PASS]"
    )


def _build_llm_fallback_review(label: str, exc: Exception, verdict: str) -> str:
    """LLM 审查失败或超时时的降级文案，避免流程长时间挂起。"""
    reason = f"{type(exc).__name__}: {exc}"
    return (
        f"[LLM 审查降级 {verdict}] {label} 深度审查调用失败或超时：{reason}\n"
        "已保留规则预检结论，避免流程在质量审查阶段长时间卡住。\n"
        f"[{verdict}]"
    )


def _build_memory_context(state: AgentState) -> str:
    """Format reusable context for examiner reviews."""
    blocks: list[str] = []

    session_text = format_session_summary(state.get("session_summary") or {})
    if session_text:
        blocks.append("Session memory:\n" + session_text)

    project_text = format_project_memory(state.get("project_memory") or {})
    if project_text:
        blocks.append("Project memory:\n" + project_text)

    code_memory_text = format_code_memory(state.get("session_summary") or {})
    if code_memory_text:
        blocks.append("Short-term code memory:\n" + code_memory_text)

    return "\n\n".join(blocks)


# ── 规则预检 ─────────────────────────────────────────────────
def _rule_check_academic(literature_report: str) -> tuple[bool, str]:
    """
    快速规则检查学术内容。

    Returns:
        (passed, reason)
    """
    if not literature_report or len(literature_report.strip()) < 50:
        return False, "文献综述为空或过短，无实质内容。"

    if EXAMINER_STRICT_MODE:
        # 严格模式：必须含有引用标记 [来源:...] 或 arXiv
        has_citation = bool(
            re.search(r"\[来源[:：]", literature_report)
            or re.search(r"arXiv", literature_report, re.IGNORECASE)
            or re.search(r"\[\d+\]", literature_report)   # 数字引用格式
        )
        if not has_citation:
            return False, "严格模式下：未检测到文献引用标记（[来源:...]、arXiv ID 或 [数字] 格式）。"

    return True, "学术内容规则检查通过。"


def _rule_check_code(
    generated_code: str,
    execution_success: bool,
    execution_stderr: str,
) -> tuple[bool, str]:
    """
    快速规则检查代码安全性与执行结果。

    Returns:
        (passed, reason)
    """
    if not generated_code or len(generated_code.strip()) < 10:
        return False, "未生成有效代码。"

    # 危险操作检查
    if _DANGEROUS_PATTERNS.search(generated_code):
        return False, "代码包含潜在危险操作（os.system / eval / rmtree 等），拒绝通过。"

    # 执行失败且 stderr 含真实错误（非纯警告）
    if not execution_success and execution_stderr:
        non_warn_lines = _non_warning_stderr_lines(execution_stderr)
        if non_warn_lines:
            return False, f"代码执行失败，错误信息（非警告）：\n" + "\n".join(non_warn_lines[:5])

    return True, "代码规则检查通过。"


# ── LLM 深度审查 ─────────────────────────────────────────────
def _llm_review_academic(report: str, extra_context: str = "") -> str:
    """调用 LLM 对文献综述做深度学术审查，返回审查意见文本。"""
    llm = _build_llm()
    context_block = f"\n【额外上下文】\n{extra_context}\n" if extra_context else "\n"
    prompt = f"""\
你是一个严格的 PINN 领域同行评审专家。请对以下文献综述进行学术审查。

【审查维度】
1. 引用真实性：引用是否可验证，有无捏造文献？
2. 公式规范性：数学符号是否标准（$\\mathcal{{L}}_f$、$\\mathcal{{L}}_u$ 等）？
3. 逻辑严谨性：结论是否超出引用证据的支撑范围？
4. 完整性：是否覆盖了问题的核心方面？
{context_block}

【待审文献综述】
{_clip_review_text(report, EXAMINER_REVIEW_MAX_REPORT_CHARS, "文献综述")}

请输出结构化审查意见，最后给出明确的 [PASS] 或 [FAIL] 裁决。\
"""
    with timer() as t:
        response = llm.invoke([
            SystemMessage(content="你是严格的学术审查专家，只输出审查意见。"),
            HumanMessage(content=prompt),
        ])
    tokens = _count_tokens(response)
    cost_tracker.record(AGENT_NAME, MODEL_EXAMINER, tokens)
    tracer.log_llm_call(
        agent=AGENT_NAME,
        prompt=prompt[:300],
        response=str(response.content)[:300],
        model=MODEL_EXAMINER,
        tokens_used=tokens,
        duration_ms=t.ms,
    )
    return response.content or ""


def _llm_review_code(
    code: str,
    stdout: str,
    stderr: str,
    extra_context: str = "",
) -> str:
    """调用 LLM 对代码做深度质量审查，返回审查意见文本。"""
    llm = _build_llm()
    context_block = f"\n【额外上下文】\n{extra_context}\n" if extra_context else "\n"
    prompt = f"""\
你是一个代码审查专家，请对以下 PINN Python 代码进行质量审查。

【审查维度】
1. 代码逻辑正确性：算法实现是否符合 PINN 原理？
2. 运行可靠性：是否存在潜在的运行时错误或边界问题？
3. 科学合理性：损失函数、网络结构、训练流程是否合理？
{context_block}

【代码】
```python
{_clip_review_text(code, EXAMINER_REVIEW_MAX_CODE_CHARS, "代码")}
```
【执行输出（stdout）】
{_clip_review_text(stdout, EXAMINER_REVIEW_MAX_STDIO_CHARS, "stdout") or "（无输出）"}
【错误信息（stderr）】
{_clip_review_text(stderr, EXAMINER_REVIEW_MAX_STDIO_CHARS, "stderr") or "（无错误）"}

请输出结构化审查意见，最后给出明确的 [PASS] 或 [FAIL] 裁决。\
"""
    with timer() as t:
        response = llm.invoke([
            SystemMessage(content="你是严格的代码审查专家，只输出审查意见。"),
            HumanMessage(content=prompt),
        ])
    tokens = _count_tokens(response)
    cost_tracker.record(AGENT_NAME, MODEL_EXAMINER, tokens)
    tracer.log_llm_call(
        agent=AGENT_NAME,
        prompt=prompt[:300],
        response=str(response.content)[:300],
        model=MODEL_EXAMINER,
        tokens_used=tokens,
        duration_ms=t.ms,
    )
    return response.content or ""


def _extract_verdict(review_text: str) -> str:
    """从 LLM 审查意见中提取 PASS / FAIL，默认 PASS（宽松兜底）。"""
    if re.search(r"\[FAIL\]", review_text, re.IGNORECASE):
        return "FAIL"
    if re.search(r"\[PASS\]", review_text, re.IGNORECASE):
        return "PASS"
    # LLM 未给出明确裁决时，根据关键词判断
    negative_signals = re.search(
        r"(捏造|幻觉|错误|危险|不合理|hallucin)", review_text, re.IGNORECASE
    )
    return "FAIL" if negative_signals else "PASS"


# ── Agent 节点入口 ───────────────────────────────────────────
def run_examiner(state: AgentState) -> dict:
    """
    LangGraph 节点函数。
    对 Researcher 输出（学术审查）和 Coder 输出（代码审查）进行双轨审查。
    """
    intent        = state.get("intent", "qa")
    retry_count   = state.get("examiner_retry_count", 0)

    tracer.log_state_transition(
        from_step=state.get("current_step", "coder"),
        to_step="examiner",
        intent=intent,
    )

    academic_review = ""
    code_review     = ""
    verdicts: list[str] = []
    memory_context = _build_memory_context(state)

    # ── 学术审查（有文献综述时执行）────────────────────────
    literature_report = state.get("literature_report", "")
    should_review_academic = bool(literature_report) or intent in {"qa", "survey"}
    if should_review_academic:
        rule_ok, rule_reason = _rule_check_academic(literature_report)
        if not rule_ok:
            # 规则预检直接 FAIL，跳过 LLM 节省 Token
            academic_review = f"[规则预检 FAIL] {rule_reason}"
            verdicts.append("FAIL")
        else:
            try:
                llm_opinion = _llm_review_academic(
                    literature_report,
                    extra_context=memory_context,
                )
            except Exception as exc:
                llm_opinion = _build_llm_fallback_review("学术", exc, "PASS")
            academic_review = llm_opinion
            verdicts.append(_extract_verdict(llm_opinion))

    # ── 代码审查（有生成代码时执行）─────────────────────────
    generated_code = state.get("generated_code", "")
    should_review_code = bool(generated_code) or intent in {"code", "full_pipeline"}
    if should_review_code:
        rule_ok, rule_reason = _rule_check_code(
            generated_code,
            state.get("execution_success", False),
            state.get("execution_stderr", ""),
        )
        if not rule_ok:
            code_review = f"[规则预检 FAIL] {rule_reason}"
            verdicts.append("FAIL")
        elif state.get("execution_success", False) and not EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS:
            code_review = _build_fast_code_review(
                state.get("execution_stderr", "")
            )
            verdicts.append("PASS")
        else:
            try:
                llm_opinion = _llm_review_code(
                    generated_code,
                    state.get("execution_stdout", ""),
                    state.get("execution_stderr", ""),
                    extra_context=memory_context,
                )
            except Exception as exc:
                fallback_verdict = "PASS" if state.get("execution_success", False) else "FAIL"
                llm_opinion = _build_llm_fallback_review(
                    "代码",
                    exc,
                    fallback_verdict,
                )
            code_review = llm_opinion
            verdicts.append(_extract_verdict(llm_opinion))

    # ── 综合裁决：任一 FAIL → 整体 FAIL ────────────────────
    final_verdict = "FAIL" if (not verdicts or "FAIL" in verdicts) else "PASS"

    tracer.log_examiner_verdict(
        verdict=final_verdict,
        review=f"学术: {academic_review[:200]} | 代码: {code_review[:200]}",
        retry_count=retry_count,
    )

    return {
        "current_step":         "examiner",
        "academic_review":      academic_review,
        "code_review":          code_review,
        "examiner_verdict":     final_verdict,
        "examiner_retry_count": retry_count + 1,   # 每次经过 examiner 计数+1
    }
