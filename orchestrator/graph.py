"""
LangGraph 主图 — SOP 状态机定义（Phase 2 真实 Agent 版本）

节点（Nodes）:
    parse_intent   → 意图识别与路由
    memory_read    → 读取会话 / 项目 / 经验记忆
    clarify        → 主动消歧 / 路由兜底提示
    researcher     → Researcher Agent（科研大脑）
    coder          → Coder Agent（编程执行）
    examiner       → Examiner Agent（质量门禁）
    synthesize     → 汇总最终输出
    memory_writeback → 会话摘要与经验写回

边（Edges）:
    parse_intent ─────────────► memory_read
    memory_read  ──(条件路由)──► researcher / coder / clarify
    researcher   ──(条件路由)──► examiner / coder
    coder        ──────────────► examiner
    examiner     ──(PASS/FAIL)─► synthesize / coder(重试，最多3次)
    clarify      ──────────────► memory_writeback ─► END
    synthesize   ──────────────► memory_writeback ─► END
"""

from __future__ import annotations

from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from orchestrator.router import looks_like_contextual_followup, resolve_intent, route_by_intent
from orchestrator.state import AgentState


# ── 节点：意图识别 ───────────────────────────────────────────
def node_parse_intent(state: AgentState) -> dict:
    """Step 1: 识别意图，初始化所有计数器字段。"""
    decision = resolve_intent(state["query"])
    intent = decision.intent
    print(
        "[Router] 意图识别: "
        f"'{state['query'][:60]}' → {intent} "
        f"(source={decision.source}, conf={decision.confidence:.2f}, entropy={decision.entropy:.2f})"
    )
    return {
        "session_id":           state.get("session_id", ""),
        "intent":               intent,
        "current_step":         "parse_intent",
        "router_source":        decision.source,
        "router_confidence":    decision.confidence,
        "router_entropy":       decision.entropy,
        "router_reason":        decision.reason,
        "router_scores":        decision.scores,
        "router_needs_clarification": decision.needs_clarification,
        "router_clarification_question": decision.clarification_question,
        "router_missing_information": decision.missing_information,
        "router_default_intent": decision.default_intent,
        "router_defaulted":     decision.defaulted,
        "total_tokens_used":    state.get("total_tokens_used", 0),
        "token_budget_exceeded": False,
        "code_retry_count":     0,
        "examiner_retry_count": 0,
        "execution_success":    False,
        "session_summary":      state.get("session_summary", {}),
        "project_memory":       state.get("project_memory", {}),
        "experience_hints":     [],
        # 清空上一轮残留输出（多轮对话场景）
        "literature_report":    "",
        "design_proposal":      "",
        "retrieved_sources":    [],
        "generated_code":       "",
        "execution_stdout":     "",
        "execution_stderr":     "",
        "artifact_paths":       [],
        "academic_review":      "",
        "code_review":          "",
        "examiner_verdict":     "",
        "final_answer":         "",
    }


def node_memory_read(state: AgentState) -> dict:
    """在执行主任务前读取短期 / 长期记忆。"""
    from memory import (
        SessionManager,
        compress_message_history,
        load_project_memory,
        retrieve_experience_hints,
    )

    session_id = str(state.get("session_id", "")).strip()
    session_manager = SessionManager()
    session_summary = session_manager.load_summary(session_id) if session_id else {}
    history = list(state.get("messages") or [])
    trimmed_messages, session_summary, compressed = compress_message_history(
        history,
        session_summary,
    )

    update: dict = {
        "session_summary": session_summary,
        "project_memory": load_project_memory(),
        "experience_hints": retrieve_experience_hints(
            state.get("query", ""),
            intent=state.get("intent", ""),
            limit=3,
        ),
    }

    if (
        state.get("intent") == "clarify"
        and state.get("router_needs_clarification")
        and looks_like_contextual_followup(state.get("query", ""))
    ):
        last_intent = str(session_summary.get("last_intent", "")).strip()
        if last_intent in {"qa", "survey", "code", "full_pipeline"}:
            update.update(
                {
                    "intent": last_intent,
                    "router_source": "session_default",
                    "router_needs_clarification": False,
                    "router_default_intent": last_intent,
                    "router_defaulted": True,
                    "router_reason": (
                        "Low-confidence follow-up reused the previous session intent "
                        f"'{last_intent}' for tolerant execution."
                    ),
                }
            )

    if compressed:
        if session_id:
            session_manager.save_summary(session_id, session_summary)
        update["messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed_messages,
        ]

    return update


def node_clarify(state: AgentState) -> dict:
    """低置信度场景下主动消歧，并告知默认可继续执行的路径。"""
    from observability.tracer import tracer

    tracer.log_state_transition(
        from_step=state.get("current_step", "memory_read"),
        to_step="clarify",
        intent=str(state.get("router_default_intent") or state.get("intent") or "qa"),
    )

    default_intent = str(state.get("router_default_intent", "qa")).strip() or "qa"
    confidence = float(state.get("router_confidence", 0.0) or 0.0)
    entropy = float(state.get("router_entropy", 1.0) or 1.0)
    question = str(state.get("router_clarification_question", "")).strip()
    missing = [
        str(item).strip()
        for item in (state.get("router_missing_information") or [])
        if str(item).strip()
    ]

    parts = [
        "## 路由澄清",
        "",
        "我现在还不能高置信度判断你希望我走哪条路径，所以先做一次主动消歧。",
        f"- 当前置信度: `{confidence:.2f}`",
        f"- 当前熵: `{entropy:.2f}`",
        f"- 默认宽容路径: `{default_intent}`",
    ]
    if missing:
        parts.append(f"- 主要缺少的信息: {'；'.join(missing[:3])}")
    parts.extend(
        [
            "",
            question or "你可以直接告诉我是要回答问题、做综述、写代码，还是先调研再实现。",
            "",
            "如果你不想来回确认，也可以直接回复：`先按 "
            + default_intent
            + " 路径继续`。",
        ]
    )

    return {
        "current_step": "clarify",
        "final_answer": "\n".join(parts),
    }


# ── 节点：Researcher Agent ───────────────────────────────────
def node_researcher(state: AgentState) -> dict:
    """Step 2: 调用 Researcher Agent 进行文献检索与方案设计。"""
    from agents.researcher import run_researcher
    return run_researcher(state)


# ── 节点：Coder Agent ────────────────────────────────────────
def node_coder(state: AgentState) -> dict:
    """Step 3: 调用 Coder Agent 生成并执行代码。"""
    from agents.coder import run_coder
    return run_coder(state)


# ── 节点：Examiner Agent ─────────────────────────────────────
def node_examiner(state: AgentState) -> dict:
    """Step 4: 调用 Examiner Agent 审查学术内容与代码质量。"""
    from agents.examiner import run_examiner
    result = run_examiner(state)

    # 如果裁决为 FAIL，在此递增 code_retry_count（供 Coder 重试时感知）
    if result.get("examiner_verdict") == "FAIL":
        result["code_retry_count"] = state.get("code_retry_count", 0) + 1

    return result


# ── 节点：汇总输出 ───────────────────────────────────────────
def node_synthesize(state: AgentState) -> dict:
    """Step 5: 汇总所有 Agent 的输出为最终回答。"""
    parts: list[str] = []

    if state.get("literature_report"):
        parts.append(f"## 文献综述\n\n{state['literature_report']}")

    if state.get("design_proposal"):
        parts.append(f"## 技术方案\n\n{state['design_proposal']}")

    if state.get("generated_code"):
        parts.append(
            f"## 代码实现\n\n```python\n{state['generated_code']}\n```"
        )

    if state.get("execution_stdout"):
        parts.append(
            f"## 运行结果\n\n```\n{state['execution_stdout']}\n```"
        )

    if state.get("execution_stderr"):
        parts.append(
            f"## 运行错误\n\n```\n{state['execution_stderr']}\n```"
        )

    if state.get("artifact_paths"):
        artifact_lines = "\n".join(
            f"- `{path}`"
            for path in state["artifact_paths"]
        )
        parts.append(f"## 产物文件\n\n{artifact_lines}")

    # 附加完整审查结果，避免在 TUI 中只显示半截意见。
    review_parts: list[str] = []
    if state.get("academic_review"):
        review_parts.append(f"### 学术审查\n\n{state['academic_review']}")
    if state.get("code_review"):
        review_parts.append(f"### 代码审查\n\n{state['code_review']}")
    if review_parts:
        parts.append("## 审查结果\n\n" + "\n\n".join(review_parts))

    final = "\n\n---\n\n".join(parts) if parts else "（暂无输出内容）"
    return {
        "current_step": "synthesize",
        "final_answer": final,
    }


def node_memory_writeback(state: AgentState) -> dict:
    """将本轮结果回写到会话摘要与经验库。"""
    from memory import (
        SessionManager,
        append_experience_record,
        build_experience_record,
        build_session_summary,
    )

    session_id = str(state.get("session_id", "")).strip()
    if not session_id:
        return {}

    session_manager = SessionManager()
    existing_summary = state.get("session_summary") or session_manager.load_summary(session_id)
    updated_summary = build_session_summary(existing_summary, state)
    session_manager.save_summary(session_id, updated_summary)

    experience_record = build_experience_record(state)
    if experience_record:
        append_experience_record(experience_record)

    return {
        "session_summary": updated_summary,
    }


# ── 条件边：Researcher 之后 ──────────────────────────────────
def _after_researcher(state: AgentState) -> str:
    """qa / survey → examiner（学术审查）；full_pipeline → coder（继续编码）"""
    return "coder" if state.get("intent") == "full_pipeline" else "examiner"


# ── 条件边：Examiner 之后 ────────────────────────────────────
def _after_examiner(state: AgentState) -> str:
    """
    PASS → synthesize
    FAIL 且未超重试上限 → 根据 intent 决定重试 coder 或 researcher
    FAIL 且已超上限    → 强制 synthesize（输出当前最佳结果）
    """
    from config import EXAMINER_MAX_RETRIES
    verdict        = state.get("examiner_verdict", "PASS")
    examiner_tries = state.get("examiner_retry_count", 0)

    if verdict == "PASS" or examiner_tries >= EXAMINER_MAX_RETRIES:
        return "synthesize"

    intent = state.get("intent", "qa")
    return "coder" if intent in ("code", "full_pipeline") else "researcher"


# ── 图构建函数 ───────────────────────────────────────────────
def build_graph(checkpointer=None):
    """
    构建并编译 LangGraph 状态机。

    Args:
        checkpointer: 可选 checkpoint 存储；None 时使用 MemorySaver（内存多轮记忆）

    Returns:
        compiled_graph: 支持 .invoke() / .ainvoke() / .stream() 的编译图
    """
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("parse_intent", node_parse_intent)
    builder.add_node("memory_read",  node_memory_read)
    builder.add_node("clarify",      node_clarify)
    builder.add_node("researcher",   node_researcher)
    builder.add_node("coder",        node_coder)
    builder.add_node("examiner",     node_examiner)
    builder.add_node("synthesize",   node_synthesize)
    builder.add_node("memory_writeback", node_memory_writeback)

    # 起点
    builder.set_entry_point("parse_intent")

    # parse_intent → memory_read
    builder.add_edge("parse_intent", "memory_read")

    # memory_read → 条件路由（qa/survey → researcher；code → coder；clarify → 追问）
    builder.add_conditional_edges(
        "memory_read",
        route_by_intent,
        {
            "clarify":    "clarify",
            "researcher": "researcher",
            "coder":      "coder",
        },
    )

    # researcher → 条件路由
    builder.add_conditional_edges(
        "researcher",
        _after_researcher,
        {
            "coder":    "coder",
            "examiner": "examiner",
        },
    )

    # coder → examiner（固定边）
    builder.add_edge("coder", "examiner")

    # examiner → 条件路由
    builder.add_conditional_edges(
        "examiner",
        _after_examiner,
        {
            "synthesize": "synthesize",
            "coder":      "coder",
            "researcher": "researcher",
        },
    )

    # clarify → memory_writeback → END
    builder.add_edge("clarify", "memory_writeback")

    # synthesize → memory_writeback → END
    builder.add_edge("synthesize", "memory_writeback")
    builder.add_edge("memory_writeback", END)

    cp = checkpointer or MemorySaver()
    return builder.compile(checkpointer=cp)
