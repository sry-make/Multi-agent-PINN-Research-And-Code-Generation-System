"""
agents/researcher.py — 科研大脑 Agent

职责: 克服幻觉，确保学术严谨性，产出有文献依据的报告和技术方案。

流程: ReAct 循环（最多 MAX_ITER 轮）
      工具调用 → 执行 → 继续推理 → … → 无工具调用时生成最终报告

工具: search_local_papers, search_arxiv, web_search,
      simplify_formula, latex_to_sympy

输出字段（写入 AgentState）:
    literature_report   文献综述报告
    design_proposal     技术方案设计（仅 full_pipeline intent）
    retrieved_sources   检索来源元数据列表
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_RESEARCHER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)
from memory import format_project_memory, format_session_summary
from observability.cost_tracker import cost_tracker
from observability.tracer import timer, tracer
from orchestrator.state import AgentState
from tools.formula_tools import latex_to_sympy, simplify_formula
from tools.rag_tools import search_local_papers
from tools.search_tools import search_arxiv, web_search

# ── 常量 ────────────────────────────────────────────────────
MAX_ITER   = 5          # ReAct 最大迭代轮数
AGENT_NAME = "Researcher"

_TOOLS   = [search_local_papers, search_arxiv, web_search,
            simplify_formula, latex_to_sympy]
_TOOL_MAP = {t.name: t for t in _TOOLS}

# ── 系统提示词 ───────────────────────────────────────────────
_SYSTEM_PROMPT = """\
你是一个顶级的 PINN（物理信息神经网络）科研专家 Agent。

【核心守则 — 绝不可违反】
1. 禁止幻觉：所有学术观点必须有工具检索的原文支撑，禁止凭空捏造引用或数据。
2. 工具优先：回答前必须先调用工具检索相关文献，再基于结果作答。
3. 严格引用：引用论文格式为 [来源: 文件名 或 arXiv:XXXX.XXXXX]。
4. 公式规范：所有数学公式使用标准 LaTeX；损失函数写作 $\\mathcal{L}_{u}$、$\\mathcal{L}_{f}$。

【工具使用策略】
- 优先 search_local_papers（本地库精度最高）
- 本地无结果时用 search_arxiv 扩展
- 验证公式时用 simplify_formula 或 latex_to_sympy
- 仅在需要非论文资源时使用 web_search

【工作流程】先检索 → 分析原文 → 综合撰写 → 给出引用。\
"""

_SURVEY_INSTRUCTION = """

请基于检索结果输出一份结构化文献综述，包含：
## 1. 核心方法与关键公式
## 2. 主要文献贡献对比
## 3. 当前研究局限性与开放问题
"""

_DESIGN_INSTRUCTION = """

在文献综述基础上，额外输出技术方案设计：
## 方案设计
- 推荐的 PINN 架构与网络结构
- 损失函数设计（含权重策略）
- 关键超参数建议
- 预期挑战与应对策略
"""


# ── LLM 工厂 ────────────────────────────────────────────────
def _build_llm(with_tools: bool = True) -> ChatOpenAI:
    llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        model=MODEL_RESEARCHER,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return llm.bind_tools(_TOOLS) if with_tools else llm


# ── Token 计数辅助 ───────────────────────────────────────────
def _count_tokens(response: AIMessage) -> int:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return 0
    if isinstance(meta, dict):
        return meta.get("total_tokens", 0)
    # langchain-core 1.x UsageMetadata 对象
    return getattr(meta, "total_tokens", 0)


def _build_memory_context(state: AgentState) -> str:
    """Format reusable memory context for the Researcher agent."""
    blocks: list[str] = []

    session_text = format_session_summary(state.get("session_summary") or {})
    if session_text:
        blocks.append("Session memory:\n" + session_text)

    project_text = format_project_memory(state.get("project_memory") or {})
    if project_text:
        blocks.append("Project memory:\n" + project_text)

    return "\n\n".join(blocks)


# ── ReAct 核心循环 ───────────────────────────────────────────
def _react_loop(
    messages: list,
) -> tuple[str, list, list[dict[str, Any]]]:
    """
    ReAct 推理-行动循环。

    Returns:
        final_content     最终文本输出
        updated_messages  含完整对话历史的消息列表
        sources           检索来源列表 [{tool, query}, ...]
    """
    llm_with_tools = _build_llm(with_tools=True)
    sources: list[dict[str, Any]] = []

    for _ in range(MAX_ITER):
        with timer() as t:
            response = llm_with_tools.invoke(messages)

        tokens = _count_tokens(response)
        cost_tracker.record(AGENT_NAME, MODEL_RESEARCHER, tokens)
        tracer.log_llm_call(
            agent=AGENT_NAME,
            prompt=str(messages[-1].content)[:500],
            response=str(response.content)[:500],
            model=MODEL_RESEARCHER,
            tokens_used=tokens,
            duration_ms=t.ms,
        )

        # 无工具调用 → 最终答案
        if not response.tool_calls:
            return response.content or "", messages + [response], sources

        # 执行工具调用
        messages = messages + [response]
        for tc in response.tool_calls:
            name    = tc["name"]
            args    = tc["args"]
            call_id = tc["id"]

            with timer() as tt:
                fn     = _TOOL_MAP.get(name)
                result = fn.invoke(args) if fn else f"[错误] 未知工具: {name}"
                if isinstance(result, Exception):
                    result = f"[工具异常] {result}"

            tracer.log_tool_call(
                agent=AGENT_NAME,
                tool_name=name,
                tool_input=args,
                tool_output=str(result),
                duration_ms=tt.ms,
            )

            if name in ("search_local_papers", "search_arxiv"):
                sources.append({"tool": name, "query": args.get("query", "")})

            messages = messages + [
                ToolMessage(content=str(result), tool_call_id=call_id)
            ]

    # 达到迭代上限 → 强制无工具总结
    llm_plain = _build_llm(with_tools=False)
    force_msg = HumanMessage(
        content="请基于以上所有检索结果，给出完整的最终综述回答。不要再调用工具。"
    )
    with timer() as t:
        final = llm_plain.invoke(messages + [force_msg])

    tokens = _count_tokens(final)
    cost_tracker.record(AGENT_NAME, MODEL_RESEARCHER, tokens)
    return (
        final.content or "",
        messages + [force_msg, final],
        sources,
    )


# ── Agent 节点入口 ───────────────────────────────────────────
def run_researcher(state: AgentState) -> dict:
    """
    LangGraph 节点函数。
    接收全局状态，调用 ReAct 循环，返回更新后的状态字段。
    """
    query   = state["query"]
    intent  = state.get("intent", "qa")
    history = list(state.get("messages") or [])

    tracer.log_state_transition(
        from_step=state.get("current_step", "parse_intent"),
        to_step="researcher",
        intent=intent,
    )

    # 根据 intent 组装用户提示
    if intent == "survey":
        user_text = query + _SURVEY_INSTRUCTION
    elif intent == "full_pipeline":
        user_text = query + _SURVEY_INSTRUCTION + _DESIGN_INSTRUCTION
    else:  # qa
        user_text = query

    memory_context = _build_memory_context(state)
    if memory_context:
        user_text = f"{memory_context}\n\nCurrent task:\n{user_text}"

    init_messages: list = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    messages = history + init_messages

    final_content, updated_messages, sources = _react_loop(messages)

    # full_pipeline 时将设计方案从最终内容中分割出来
    literature_report = final_content
    design_proposal   = ""
    if intent == "full_pipeline" and "## 方案设计" in final_content:
        parts             = final_content.split("## 方案设计", 1)
        literature_report = parts[0].strip()
        design_proposal   = "## 方案设计" + parts[1].strip()

    return {
        "current_step":      "researcher",
        "literature_report": literature_report,
        "design_proposal":   design_proposal,
        "retrieved_sources": sources,
        "messages":          updated_messages,
    }
