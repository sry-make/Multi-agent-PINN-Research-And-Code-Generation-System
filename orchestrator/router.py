"""
意图路由器 — 解析用户输入，决定走哪条 SOP 分支

意图类型:
    "qa"            → 快速问答（仅 Researcher，无代码）
    "survey"        → 文献综述（Researcher 深度检索）
    "code"          → 纯代码任务（Coder + Examiner）
    "full_pipeline" → 完整 SOP（Researcher → Coder → Examiner）
"""

from __future__ import annotations

import re
from openai import OpenAI
from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, MODEL_ROUTER, LLM_TEMPERATURE
from orchestrator.state import AgentState


_llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# 规则兜底（避免 LLM 调用失败时无路可走）
_CODE_KEYWORDS   = re.compile(r"(代码|code|implement|编写|实现|程序|脚本|script|notebook)", re.I)
_SURVEY_KEYWORDS = re.compile(r"(综述|survey|文献|调研|review|overview|最新进展)", re.I)


def _rule_based_intent(query: str) -> str:
    """规则兜底意图识别（不调用 LLM）"""
    if _CODE_KEYWORDS.search(query) and _SURVEY_KEYWORDS.search(query):
        return "full_pipeline"
    if _CODE_KEYWORDS.search(query):
        return "code"
    if _SURVEY_KEYWORDS.search(query):
        return "survey"
    return "qa"


def detect_intent(query: str) -> str:
    """使用 LLM 识别意图，失败时降级到规则"""
    prompt = f"""你是一个任务路由器。根据用户的问题，判断它属于以下哪种任务类型：

- "qa"            : 简单的知识问答，只需检索文献直接回答
- "survey"        : 需要系统性文献综述，涉及多篇论文对比
- "code"          : 需要编写并运行代码（如实现 PINN 模型、画图）
- "full_pipeline" : 既需要文献调研，又需要编写并运行代码

请只输出一个词（qa / survey / code / full_pipeline），不要有任何其他文字。

用户问题：{query}
任务类型："""

    try:
        resp = _llm.chat.completions.create(
            model=MODEL_ROUTER,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip().lower()
        if raw in {"qa", "survey", "code", "full_pipeline"}:
            return raw
    except Exception:
        pass

    # 降级到规则
    return _rule_based_intent(query)


def route_by_intent(state: AgentState) -> str:
    """
    LangGraph conditional_edges 回调函数。
    返回下一个节点名称字符串。
    """
    intent = state.get("intent", "qa")
    routing = {
        "qa":            "researcher",
        "survey":        "researcher",
        "code":          "coder",
        "full_pipeline": "researcher",
    }
    return routing.get(intent, "researcher")
