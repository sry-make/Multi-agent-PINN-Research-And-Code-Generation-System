"""
意图路由器 — 解析用户输入，决定走哪条 SOP 分支

意图类型:
    "qa"            → 快速问答（仅 Researcher，无代码）
    "survey"        → 文献综述（Researcher 深度检索）
    "code"          → 纯代码任务（Coder + Examiner）
    "full_pipeline" → 完整 SOP（Researcher → Coder → Examiner）
    "clarify"       → 主动消歧，先向用户追问
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config import (
    MODEL_ROUTER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    ROUTER_AMBIGUITY_MARGIN,
    ROUTER_CONFIDENCE_THRESHOLD,
    ROUTER_ENTROPY_THRESHOLD,
)
from orchestrator.state import AgentState


INTENTS = ("qa", "survey", "code", "full_pipeline")
CLARIFY_INTENT = "clarify"

_llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# 高频显式模式: 命中后零成本直达。
_CODE_KEYWORDS = re.compile(
    r"(代码|code|implement|编写|实现|程序|脚本|script|notebook|demo|示例|调试|debug|报错|bug|修复|运行|run|训练|画图)",
    re.I,
)
_SURVEY_KEYWORDS = re.compile(
    r"(综述|survey|文献|调研|review|overview|最新进展|对比.*论文|研究现状|相关工作)",
    re.I,
)
_QA_KEYWORDS = re.compile(
    r"(什么是|解释|说明|原理|为什么|怎么回事|含义|区别|作用|是什么|如何理解|\?|？)",
    re.I,
)
_FOLLOWUP_CONTEXT_KEYWORDS = re.compile(
    r"(这个|那个|它|上次|之前|前面|继续|接着|刚才|前一个|上述|上面|same one|continue)",
    re.I,
)
_BOTH_CONNECTORS = re.compile(r"(先.*再|并且|同时|以及|然后再|and then|both)", re.I)

_ROUTER_PROMPT = """\
你是一个科研 Agent 的意图路由器。你需要判断用户输入最适合走哪条执行路径。

候选意图只有 4 种:
- qa: 简单知识问答，主要是解释概念、回答问题，不需要系统性综述，也不需要写代码
- survey: 文献综述 / 多篇论文对比 / 研究现状梳理
- code: 需要编写、运行、调试、修改代码
- full_pipeline: 同时需要文献调研和代码实现，或者明确表达“先调研再实现”

请特别注意:
1. 如果用户信息不足、上下文依赖强、或存在隐含意图，不要假装很确定。
2. 你必须给出四个意图的概率分布，四者之和约为 1。
3. 如果你认为需要主动追问，请在 clarification_question 中写一个简洁中文问题。
4. 如果不需要追问，clarification_question 置为空字符串。
5. 只输出 JSON，不要输出任何额外解释。

输出格式:
{
  "intent": "qa|survey|code|full_pipeline",
  "scores": {
    "qa": 0.10,
    "survey": 0.20,
    "code": 0.30,
    "full_pipeline": 0.40
  },
  "reason": "简短解释",
  "missing_information": ["缺少的信息1", "缺少的信息2"],
  "clarification_question": "如果需要追问，在这里给出一句中文问题"
}
"""


@dataclass
class RouterDecision:
    intent: str
    source: str
    confidence: float
    entropy: float
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str = ""
    missing_information: list[str] = field(default_factory=list)
    default_intent: str = "qa"
    defaulted: bool = False

    def legacy_intent(self) -> str:
        if self.intent == CLARIFY_INTENT:
            return self.default_intent or "qa"
        return self.intent


def _normalize_probability(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if number < 0:
        return 0.0
    return number


def _normalize_scores(scores: dict[str, object] | None) -> dict[str, float]:
    normalized = {intent: 0.0 for intent in INTENTS}
    if scores:
        for intent in INTENTS:
            normalized[intent] = _normalize_probability(scores.get(intent, 0.0))

    total = sum(normalized.values())
    if total <= 0:
        fallback = 1.0 / len(INTENTS)
        return {intent: fallback for intent in INTENTS}

    return {intent: value / total for intent, value in normalized.items()}


def _normalized_entropy(scores: dict[str, float]) -> float:
    entropy = 0.0
    for prob in scores.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    max_entropy = math.log2(len(INTENTS))
    if max_entropy <= 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def _top_intents(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _format_reason(label: str, detail: str) -> str:
    detail = str(detail or "").strip()
    return f"{label}: {detail}" if detail else label


def looks_like_contextual_followup(query: str) -> bool:
    normalized = str(query or "").strip()
    return bool(_FOLLOWUP_CONTEXT_KEYWORDS.search(normalized)) or len(normalized) <= 8


def _rule_based_intent(query: str) -> str:
    """兼容旧接口的规则兜底意图识别。"""
    decision = _rule_router(query)
    if decision:
        return decision.intent
    if _QA_KEYWORDS.search(query):
        return "qa"
    return "qa"


def _rule_router(query: str) -> RouterDecision | None:
    """只匹配高频显式模式，命中后直接路由。"""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return RouterDecision(
            intent="qa",
            source="rule",
            confidence=0.95,
            entropy=0.0,
            reason="Rule match: empty query defaults to qa",
            scores={"qa": 1.0, "survey": 0.0, "code": 0.0, "full_pipeline": 0.0},
            default_intent="qa",
        )

    has_code = bool(_CODE_KEYWORDS.search(normalized_query))
    has_survey = bool(_SURVEY_KEYWORDS.search(normalized_query))
    has_connector = bool(_BOTH_CONNECTORS.search(normalized_query))

    if has_code and has_survey:
        return RouterDecision(
            intent="full_pipeline",
            source="rule",
            confidence=0.99,
            entropy=0.0,
            reason="Rule match: explicit research + code keywords",
            scores={"qa": 0.0, "survey": 0.0, "code": 0.0, "full_pipeline": 1.0},
            default_intent="full_pipeline",
        )

    if has_connector and has_code and not has_survey:
        return RouterDecision(
            intent="code",
            source="rule",
            confidence=0.93,
            entropy=0.22,
            reason="Rule match: action-oriented multi-step code request",
            scores={"qa": 0.02, "survey": 0.05, "code": 0.83, "full_pipeline": 0.10},
            default_intent="code",
        )

    if has_code:
        return RouterDecision(
            intent="code",
            source="rule",
            confidence=0.97,
            entropy=0.0,
            reason="Rule match: explicit code/debug keywords",
            scores={"qa": 0.0, "survey": 0.0, "code": 1.0, "full_pipeline": 0.0},
            default_intent="code",
        )

    if has_survey:
        return RouterDecision(
            intent="survey",
            source="rule",
            confidence=0.97,
            entropy=0.0,
            reason="Rule match: explicit survey/literature keywords",
            scores={"qa": 0.0, "survey": 1.0, "code": 0.0, "full_pipeline": 0.0},
            default_intent="survey",
        )

    return None


def _extract_json_object(text: str) -> dict[str, object]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        payload = json.loads(raw[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_clarification_question(
    query: str,
    top_intent: str,
    missing_information: list[str],
) -> str:
    missing_text = ""
    if missing_information:
        missing_text = "我主要还缺少：" + "、".join(item for item in missing_information[:2] if item) + "。"

    if top_intent == "code":
        return (
            "你是希望我直接写并运行代码，还是先解释/调研一下思路？"
            + missing_text
        )
    if top_intent == "survey":
        return (
            "你更想要简短回答，还是要我做一份结构化文献综述？"
            + missing_text
        )
    if top_intent == "full_pipeline":
        return (
            "你是想先做文献调研再实现代码，还是只做其中一部分？"
            + missing_text
        )
    if _FOLLOWUP_CONTEXT_KEYWORDS.search(query):
        return "你说的“这个/上次那个”具体是指哪一部分？你也可以直接说明是要回答问题、做综述，还是写代码。"
    return (
        "你更希望我走哪条路径：1) 直接回答问题 2) 做文献综述 3) 写并运行代码 4) 先调研再实现？"
        + missing_text
    )


def _llm_router(query: str) -> RouterDecision:
    response = _llm.chat.completions.create(
        model=MODEL_ROUTER,
        messages=[
            {"role": "system", "content": _ROUTER_PROMPT},
            {"role": "user", "content": str(query or "").strip()},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    raw = response.choices[0].message.content or ""
    payload = _extract_json_object(raw)

    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        intent = "qa"

    scores = _normalize_scores(payload.get("scores") if isinstance(payload.get("scores"), dict) else {})
    ranked = _top_intents(scores)
    top_intent, confidence = ranked[0]
    entropy = _normalized_entropy(scores)
    reason = str(payload.get("reason", "")).strip()
    missing_information = [
        str(item).strip()
        for item in (payload.get("missing_information") or [])
        if str(item).strip()
    ]
    clarification_question = str(payload.get("clarification_question", "")).strip()

    # 当模型的显式 intent 与概率最高项冲突时，以概率最高项为准。
    intent = top_intent if scores.get(top_intent, 0.0) >= scores.get(intent, 0.0) else intent
    if not clarification_question and missing_information:
        clarification_question = _build_clarification_question(query, top_intent, missing_information)

    return RouterDecision(
        intent=intent,
        source="llm",
        confidence=confidence,
        entropy=entropy,
        reason=reason or "LLM inference",
        scores=scores,
        needs_clarification=False,
        clarification_question=clarification_question,
        missing_information=missing_information,
        default_intent=top_intent,
    )


def _choose_tolerant_default_intent(
    query: str,
    scores: dict[str, float],
) -> str:
    normalized_query = str(query or "").strip()
    has_code = bool(_CODE_KEYWORDS.search(normalized_query))
    has_survey = bool(_SURVEY_KEYWORDS.search(normalized_query))

    if has_code and has_survey:
        return "full_pipeline"
    if has_code:
        return "code"
    if has_survey:
        return "survey"

    ranked = _top_intents(scores)
    if ranked:
        return ranked[0][0]
    return "qa"


def _should_ask_for_clarification(
    query: str,
    llm_decision: RouterDecision,
    default_intent: str,
) -> bool:
    ranked = _top_intents(llm_decision.scores)
    top_score = ranked[0][1] if ranked else 0.0
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    explicit_bias = bool(_CODE_KEYWORDS.search(query) or _SURVEY_KEYWORDS.search(query))
    context_heavy = bool(_FOLLOWUP_CONTEXT_KEYWORDS.search(query))
    short_query = len(str(query or "").strip()) <= 8
    flat_distribution = llm_decision.entropy >= ROUTER_ENTROPY_THRESHOLD
    low_confidence = llm_decision.confidence < ROUTER_CONFIDENCE_THRESHOLD
    near_tie = margin < ROUTER_AMBIGUITY_MARGIN
    has_missing_info = bool(llm_decision.missing_information)

    # 显式代码/综述请求即使不够具体，也优先宽容执行。
    if explicit_bias and default_intent in {"code", "survey", "full_pipeline"}:
        return False

    return (
        (low_confidence and (flat_distribution or near_tie))
        or (context_heavy and (low_confidence or has_missing_info))
        or (short_query and flat_distribution)
    )


def _fallback_router(query: str, reason: str = "") -> RouterDecision:
    default_intent = _rule_based_intent(query)
    scores = {intent: 0.0 for intent in INTENTS}
    scores[default_intent] = 1.0
    return RouterDecision(
        intent=default_intent,
        source="fallback",
        confidence=0.51,
        entropy=0.5,
        reason=reason or "Fallback to tolerant default router",
        scores=scores,
        needs_clarification=False,
        clarification_question="",
        missing_information=[],
        default_intent=default_intent,
        defaulted=True,
    )


def resolve_intent(query: str) -> RouterDecision:
    """
    三层路由:
    1) 高频显式规则直达
    2) LLM 输出概率分布，计算 confidence / entropy
    3) 低置信度时触发主动消歧或宽容默认路径
    """
    rule_decision = _rule_router(query)
    if rule_decision is not None:
        return rule_decision

    try:
        llm_decision = _llm_router(query)
    except Exception as exc:
        return _fallback_router(query, reason=_format_reason("LLM router failed", str(exc)))

    is_high_confidence = (
        llm_decision.confidence >= ROUTER_CONFIDENCE_THRESHOLD
        and llm_decision.entropy <= ROUTER_ENTROPY_THRESHOLD
    )
    if is_high_confidence:
        return llm_decision

    default_intent = _choose_tolerant_default_intent(query, llm_decision.scores)
    if _should_ask_for_clarification(query, llm_decision, default_intent):
        clarification_question = (
            llm_decision.clarification_question
            or _build_clarification_question(query, default_intent, llm_decision.missing_information)
        )
        return RouterDecision(
            intent=CLARIFY_INTENT,
            source="llm_clarify",
            confidence=llm_decision.confidence,
            entropy=llm_decision.entropy,
            reason=llm_decision.reason or "Low-confidence route requires clarification",
            scores=llm_decision.scores,
            needs_clarification=True,
            clarification_question=clarification_question,
            missing_information=llm_decision.missing_information,
            default_intent=default_intent,
            defaulted=False,
        )

    return RouterDecision(
        intent=default_intent,
        source="llm_default",
        confidence=llm_decision.confidence,
        entropy=llm_decision.entropy,
        reason=llm_decision.reason or "Low-confidence route tolerated via default path",
        scores=llm_decision.scores,
        needs_clarification=False,
        clarification_question="",
        missing_information=llm_decision.missing_information,
        default_intent=default_intent,
        defaulted=True,
    )


def detect_intent(query: str) -> str:
    """兼容旧接口：仅返回字符串意图。"""
    return resolve_intent(query).legacy_intent()


def route_by_intent(state: AgentState) -> str:
    """
    LangGraph conditional_edges 回调函数。
    返回下一个节点名称字符串。
    """
    intent = state.get("intent", "qa")
    routing = {
        CLARIFY_INTENT: "clarify",
        "qa": "researcher",
        "survey": "researcher",
        "code": "coder",
        "full_pipeline": "researcher",
    }
    return routing.get(intent, "researcher")
