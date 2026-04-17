"""
全局状态 Schema — LangGraph 图的"共享黑板"

所有 Agent 从此处读写状态，确保信息在节点间无损传递。
TypedDict 定义让 LangGraph 能做静态类型检查。
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph 图全局状态"""

    # ── 输入 ─────────────────────────────────────────────────
    query: str                              # 用户原始问题
    session_id: str                         # 当前会话 ID（与 LangGraph thread_id 对齐）

    # ── 路由 ─────────────────────────────────────────────────
    intent: str                             # 意图类型: "qa" | "survey" | "code" | "full_pipeline"
    current_step: str                       # 当前 SOP 步骤名

    # ── 多轮消息历史（LangGraph 原生 add_messages reducer）──
    messages: Annotated[list, add_messages]

    # ── 记忆层 ───────────────────────────────────────────────
    session_summary: dict[str, Any]         # 当前 session 的压缩摘要
    project_memory: dict[str, Any]          # 项目长期事实 / 约束
    experience_hints: list[dict[str, Any]]  # 相似历史经验提示

    # ── Researcher Agent 输出 ────────────────────────────────
    literature_report: str                  # 文献综述报告
    design_proposal: str                    # 技术方案设计
    retrieved_sources: list[dict[str, Any]] # 检索来源元数据列表

    # ── Coder Agent 输出 ─────────────────────────────────────
    generated_code: str                     # 生成的代码字符串
    execution_stdout: str                   # 沙盒执行标准输出
    execution_stderr: str                   # 沙盒执行标准错误
    execution_success: bool                 # 执行是否成功
    artifact_paths: list[str]               # 本轮运行导出的宿主机产物路径
    code_retry_count: int                   # 当前 debug 重试次数

    # ── Examiner Agent 输出 ──────────────────────────────────
    academic_review: str                    # 学术审查意见
    code_review: str                        # 代码审查意见
    examiner_verdict: str                   # "PASS" | "FAIL"
    examiner_retry_count: int               # Examiner 循环次数

    # ── 最终输出 ─────────────────────────────────────────────
    final_answer: str                       # 汇总后的最终回答

    # ── Token 成本追踪 ───────────────────────────────────────
    total_tokens_used: int                  # 本次会话累计 Token
    token_budget_exceeded: bool             # 是否触发预算限制
