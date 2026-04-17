"""
agents 包 — 三大 Agent 实现（Phase 2）

langchain-core 1.x 导入规范（验证于 langchain-core==1.2.28）
─────────────────────────────────────────────────────────────

  消息类型:
    from langchain_core.messages import (
        BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
    )

  文档 / 检索:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

  输出解析:
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

  提示词:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

  工具定义:
    from langchain_core.tools import tool          # @tool 装饰器
    from langchain_core.tools import BaseTool, StructuredTool

  LangGraph 状态/图:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph.message import add_messages   # state reducer

  禁止使用（已废弃或移除）:
    x  from langchain.schema import BaseMessage        # 0.3 前旧路径
    x  from langchain.schema import Document           # 同上
    x  from langchain_core.memory import BaseMemory    # 1.x 已移除，用 MemorySaver
"""
