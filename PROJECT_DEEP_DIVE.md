# PINN Agent V2 — 项目深度解析

> 本文档面向希望快速上手本项目的开发者，从架构全景到每个模块的实现细节，逐层拆解。

---

## 目录

1. [项目总览与快速启动](#1-项目总览与快速启动)
2. [系统架构全景](#2-系统架构全景)
3. [多智能体编排 — LangGraph SOP 状态机](#3-多智能体编排--langgraph-sop-状态机)
4. [三个 Agent 的 ReAct 实现](#4-三个-agent-的-react-实现)
5. [工具层详解](#5-工具层详解)
6. [Docker 沙盒执行引擎](#6-docker-沙盒执行引擎)
7. [三层记忆系统](#7-三层记忆系统)
8. [RAG 知识库与重排序](#8-rag-知识库与重排序)
9. [可观测性 — Tracing 与成本控制](#9-可观测性--tracing-与成本控制)
10. [评估系统 — 规则评分 + LLM Judge](#10-评估系统--规则评分--llm-judge)
11. [TUI 终端界面](#11-tui-终端界面)
12. [数据流全链路追踪](#12-数据流全链路追踪)

---

## 1. 项目总览与快速启动

### 一句话定位

Multi-Agent PINN 科研助手：用户提出 PINN 相关问题 → 系统自动完成文献检索、代码生成、沙盒执行、质量审查，最终返回带引用的综述报告和/或可运行的代码。

### 技术栈

| 层 | 技术选型 | 用途 |
|---|---|---|
| 编排 | LangGraph + LangChain | 状态机驱动的多 Agent SOP |
| LLM | Ollama（本地）/ Qwen DashScope（云端） | 通过 OpenAI 兼容接口统一调用 |
| 向量库 | ChromaDB + BAAI/bge-m3 | 本地论文 RAG |
| 重排序 | BAAI/bge-reranker-base (CrossEncoder) | 精排检索结果 |
| 沙盒 | Docker（--network none, 非 root） | 隔离执行生成代码 |
| TUI | Textual + Rich | 终端多面板交互界面 |
| 测试 | pytest + 自建 eval 框架 | 单元/集成/工作流/评估 |

### 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，设置 LLM_PROVIDER 和对应的 API Key/URL

# 3. 构建沙盒镜像（代码执行必需）
docker build -t pinn_agent_sandbox:latest -f sandbox/Dockerfile.sandbox .

# 4. 构建 RAG 知识库（可选，需要 papers/ 目录下有 PDF）
python -m rag.build_memory

# 5. 运行
python main.py              # TUI 模式
python main.py --cli        # CLI 调试模式
python main.py --query "写一个最小 PINN 示例"  # 单次查询
```

### 配置中心：`config.py`

所有配置集中在此文件，禁止其他文件硬编码。核心机制：

```python
# 自动加载 .env 和 .env.local（后者覆盖前者）
_load_env_files()

# 双后端切换：通过 PINN_AGENT_LLM_PROVIDER 环境变量
if LLM_PROVIDER in {"qwen", "dashscope", "bailian"}:
    OLLAMA_BASE_URL = QWEN_BASE_URL      # 指向 DashScope
    MODEL_CODER = _QWEN_MODEL_CODER      # 使用 qwen3-coder-plus
else:
    OLLAMA_BASE_URL = _OLLAMA_BASE_URL_LOCAL  # 指向 localhost:11434
    MODEL_CODER = _LOCAL_MODEL_CODER          # 使用 qwen2.5:7b
```

每个 Agent 角色有独立的模型配置：`MODEL_RESEARCHER`、`MODEL_CODER`、`MODEL_EXAMINER`、`MODEL_ROUTER`，通过 `MODEL_BY_STEP` 字典映射到 SOP 步骤名。

---

## 2. 系统架构全景

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 状态机                          │
│                                                             │
│  parse_intent ──► memory_read ──► [researcher | coder | clarify] │
│                                        │                    │
│                                        ▼                    │
│                                    examiner                 │
│                                   ╱        ╲                │
│                              PASS            FAIL           │
│                               │          (重试≤3次)          │
│                               ▼              │              │
│                          synthesize ◄────────┘              │
│                               │                             │
│                               ▼                             │
│                        memory_writeback ──► END             │
└─────────────────────────────────────────────────────────────┘
  │                    │                │
  ▼                    ▼                ▼
┌──────┐        ┌──────────┐     ┌──────────┐
│ 工具层 │        │ Docker   │     │ 三层记忆  │
│ RAG  │        │ 沙盒     │     │ Session  │
│ arXiv│        │ 代码执行  │     │ Project  │
│ Web  │        │          │     │ Experience│
│ 公式  │        └──────────┘     └──────────┘
└──────┘
```

项目分为 8 个模块目录，每个目录职责单一：

| 目录 | 职责 | 核心文件 |
|---|---|---|
| `orchestrator/` | SOP 状态机定义与路由 | `graph.py`, `state.py`, `router.py` |
| `agents/` | 三个 Agent 的 ReAct 实现 | `researcher.py`, `coder.py`, `examiner.py` |
| `tools/` | Agent 可调用的工具函数 | `code_tools.py`, `rag_tools.py`, `search_tools.py`, `formula_tools.py` |
| `sandbox/` | Docker 隔离执行引擎 | `docker_runner.py`, `Dockerfile.sandbox` |
| `memory/` | 三层记忆系统 | `session_manager.py`, `project_store.py`, `experience_store.py` |
| `rag/` | 向量知识库构建与检索 | `build_memory.py`, `reranker.py` |
| `observability/` | 追踪与成本控制 | `tracer.py`, `cost_tracker.py` |
| `eval/` | 评估框架 | `runner.py`, `judge.py`, `rubrics.py`, `report.py` |
| `tui/` | 终端交互界面 | `app.py` |

---

## 3. 多智能体编排 — LangGraph SOP 状态机

这是整个系统的"中枢神经"，理解它就理解了项目的骨架。

### 3.1 共享状态：AgentState（`orchestrator/state.py`）

所有 Agent 通过一个 TypedDict "共享黑板"传递数据，不存在 Agent 之间的直接调用：

```python
class AgentState(TypedDict):
    # ── 输入 ──
    query: str                    # 用户原始问题
    session_id: str               # 会话 ID

    # ── 路由 ──
    intent: str                   # "qa" | "survey" | "code" | "full_pipeline" | "clarify"
    current_step: str             # 当前 SOP 步骤名
    router_source: str            # rule | llm | llm_default | llm_clarify | session_default | fallback
    router_confidence: float      # 最大意图概率
    router_entropy: float         # 归一化熵（越高越不确定）
    router_reason: str            # 路由解释
    router_scores: dict[str, float]  # 四类意图的概率分布
    router_needs_clarification: bool
    router_clarification_question: str
    router_missing_information: list[str]
    router_default_intent: str    # 低置信度时的宽容默认路径
    router_defaulted: bool        # 是否走了宽容默认

    # ── 多轮消息历史 ──
    messages: Annotated[list, add_messages]  # LangGraph 原生 reducer

    # ── 记忆层 ──
    session_summary: dict         # 当前 session 的压缩摘要
    project_memory: dict          # 项目长期事实
    experience_hints: list[dict]  # 相似历史经验

    # ── Researcher 输出 ──
    literature_report: str        # 文献综述
    design_proposal: str          # 技术方案
    retrieved_sources: list[dict] # 检索来源

    # ── Coder 输出 ──
    generated_code: str           # 生成的代码
    execution_stdout: str         # 沙盒标准输出
    execution_stderr: str         # 沙盒标准错误
    execution_success: bool       # 执行是否成功
    artifact_paths: list[str]     # 导出的产物路径
    code_retry_count: int         # debug 重试次数

    # ── Examiner 输出 ──
    academic_review: str          # 学术审查意见
    code_review: str              # 代码审查意见
    examiner_verdict: str         # "PASS" | "FAIL"
    examiner_retry_count: int     # 审查循环次数

    # ── 最终输出 ──
    final_answer: str             # 汇总后的最终回答
    total_tokens_used: int        # 累计 Token
    token_budget_exceeded: bool   # 是否超预算
```

关键设计：`messages` 字段使用 LangGraph 的 `add_messages` reducer，支持消息追加和删除（用于历史压缩）。

### 3.2 意图路由（`orchestrator/router.py`）

V2 早期版本的 Router 只是“LLM 输出一个标签，失败时回退到正则”。为了处理**信息不足、上下文依赖强、隐含意图明显**的 query，当前版本已经重写为一个**三层结构化路由器**：

```text
用户 Query
  ↓
规则 Router（快速匹配高频显式模式）
  ├─ 命中 → 直接路由（零成本、零延迟）
  └─ 未命中 → LLM Router（输出意图分布）
                  ├─ 高置信度 → 直接路由
                  └─ 低置信度 → 主动消歧 / 宽容默认
```

对外的主路径仍然是：

```
"qa"            → Researcher only（快速问答）
"survey"        → Researcher only（深度文献综述）
"code"          → Coder → Examiner（纯代码任务）
"full_pipeline" → Researcher → Coder → Examiner（完整 SOP）
```

但内部新增了一个临时意图态：

```python
"clarify"       → 进入 clarify 节点，先向用户追问
```

### 3.2.1 RouterDecision：结构化路由结果

路由器不再只返回一个字符串，而是先返回一个结构化决策对象：

```python
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
```

核心字段含义：
- `scores`：四类意图 `qa/survey/code/full_pipeline` 的概率分布
- `confidence`：最大类别概率
- `entropy`：对分布做归一化 Shannon entropy，越接近 `1` 代表越不确定
- `clarification_question`：需要主动消歧时给用户的追问
- `default_intent`：低置信度场景下的“宽容执行”默认路径

为了兼容旧调用点，`detect_intent(query)` 仍然保留，但它只是对 `resolve_intent(query)` 的薄包装，会返回兼容旧逻辑的字符串意图。

### 3.2.2 第一层：规则 Router（高频显式模式直达）

规则层只负责匹配**高频、显式、低歧义**的模式，不做模糊猜测：

- 同时命中代码关键词和综述关键词 → `full_pipeline`
- 命中代码 / 调试 / 修复 / 运行 / 训练等关键词 → `code`
- 命中文献 / 综述 / 调研 / review / overview 等关键词 → `survey`

示意代码：

```python
def _rule_router(query: str) -> RouterDecision | None:
    has_code = bool(_CODE_KEYWORDS.search(query))
    has_survey = bool(_SURVEY_KEYWORDS.search(query))

    if has_code and has_survey:
        return RouterDecision(intent="full_pipeline", source="rule", confidence=0.99, ...)
    if has_code:
        return RouterDecision(intent="code", source="rule", confidence=0.97, ...)
    if has_survey:
        return RouterDecision(intent="survey", source="rule", confidence=0.97, ...)
    return None
```

这层的优势是：
- 零成本、零延迟
- 误判风险低
- 能覆盖大量中文口语里的显式任务表达

### 3.2.3 第二层：LLM Router（意图分布 + 置信度 + 熵）

规则层未命中时，才调用 LLM Router。与旧版本“只输出一个标签”不同，当前 Prompt 强制模型输出 JSON，内容包括：

- `intent`
- `scores`
- `reason`
- `missing_information`
- `clarification_question`

LLM Router 的输出流程是：

1. 解析 JSON  
2. 对四类分数做归一化  
3. 计算 `confidence` 和 `entropy`  
4. 如果模型写出来的 `intent` 与最大概率项冲突，则以最大概率项为准

核心逻辑：

```python
scores = _normalize_scores(payload["scores"])
ranked = _top_intents(scores)
top_intent, confidence = ranked[0]
entropy = _normalized_entropy(scores)
```

高置信度阈值由配置决定：

```python
ROUTER_CONFIDENCE_THRESHOLD = 0.68
ROUTER_ENTROPY_THRESHOLD    = 0.82
ROUTER_AMBIGUITY_MARGIN     = 0.10
```

当 `confidence` 足够高、`entropy` 足够低时，LLM 结果直接生效。

### 3.2.4 第三层：主动消歧 + 宽容默认

如果 LLM 路由结果不够确定，系统不会强行拍脑袋，而是进入第三层。

**A. 主动消歧（clarify）**

以下情况会优先追问：
- 置信度低，且分布很平
- 第一、第二候选意图分差过小
- query 本身是“这个 / 上次那个 / 继续”这类强上下文表达
- LLM 显式指出缺少关键信息

这时会返回类似：

```python
RouterDecision(
    intent="clarify",
    source="llm_clarify",
    needs_clarification=True,
    clarification_question="你是希望我先解释一下，还是直接写并运行代码？",
    default_intent="qa",
)
```

**B. 宽容默认（tolerant default）**

有些请求虽然信息不完整，但行动导向很强，例如“帮我修一下这个 PINN 报错”。这类场景下系统会尽量继续执行，而不是过度打断用户。

默认路径选择逻辑：
- 同时含代码 + 综述信号 → `full_pipeline`
- 只含代码信号 → `code`
- 只含综述信号 → `survey`
- 否则回落到概率最高项

### 3.2.5 多轮 Follow-up 的 session-aware 路由修正

低置信度场景还有一个额外优化：如果当前 query 像“继续上次那个”“这个再改一下”这种**上下文型 follow-up**，并且本轮初判进入 `clarify`，`memory_read` 节点会尝试复用 `session_summary.last_intent`。

也就是说：
- Router 先谨慎地判断“这句话单独看不够明确”
- 但记忆层发现上一轮明确是 `code`
- 那么系统会在读取 session memory 后，把当前意图从 `clarify` 修正回 `code`

这让多轮对话里的“继续修刚才的代码”更自然，减少了不必要的追问。

`route_by_intent()` 仍然是 LangGraph 条件边回调，但现在多了 `clarify` 分支：

```python
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "qa")
    return {
        "clarify": "clarify",
        "qa": "researcher",
        "survey": "researcher",
        "code": "coder",
        "full_pipeline": "researcher",  # 先走 Researcher
    }.get(intent, "researcher")
```

### 3.3 图构建与节点注册（`orchestrator/graph.py`）

`build_graph()` 是整个系统的组装入口。因为 Router 新增了主动消歧能力，图结构也从原来的 7 个节点扩展为 8 个节点：

```python
def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    # 注册 8 个节点
    builder.add_node("parse_intent",    node_parse_intent)
    builder.add_node("memory_read",     node_memory_read)
    builder.add_node("clarify",         node_clarify)
    builder.add_node("researcher",      node_researcher)
    builder.add_node("coder",           node_coder)
    builder.add_node("examiner",        node_examiner)
    builder.add_node("synthesize",      node_synthesize)
    builder.add_node("memory_writeback", node_memory_writeback)

    # 起点
    builder.set_entry_point("parse_intent")

    # 固定边
    builder.add_edge("parse_intent", "memory_read")
    builder.add_edge("coder", "examiner")
    builder.add_edge("clarify", "memory_writeback")
    builder.add_edge("synthesize", "memory_writeback")
    builder.add_edge("memory_writeback", END)

    # 条件边：memory_read 之后根据意图路由
    builder.add_conditional_edges("memory_read", route_by_intent,
        {"clarify": "clarify", "researcher": "researcher", "coder": "coder"})

    # 条件边：researcher 之后
    builder.add_conditional_edges("researcher", _after_researcher,
        {"coder": "coder", "examiner": "examiner"})

    # 条件边：examiner 之后（核心重试逻辑）
    builder.add_conditional_edges("examiner", _after_examiner,
        {"synthesize": "synthesize", "coder": "coder", "researcher": "researcher"})

    return builder.compile(checkpointer=checkpointer or MemorySaver())
```

### 3.4 条件边的路由逻辑

**Researcher 之后**：`full_pipeline` 意图继续走 Coder，其他意图直接走 Examiner 审查：

```python
def _after_researcher(state):
    return "coder" if state.get("intent") == "full_pipeline" else "examiner"
```

**Examiner 之后**（重试核心）：

```python
def _after_examiner(state):
    verdict = state.get("examiner_verdict", "PASS")
    examiner_tries = state.get("examiner_retry_count", 0)

    # PASS 或超过最大重试次数 → 结束
    if verdict == "PASS" or examiner_tries >= EXAMINER_MAX_RETRIES:  # 默认 3 次
        return "synthesize"

    # FAIL → 根据意图决定重试哪个 Agent
    intent = state.get("intent", "qa")
    return "coder" if intent in ("code", "full_pipeline") else "researcher"
```

这形成了一个 Examiner ↔ Coder/Researcher 的重试闭环，最多循环 3 次。

### 3.5 节点实现模式

每个节点都是一个薄包装函数，通过延迟导入调用对应 Agent：

```python
def node_researcher(state: AgentState) -> dict:
    from agents.researcher import run_researcher  # 延迟导入，避免循环依赖
    return run_researcher(state)
```

节点返回的 dict 会被 LangGraph 自动合并到 AgentState 中（只更新返回的字段）。

**`node_parse_intent`** 现在除了识别意图，还会把完整的结构化路由信息写回状态，包括：
- `router_source`
- `router_confidence`
- `router_entropy`
- `router_reason`
- `router_scores`
- `router_needs_clarification`
- `router_clarification_question`
- `router_default_intent`

这样后续节点、trace、TUI 甚至 memory 写回都能知道“为什么这样路由”。

**`node_memory_read`** 在执行主任务前读取三层记忆，并执行消息历史压缩（详见第 7 节）。此外它还承担一个 Router 相关职责：
- 如果当前意图是 `clarify`
- 且 query 看起来像上下文型 follow-up
- 则尝试从 `session_summary.last_intent` 继承上一轮主意图

这是 Router 和记忆层协同工作的关键点。

**`node_clarify`** 是新增节点。它不会调用 Researcher 或 Coder，而是直接生成一段“路由澄清”回答，内容包括：
- 当前置信度
- 当前熵
- 缺少的关键信息
- 主动追问句
- 默认宽容路径提示

**`node_synthesize`** 将所有 Agent 输出拼接为 Markdown 格式的最终回答，包含：文献综述、技术方案、代码实现、运行结果、产物文件、审查结果。

**`node_memory_writeback`** 将本轮结果写回会话摘要和经验库（详见第 7 节）。

### 3.6 图的调用方式

所有调用都必须提供 `thread_id`（对应 `session_id`），这是 LangGraph MemorySaver 的要求：

```python
graph = build_graph()
result = await graph.ainvoke(
    {"query": query, "messages": [], "session_id": session_id},
    config={"configurable": {"thread_id": session_id}},
)
```

---

## 4. 三个 Agent 的 ReAct 实现

三个 Agent 共享相同的 ReAct（Reasoning + Acting）循环模式，但各自有不同的工具集、系统提示和迭代上限。

### 4.1 Researcher Agent（`agents/researcher.py`）

**角色**：科研大脑，负责文献检索和方案设计。

**工具集**（5 个）：
- `search_local_papers` — 本地 ChromaDB 论文检索（优先使用）
- `search_arxiv` — arXiv 在线搜索
- `web_search` — DuckDuckGo/SerpAPI 网络搜索
- `simplify_formula` — SymPy 公式化简
- `latex_to_sympy` — LaTeX 转 SymPy

**系统提示核心守则**：
1. 禁止幻觉：所有学术观点必须有工具检索的原文支撑
2. 工具优先：回答前必须先调用工具检索
3. 严格引用：格式为 `[来源: 文件名 或 arXiv:XXXX.XXXXX]`
4. 公式规范：使用标准 LaTeX，损失函数写作 `$\mathcal{L}_{u}$`、`$\mathcal{L}_{f}$`

**ReAct 循环**（最多 5 次迭代）：

```python
def _react_loop(messages):
    for i in range(MAX_ITER):  # MAX_ITER = 5
        # 1. 调用 LLM（绑定了工具）
        response = llm_with_tools.invoke(messages)

        # 2. 记录 token 和 trace
        cost_tracker.record("Researcher", MODEL_RESEARCHER, tokens)
        tracer.log_llm_call(...)

        # 3. 如果 LLM 没有调用工具 → 返回最终回答
        if not response.tool_calls:
            return response.content, messages + [response], sources

        # 4. 执行每个工具调用
        messages.append(response)
        for tool_call in response.tool_calls:
            fn = _TOOL_MAP[tool_call["name"]]
            result = fn.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

            # 记录检索来源
            if tool_call["name"] in ("search_local_papers", "search_arxiv"):
                sources.append({"tool": tool_call["name"], "query": tool_call["args"]["query"]})

    # 5. 达到迭代上限 → 强制总结（不带工具调用）
    force_msg = HumanMessage(content="请基于以上所有检索结果，给出完整的最终综述回答。不要再调用工具。")
    final = llm_plain.invoke(messages + [force_msg])
    return final.content, messages + [force_msg, final], sources
```

**`run_researcher()` 入口逻辑**：

根据意图组装不同的用户消息：
- `qa`：直接传入用户问题
- `survey`：追加结构化综述指令（要求分 3 节：研究现状、关键方法对比、未来方向）
- `full_pipeline`：追加综述指令 + 技术方案设计指令

如果意图是 `full_pipeline`，会尝试从 LLM 回答中分割出 `literature_report` 和 `design_proposal`（以 `## 方案设计` 为分隔符）。

### 4.2 Coder Agent（`agents/coder.py`）

**角色**：编程执行者，负责代码生成、沙盒运行和错误修复。

**工具集**（4 个）：
- `execute_python` — 在 Docker 沙盒中执行 Python 代码
- `read_file` — 读取项目文件
- `write_file` — 写入文件到项目目录
- `run_shell` — 执行白名单 Shell 命令

**系统提示核心职责**：
1. 编写高质量、可直接运行的 Python 代码
2. 写完代码后必须用 `execute_python` 运行验证
3. 执行失败时分析 stderr 并修复
4. 成功后用 `write_file` 保存到 `outputs/` 目录
5. 沙盒边界意识：`execute_python` 运行在独立的 `/workspace` 中，不要先 `write_file` 到宿主机再在沙盒里读取

**ReAct 循环**（最多 4 次迭代：1 次生成 + 3 次 debug）：

与 Researcher 的循环结构相同，但增加了代码提取和执行结果解析：

```python
# 从 LLM 回答中提取代码块
code = _extract_code_block(response.content)

# 执行 execute_python 后解析结果
if tool_call["name"] == "execute_python":
    success, stdout, stderr, artifacts = _parse_execution_result(result)
```

**代码提取链**（`_extract_code_block`）：

LLM 返回的代码格式不固定，提取逻辑按优先级尝试：
1. ` ```python ` 代码块
2. ` ```py ` 代码块
3. ` ``` ` 通用代码块
4. 取最后一个匹配（LLM 倾向于把最终版本放在最后）
5. 对提取结果做 `_normalize_code_candidate` 处理：
   - 检测是否是 JSON 格式的工具调用 payload，如果是则提取 `code` 字段
   - 找到第一行 Python 代码（以 `import`/`from`/`def`/`class`/`@`/`if __name__` 开头）
   - 用 `_looks_like_python_source` 验证是否像 Python 源码

**重试机制**：

当 `code_retry_count > 0` 时（由 Examiner FAIL 触发），Coder 使用专门的重试模板：

```python
_RETRY_TEMPLATE = """
上一次代码执行失败，请分析错误并修复。

【失败代码】
```python
{code}
```

【错误信息（stderr）】
```
{stderr}
```

这是第 {retry_num} 次修复尝试（最多 {max_retries} 次）。
请直接给出修复后的完整代码并重新执行，不要只修改片段。
"""
```

**`run_coder()` 入口逻辑**：

- 首次调用（retry_count=0）：组装记忆上下文 + 文献报告（前 1000 字符）+ 设计方案（前 800 字符）+ 用户需求
- 重试调用（retry_count>0）：使用 `_RETRY_TEMPLATE` 填入失败代码和错误信息



### 4.3 Examiner Agent（`agents/examiner.py`）

**角色**：质量门禁，负责学术审查和代码审查，决定 PASS/FAIL。

**无工具**：Examiner 不调用外部工具，而是对其他 Agent 的输出做规则检查 + LLM 深度审查。

**审查流程**（双轨并行）：

```
                    ┌─── 学术审查轨 ───┐
                    │                  │
输入 ──► 规则预检 ──┤                  ├──► 合并裁决
                    │                  │
                    └─── 代码审查轨 ───┘
```

**学术审查轨**（当 `literature_report` 存在或意图为 qa/survey 时触发）：

1. **规则预检** `_rule_check_academic()`：
   - 报告长度 > 50 字符
   - 严格模式下：必须包含 `[来源:...]`、`arXiv` 或 `[数字]` 引用标记
   - 预检失败 → 直接 FAIL，跳过 LLM 审查

2. **LLM 深度审查** `_llm_review_academic()`：
   - 系统提示："你是严格的学术审查专家，只输出审查意见。"
   - 审查维度：引用真实性、公式规范性、逻辑严谨性、完整性
   - 输入截断：报告最多 4000 字符（`EXAMINER_REVIEW_MAX_REPORT_CHARS`）

**代码审查轨**（当 `generated_code` 存在或意图为 code/full_pipeline 时触发）：

1. **规则预检** `_rule_check_code()`：
   - 代码长度 > 10 字符
   - **危险代码检测**：正则匹配 `os.system`、`subprocess.call`、`shutil.rmtree`、`rm -rf`、`eval`、`exec`、`__import__` → 直接 FAIL
   - 执行失败且 stderr 包含非警告错误 → FAIL
   - 警告过滤：`DeprecationWarning`、`FutureWarning` 等不算错误

2. **快速通过优化**：如果代码执行成功且 `EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS=false`（默认），跳过 LLM 审查直接 PASS

3. **LLM 深度审查** `_llm_review_code()`：
   - 审查维度：逻辑正确性、运行可靠性、科学合理性
   - 输入截断：代码 12000 字符、stdout/stderr 各 1200 字符

**裁决提取** `_extract_verdict()`：
- 优先查找 `[FAIL]` 或 `[PASS]` 标记
- 兜底：检查负面信号词（捏造、幻觉、错误、危险、hallucin）→ FAIL
- 默认：PASS（宽容兜底）

**最终裁决**：两轨的裁决列表中只要有一个 FAIL，最终就是 FAIL。

**输入截断机制** `_clip_review_text()`：

为避免截断导致 LLM 误判半截代码，采用"头 70% + 尾 30%"的截断策略：

```python
def _clip_review_text(text, limit, label):
    if len(text) <= limit:
        return text
    head_len = int(limit * 0.70)
    tail_len = limit - head_len - 60  # 60 字符留给省略标记
    return text[:head_len] + f"\n\n... [{label} 中间省略] ...\n\n" + text[-tail_len:]
```

---

## 5. 工具层详解

### 5.1 代码工具（`tools/code_tools.py`）

**`execute_python(code, timeout=30)`**：
- 在 Docker 沙盒中执行 Python 代码
- 超时硬上限 60 秒
- 返回格式化字符串：`[执行成功/失败]`、stdout、stderr、artifacts 列表
- Docker 不可用时返回 `[运行时错误]`

**`read_file(path)`**：
- 路径安全检查：`_safe_path()` 使用 `Path.is_relative_to()` 防止目录穿越
- 内容截断：最多 4000 字符

**`write_file(path, content)`**：
- 同样经过 `_safe_path()` 检查
- 自动创建父目录

**`run_shell(cmd)`**：
- 白名单机制：只允许 `ls, cat, head, tail, pwd, python, pip, mkdir, cp, mv, echo`
- 操作符限制：禁止 `||`、`;`、`|`（管道），只允许 `&&` 链接
- 在 Docker 沙盒中执行（非宿主机）

**路径安全** `_safe_path()`：

```python
def _safe_path(path: str) -> Path:
    resolved = (Path(_ALLOWED_ROOT) / path).resolve()
    if not resolved.is_relative_to(Path(_ALLOWED_ROOT).resolve()):
        raise PermissionError(f"路径越界: {path}")
    return resolved
```

### 5.2 RAG 工具（`tools/rag_tools.py`）

**`search_local_papers(query, mode="hyde_reranker")`**：

三种检索模式：
| 模式 | HyDE 扩展 | BGE 重排序 | 精度 | 速度 |
|---|---|---|---|---|
| `hyde_reranker` | ✓ | ✓ | 最高 | 最慢 |
| `hyde` | ✓ | ✗ | 中等 | 中等 |
| `direct` | ✗ | ✗ | 最低 | 最快 |

调用 `rag.build_memory.retrieve_context()` 完成实际检索，返回格式化的片段列表：

```
【片段 1 | 来源: paper_name.pdf】
检索到的文本内容...

【片段 2 | 来源: another_paper.pdf】
...
```

### 5.3 搜索工具（`tools/search_tools.py`）

**`search_arxiv(query, max_results=5)`**：
- 使用 `arxiv` Python SDK
- 支持按相关性或更新时间排序
- 返回：标题、作者（前 3 位 + et al.）、年份、摘要、链接

**`web_search(query, max_results=5)`**：
- 优先使用 SerpAPI（如果配置了 `SERPAPI_KEY`）
- 否则降级到 DuckDuckGo（免费，无需 API Key）
- SerpAPI 失败时也会降级到 DuckDuckGo

### 5.4 公式工具（`tools/formula_tools.py`）

| 工具 | 功能 | 示例 |
|---|---|---|
| `simplify_formula(expr)` | 化简数学表达式 | `x**2 + 2*x + 1` → `(x+1)**2` |
| `solve_equation(expr, var)` | 解方程（expr=0） | `x**2 - 4` → `[-2, 2]` |
| `differentiate(expr, var, order)` | 求偏导数 | `x**2*y` 对 x → `2*x*y` |
| `latex_to_sympy(latex_expr)` | LaTeX 转 SymPy | `\frac{x}{y}` → `x/y` |

---

## 6. Docker 沙盒执行引擎

### 6.1 沙盒镜像（`sandbox/Dockerfile.sandbox`）

```dockerfile
FROM python:3.11-slim

# 科学计算依赖（固定版本）
RUN pip install --no-cache-dir \
    numpy==1.26.4  scipy==1.13.0  matplotlib==3.8.4 \
    sympy==1.12  torch==2.3.0+cpu

WORKDIR /workspace

# 安全加固：非 root 用户
RUN useradd -m -u 1001 sandbox_user
USER sandbox_user
```

注意：沙盒使用 CPU 版 PyTorch，与宿主机的 GPU 版本隔离。

### 6.2 DockerSandbox 类（`sandbox/docker_runner.py`）

**单例模式**：通过 `get_sandbox()` 获取全局唯一实例。

**`run_python(code, timeout=30, artifact_root=None)`** 执行流程：

```
1. 创建唯一导出目录: outputs/sandbox_runs/run_20260421_143022_a1b2c3d4/
2. 创建临时运行目录: /tmp/pinn_agent_run_xxxxx/
   ├── workspace/solution.py  ← 写入用户代码
   └── container_tmp/         ← 映射到容器 /tmp
3. 启动 Docker 容器:
   docker run --rm --network none \
     --cpus="2" --memory="2g" \
     -v workspace:/workspace:rw \
     -v container_tmp:/tmp:rw \
     --user 1001:1001 \
     pinn_agent_sandbox:latest \
     python /workspace/solution.py
4. 等待执行完成（超时则 kill）
5. 提取 stdout 和 stderr
6. 导出产物（/workspace 和 /tmp 中除 solution.py 外的所有文件）
7. 清理临时目录
```

**安全隔离措施**：

| 措施 | 配置 | 说明 |
|---|---|---|
| 网络隔离 | `network_mode="none"` | 完全断网，无法外连 |
| CPU 限制 | `cpu_period=100000, cpu_quota=200000` | 最多 2 核 |
| 内存限制 | `mem_limit="2g"` | 最多 2GB |
| 用户隔离 | `user="1001:1001"` | 非 root 执行 |
| 超时控制 | `container.wait(timeout=30)` | 超时强制 kill |
| 文件隔离 | 临时目录挂载 | 不挂载宿主机项目目录 |

**产物导出** `_export_runtime_artifacts()`：

```python
# 导出范围：
# - /workspace 下除 solution.py 外的所有文件（代码生成的图片、日志等）
# - /tmp 下的所有文件
# 保持相对目录结构，使用 shutil.copy2 保留元数据
# 如果没有产物，删除空的导出目录
```

**`run_command(command, timeout, mount_dir)`**：

用于 `run_shell` 工具，执行预分词的命令列表。如果提供了 `mount_dir`，会将其挂载到容器的 `/workspace/project`。

---

## 7. 三层记忆系统

记忆系统是本项目的一大亮点，分为短期、长期、经验三层，各自解决不同的问题。

### 7.1 架构总览

```
┌─────────────────────────────────────────────────────┐
│                    记忆层                            │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Session     │  │ Project      │  │ Experience │ │
│  │ 会话记忆    │  │ 项目记忆     │  │ 经验记忆   │ │
│  │             │  │              │  │            │ │
│  │ 短期/压缩   │  │ 长期/人工    │  │ 学习/去重  │ │
│  │ 每会话独立  │  │ 全局共享     │  │ 全局共享   │ │
│  │             │  │              │  │            │ │
│  │ sessions/   │  │ project_     │  │ experience_│ │
│  │ <id>.json   │  │ memory.json  │  │ db.jsonl   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│                                                     │
│  读取时机: memory_read 节点（图执行前）              │
│  写回时机: memory_writeback 节点（图执行后）         │
└─────────────────────────────────────────────────────┘
```

### 7.2 会话记忆（`memory/session_manager.py`）

**职责**：跟踪当前会话的上下文，支持消息历史压缩。

**会话摘要结构**（`default_session_summary()`）：

```python
{
    "session_id": "cli-20260421-143022-a1b2",
    "created_at": "2026-04-21T14:30:22",
    "user_goal": "",                    # 用户目标（从首次查询推断）
    "recent_queries": [],               # 最近 6 条查询
    "last_intent": "",                  # 最近一次意图
    "last_research_summary": "",        # 最近一次文献综述摘要
    "last_code_summary": "",            # 最近一次代码执行摘要
    "last_examiner_summary": "",        # 最近一次审查摘要
    "last_artifacts": [],               # 最近一次产物路径
    "open_todos": [],                   # 待办事项
    "conversation_digest": "",          # 压缩后的对话摘要
    "compressed_turns": 0,              # 已压缩的轮次数
    "message_window_size": 0,           # 当前消息窗口大小
    # 代码相关的细粒度记忆
    "last_code_snippet": "",            # 最近一次代码片段
    "last_error_summary": "",           # 最近一次错误摘要
    "last_successful_code_snippet": "", # 最近一次成功的代码
    "last_failed_code_snippet": "",     # 最近一次失败的代码
    "last_successful_artifacts": [],    # 成功的产物列表
    "last_failure_error_summary": "",   # 失败的错误摘要
}
```

**消息历史压缩**（`compress_message_history()`）：

这是避免 Token 爆炸的关键机制。在每轮新查询开始时：

```python
def compress_message_history(messages, session_summary):
    # 1. 从消息历史中提取摘要行
    digest_lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        content = str(msg.content)[:200]
        digest_lines.append(f"[{role}] {content}")

    # 2. 如果行数超过 MAX_DIGEST_LINES(8)，采样头尾
    if len(digest_lines) > MAX_DIGEST_LINES:
        head = digest_lines[:4]
        tail = digest_lines[-3:]
        digest_lines = head + ["... (中间省略) ..."] + tail

    # 3. 追加到 conversation_digest（限制 1200 字符）
    new_digest = "\n".join(digest_lines)
    session_summary["conversation_digest"] = (
        session_summary.get("conversation_digest", "") + "\n---\n" + new_digest
    )[-CONVERSATION_DIGEST_LIMIT:]

    # 4. 清空消息窗口，递增压缩计数
    session_summary["compressed_turns"] += 1
    return [], session_summary, True  # 返回空消息列表
```

压缩后，原始消息被清空，摘要保存在 `conversation_digest` 中。Agent 通过 `format_session_summary()` 读取这个摘要作为上下文。

**会话摘要构建**（`build_session_summary()`）：

每轮执行后，从 AgentState 中提取关键信息更新摘要：

```python
def build_session_summary(previous_summary, state):
    summary = dict(previous_summary)

    # 追加最近查询（最多保留 6 条）
    summary["recent_queries"].append(state["query"])
    summary["recent_queries"] = summary["recent_queries"][-MAX_RECENT_QUERIES:]

    # 更新代码摘要（区分成功/失败）
    if state.get("generated_code"):
        if state.get("execution_success"):
            summary["last_successful_code_snippet"] = state["generated_code"][:CODE_SNIPPET_CHAR_LIMIT]
            summary["last_successful_artifacts"] = state.get("artifact_paths", [])
        else:
            summary["last_failed_code_snippet"] = state["generated_code"][:CODE_SNIPPET_CHAR_LIMIT]
            summary["last_failure_error_summary"] = state.get("execution_stderr", "")[:SUMMARY_TEXT_LIMIT]

    # 更新审查摘要
    if state.get("examiner_verdict"):
        summary["last_examiner_summary"] = f"Verdict: {state['examiner_verdict']}"

    # 更新待办事项
    if state.get("examiner_verdict") == "FAIL" or not state.get("execution_success"):
        summary["open_todos"].append("修复上一轮的问题")

    return summary
```

**代码记忆格式化**（`format_code_memory()`）：

专门为 Coder 和 Examiner 提供的代码上下文视图：

```
【上次代码运行】成功/失败
【上次真实错误】ImportError: No module named 'xxx'
【上次代码片段】(前 900 字符)
【上次成功基线】(前 900 字符)
【上次失败尝试】(前 900 字符)
【成功产物】['loss.png', 'train_log.txt']
```

### 7.3 项目记忆（`memory/project_store.py`）

**职责**：存储项目级别的长期事实和决策，跨会话共享。

**默认项目记忆结构**：

```python
{
    "memory_version": 2,
    "project_name": "PINN Agent V2",
    "goal": "面试级 Multi-Agent 科研 + 编码助手",
    "architecture_rules": [
        "LangGraph 编排 SOP",
        "Textual TUI 交互",
        "Docker 沙盒执行",
        "可解释决策链",
    ],
    "tech_stack": ["Python", "LangGraph", "LangChain", "Textual", "ChromaDB", "Docker"],
    "current_priorities": [
        "稳定会话记忆",
        "保持演示路径畅通",
        "展示多 Agent 协作",
    ],
    "known_risks": [
        "沙盒边界（run_shell 白名单 vs Docker 隔离）",
        "长期记忆演化",
        "TUI 性能",
    ],
    "coding_preferences": ["轻量本地优先", "结构化输出", "显式权衡"],
    "decisions": [
        {
            "id": "use_langgraph",
            "summary": "选择 LangGraph 而非 AutoGen/CrewAI",
            "rationale": "显式状态机，可调试性强",
            "status": "accepted",
        }
    ],
    "rejected_options": [
        {
            "option": "AutoGen 多 Agent 框架",
            "reason": "隐式消息传递，难以调试和控制流程",
        }
    ],
}
```

**API**：
- `load_project_memory()` — 从 JSON 加载，缺失字段自动补全
- `save_project_memory()` — 保存并更新时间戳
- `record_project_decision()` — 记录架构决策（按 ID 去重）
- `record_rejected_option()` — 记录被否决的方案
- `format_project_memory()` — 渲染为 Agent 可读的文本块

### 7.4 经验记忆（`memory/experience_store.py`）

**职责**：从历史运行中学习，记录成功/失败模式，为未来查询提供经验提示。

**经验记录结构**：

```python
{
    "fingerprint": "code::missing_torch::写一个最小",  # 去重指纹
    "session_id": "cli-20260421-143022",
    "query": "写一个最小 PINN 示例",
    "intent": "code",
    "error_type": "missing_torch",        # 自动推导的错误类型
    "symptom": "ModuleNotFoundError: No module named 'torch'",
    "resolution_hint": "沙盒镜像已包含 torch，检查 import 路径",
    "tags": ["code", "missing_torch", "execution_failed"],
    "occurrence_count": 3,                # 出现次数
    "success_count": 1,                   # 成功次数
    "failure_count": 2,                   # 失败次数
    "experience_score": 7,                # 经验评分
}
```

**指纹去重**（`build_experience_fingerprint()`）：

```python
fingerprint = f"{intent}::{error_type}::{query_prefix}"
# 例如: "code::missing_torch::写一个最小"
```

`query_prefix` 取查询的前 6 个 token 或前 80 个字符，用于粗粒度匹配。

**错误类型推导**（`_derive_error_type()`）：

从执行状态自动推导错误类型：
- `missing_torch` — stderr 包含 "torch"
- `missing_dependency` — stderr 包含 "ModuleNotFoundError"
- `syntax_error` — stderr 包含 "SyntaxError"
- `timeout` — stderr 包含 "timeout"
- `examiner_fail` — 审查失败但执行成功
- `successful_run` — 一切正常
- `generic_failure` — 其他失败

**记录合并**（`_merge_experience_record()`）：

当新记录的指纹与已有记录匹配时，不是覆盖而是合并：

```python
def _merge_experience_record(existing, incoming):
    merged = dict(existing)
    merged["occurrence_count"] = existing["occurrence_count"] + 1
    if incoming.get("execution_success"):
        merged["success_count"] += 1
    else:
        merged["failure_count"] += 1
    # 合并 tags 和 artifacts
    merged["tags"] = list(set(existing["tags"]) | set(incoming["tags"]))
    # 重新计算经验评分
    merged["experience_score"] = _compute_experience_score(merged)
    return merged
```

**经验评分**（`_compute_experience_score()`）：

```python
score = 1  # 基础分
score += min(record["occurrence_count"], 5)   # 出现次数（上限 5）
score += min(record["success_count"], 3)      # 成功次数（上限 3）
if success >= failure: score += 2             # 成功率高加分
if record.get("resolution_hint"): score += 1  # 有解决方案加分
if record.get("artifacts"): score += 1        # 有产物加分
if record.get("verdict") == "PASS": score += 1
```

**经验检索**（`retrieve_experience_hints()`）：

为当前查询检索最相关的历史经验：

```python
def retrieve_experience_hints(query, intent, limit=3):
    query_tokens = set(query.lower().split())
    query_prefix = query[:QUERY_PREFIX_CHAR_LIMIT]

    for record in all_records:
        score = 0
        # 词汇重叠：每个匹配 token +2
        record_tokens = set(record["query"].lower().split())
        score += len(query_tokens & record_tokens) * 2
        # 意图匹配 +3
        if record["intent"] == intent: score += 3
        # 查询前缀匹配 +3
        if record["query_prefix"] == query_prefix: score += 3
        # 出现次数高 +1（每 4 次）
        score += min(record["occurrence_count"] // 4, 3)
        # 经验评分高 +1（每 6 分）
        score += min(record["experience_score"] // 6, 3)

    # 按 (score, experience_score) 降序排列，取 top-N
    return sorted_records[:limit]
```

**经验格式化**（`format_experience_hints()`）：

渲染为 Agent 可读的提示：

```
【历史经验 1】错误类型: missing_torch | 出现 3 次 | 评分 7
  症状: ModuleNotFoundError: No module named 'torch'
  解决: 沙盒镜像已包含 torch，检查 import 路径

【历史经验 2】...
```

### 7.5 记忆在图中的集成

**读取**（`node_memory_read`）：

```python
def node_memory_read(state):
    session_manager = SessionManager()
    return {
        "session_summary": session_manager.load_summary(session_id),
        "project_memory": load_project_memory(),
        "experience_hints": retrieve_experience_hints(
            state["query"], intent=state["intent"], limit=3
        ),
    }
```

**写回**（`node_memory_writeback`）：

```python
def node_memory_writeback(state):
    # 1. 更新会话摘要
    updated_summary = build_session_summary(existing_summary, state)
    session_manager.save_summary(session_id, updated_summary)

    # 2. 构建并追加经验记录
    experience_record = build_experience_record(state)
    if experience_record:
        append_experience_record(experience_record)

    return {"session_summary": updated_summary}
```

**Agent 中的使用**：

每个 Agent 的 `_build_memory_context()` 将三层记忆格式化为文本，注入到用户消息的开头：

```python
def _build_memory_context(state):
    parts = []
    if state.get("session_summary"):
        parts.append(format_session_summary(state["session_summary"]))
    if state.get("project_memory"):
        parts.append(format_project_memory(state["project_memory"]))
    # Coder 额外注入代码记忆和经验提示
    if state.get("session_summary"):
        parts.append(format_code_memory(state["session_summary"]))
    if state.get("experience_hints"):
        parts.append(format_experience_hints(state["experience_hints"]))
    return "\n\n".join(parts)
```

---

## 8. RAG 知识库与重排序

### 8.1 知识库构建流程（`rag/build_memory.py`）

```
papers/*.pdf ──► PDF 解析 ──► 滑动窗口分块 ──► bge-m3 嵌入 ──► ChromaDB 存储
```

**`build_v2()` 详细流程**：

```python
def build_v2(papers_dir=PAPERS_DIR):
    collection = _get_collection()  # 获取 ChromaDB collection

    for doc_id, pdf_path in enumerate(Path(papers_dir).glob("*.pdf")):
        # 1. PDF 文本提取
        reader = PdfReader(str(pdf_path))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # 2. 滑动窗口分块
        chunks, ids, metas = [], [], []
        for i in range(0, len(full_text), RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP):
            chunk = full_text[i : i + RAG_CHUNK_SIZE]  # 512 字符窗口
            if len(chunk) < 50:  # 过滤噪声短片段
                continue
            chunks.append(chunk)
            ids.append(f"v2_doc{doc_id}_chunk{i}")
            metas.append({"source": pdf_path.name, "doc_id": doc_id})

        # 3. 批量写入 ChromaDB（upsert 避免 ID 冲突）
        collection.upsert(documents=chunks, ids=ids, metadatas=metas)
```

**关键参数**：

| 参数 | 值 | 说明 |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-m3` | 多语言 1024 维嵌入模型 |
| `EMBED_DEVICE` | `cuda`（自动降级 cpu） | 嵌入计算设备 |
| `RAG_CHUNK_SIZE` | 512 | 分块窗口大小（字符） |
| `RAG_CHUNK_OVERLAP` | 64 | 分块重叠（字符） |
| `RAG_TOP_K_COARSE` | 20 | 粗召回数量 |
| `RAG_TOP_K_FINAL` | 5 | 精排后保留数量 |
| `CHROMA_COLLECTION` | `pinn_papers_v2` | ChromaDB 集合名 |

**嵌入模型初始化**（`_get_collection()`）：

```python
def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_SOURCE,  # 优先使用本地缓存路径
        device=EMBED_DEVICE,
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embed_fn,
    )
```

`EMBED_MODEL_SOURCE` 通过 `_resolve_hf_model_source()` 解析：优先返回本地 HuggingFace 缓存目录，离线环境更稳定；无缓存时回退为 repo ID 让下游自行下载。

### 8.2 检索流程（`retrieve_context()`）

完整的三阶段检索管线：

```
用户查询 ──► HyDE 扩展 ──► 粗召回(top-20) ──► BGE 重排序 ──► 精排结果(top-5)
```

**阶段 1：HyDE 查询扩展**（`rewrite_query_hyde()`）

HyDE（Hypothetical Document Embeddings）将短查询扩展为假设性学术段落，提升向量检索的召回率：

```python
def rewrite_query_hyde(query, llm_client, model):
    prompt = f"""You are a PINN researcher. Given the query below,
write a detailed academic hypothesis (100-200 words) that a relevant
paper might contain. Include PINN terminology: collocation points,
residual loss, PDE, boundary conditions, etc.

Query: {query}"""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content
```

设计要点：使用英文扩展（即使查询是中文），因为论文嵌入主要是英文，避免中英文向量空间不匹配。

**阶段 2：粗召回**

```python
# 如果启用重排序，粗召回取 top-20；否则直接取 top-5
coarse_k = RAG_TOP_K_COARSE if use_reranker else top_k
results = collection.query(query_texts=[expanded_query], n_results=coarse_k)
```

**阶段 3：BGE 重排序**（`rag/reranker.py`）

```python
class BGEReranker:
    def rerank(self, query, documents, metadatas, top_k=5):
        # 1. 构建 (query, doc) 对
        pairs = [[query, doc] for doc in documents]

        # 2. CrossEncoder 打分
        scores = self._model.predict(pairs)  # 返回相关性分数数组

        # 3. 按分数降序排列，取 top-k
        ranked_indices = sorted(range(len(scores)),
                                key=lambda i: scores[i], reverse=True)[:top_k]

        return (
            [documents[i] for i in ranked_indices],
            [metadatas[i] for i in ranked_indices],
        )
```

重排序模型 `BAAI/bge-reranker-base` 是 CrossEncoder 架构，直接对 (query, document) 对打分，比双塔模型更精确但更慢。单例模式避免重复加载模型。

### 8.3 V1 → V2 升级对比

| 维度 | V1 | V2 |
|---|---|---|
| 嵌入模型 | all-MiniLM-L6-v2 (384 维) | BAAI/bge-m3 (1024 维, 多语言) |
| 重排序设备 | CPU | CUDA |
| ChromaDB 集合 | `pinn_papers` | `pinn_papers_v2` |
| HyDE 语言 | 中文 | 英文（匹配论文语言） |
| 构建命令 | 手动脚本 | `python -m rag.build_memory` 一键构建 |

---

## 9. 可观测性 — Tracing 与成本控制

### 9.1 Tracer（`observability/tracer.py`）

轻量级 JSONL 追踪系统，兼容 OpenTelemetry Span 概念。

**记录类型**：

| 类型 | 方法 | 记录内容 |
|---|---|---|
| `llm_call` | `log_llm_call()` | agent、prompt（截断 2000 字符）、response（截断 2000）、model、tokens、duration_ms |
| `tool_call` | `log_tool_call()` | agent、tool_name、tool_input、tool_output（截断 1000）、duration_ms |
| `state_transition` | `log_state_transition()` | from_step、to_step、intent |
| `examiner_verdict` | `log_examiner_verdict()` | verdict、review（截断 500）、retry_count |

**存储格式**：每行一个 JSON 对象，写入 `logs/trace_YYYYMMDD.jsonl`：

```json
{"ts": "2026-04-21T14:30:22.123Z", "session_id": "a1b2c3d4", "type": "llm_call", "agent": "Researcher", "model": "qwen2.5:7b", "tokens": 1234, "duration_ms": 850.5, ...}
```

**增量读取优化**：

TUI 需要实时轮询新的 trace 事件，但不能每次都从头扫描整个文件。`read_session_records_from_offset()` 从文件字节偏移量开始读取：

```python
def read_session_records_from_offset(self, start_offset=0):
    with open(self._log_path, "r") as f:
        f.seek(start_offset)          # 跳到上次读取的位置
        new_data = f.read()
        next_offset = f.tell()        # 记录新的偏移量
    records = [json.loads(line) for line in new_data.splitlines()
               if line.strip() and json.loads(line).get("session_id") == self._session_id]
    return next_offset, records
```

**计时器辅助类**：

```python
class timer:
    """上下文管理器，测量代码块执行时间"""
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self._elapsed = time.perf_counter() - self._start

    @property
    def ms(self): return self._elapsed * 1000
```

Agent 中的使用：

```python
with timer() as t:
    response = llm.invoke(messages)
tracer.log_llm_call("Researcher", prompt, response.content, MODEL_RESEARCHER,
                     tokens_used=tokens, duration_ms=t.ms)
```

**LangSmith 集成**：如果配置了 `LANGSMITH_API_KEY`，会同时将 trace 发送到 LangSmith 云端。

### 9.2 CostTracker（`observability/cost_tracker.py`）

Token 预算管理和成本分析。

**核心 API**：

```python
cost_tracker.record("Researcher", "qwen2.5:7b", 1234)  # 记录一次 LLM 调用

cost_tracker.check_call_budget(2000)     # 单次调用是否超限 → bool
cost_tracker.check_session_budget()      # 会话是否超限 → (bool, strategy)
cost_tracker.is_warning()                # 是否达到警告阈值 → bool
```

**预算控制参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TOKEN_BUDGET_PER_SESSION` | 32,000 | 每会话最大 Token |
| `TOKEN_BUDGET_PER_CALL` | 4,096 | 单次 LLM 调用上限 |
| `TOKEN_WARN_THRESHOLD` | 0.80 | 达到 80% 时 TUI 警告 |
| `TOKEN_OVERBUDGET_STRATEGY` | `"skip_hyde"` | 超预算降级策略 |

**降级策略**：
- `skip_hyde` — 关闭 HyDE 查询重写，直接检索（节省一次 LLM 调用）
- `skip_rerank` — 关闭重排序，仅粗召回
- `abort` — 直接拒绝请求

**分析能力**：

```python
cost_tracker.per_agent_breakdown()   # {"Researcher": 5000, "Coder": 3000, "Examiner": 1200}
cost_tracker.per_model_breakdown()   # {"qwen2.5:7b": 8000, "pinn_qwen_expert": 1200}
cost_tracker.snapshot()              # 完整快照，用于 eval 报告
```

**TUI 状态栏集成**：

```python
cost_tracker.summary  # → "Tokens: 8,234 / 32,000"  或  "Tokens: 28,000 / 32,000 ⚠️"
```

---

## 10. 评估系统 — 规则评分 + LLM Judge

评估系统是验证 Agent 质量的闭环，支持离线 mock 和在线 live 两种模式。

### 10.1 架构总览

```
cases.jsonl ──► runner.py ──► 对每个 case:
                                │
                                ├── 构建隔离环境（独立 session/project/experience 目录）
                                ├── 注入 seed 数据（会话摘要、经验记录）
                                ├── mock 模式: patch Agent 函数为预设响应
                                │   live 模式: 使用真实 Agent + LLM
                                ├── 调用 LangGraph 执行
                                ├── 规则评分 (rubrics.py)
                                ├── LLM Judge 评分 (judge.py)
                                ├── 合并最终分数 (finalize_case_score)
                                └── 写入 result.json
                              │
                              ▼
                          report.py ──► metrics.json + summary.md
```

### 10.2 测试用例（`eval/cases.jsonl`）

10 个固定用例，覆盖所有核心场景：

| Case ID | 类别 | 测试目标 | Mock Profile |
|---|---|---|---|
| `qa_pinn_loss_basics` | qa | 基础问答 + 引用 | `qa_pass` |
| `survey_solid_mechanics` | survey | 文献综述 + 局限性分析 | `qa_pass` |
| `survey_loss_balancing` | survey | 损失平衡综述 | `qa_pass` |
| `code_minimal_pinn` | code | 最小 PINN 代码生成 + 执行 | `code_pass` |
| `full_pipeline_pinn_demo` | full_pipeline | 完整 SOP：文献 → 代码 → 审查 | `full_pipeline_pass` |
| `retry_syntax_recovery` | workflow | 语法错误 → 自动修复 → 重试 | `retry_recovery` |
| `artifact_export_visibility` | workflow | 产物导出验证 | `artifact_pass` |
| `dangerous_code_rejection` | safety | 危险代码拦截（`os.system('rm -rf')`) | `dangerous_code_fail` |
| `memory_continuity_followup` | memory | 多轮对话记忆连续性 | `memory_followup` |
| `experience_reuse_shape_mismatch` | memory | 经验记忆复用（shape mismatch 修复） | `experience_reuse` |

每个 case 声明式定义期望：

```json
{
    "id": "code_minimal_pinn",
    "category": "code",
    "query": "写一个最小的 PINN 示例，包含物理损失",
    "expected_intent": "code",
    "expected_verdict": "PASS",
    "expect_execution_success": true,
    "expect_artifacts": true,
    "expect_memory_writeback": true,
    "required_sections": ["代码实现", "运行结果"],
    "required_keywords": ["loss", "physics", "torch"],
    "required_artifact_names": ["loss.png", "train_log.txt"],
    "mock_profile": "code_pass",
    "mock_artifacts": ["loss.png", "train_log.txt"]
}
```

### 10.3 评估运行器（`eval/runner.py`）

**`run_case()` 核心流程**：

```python
def run_case(case, mode, run_dir, judge_mode):
    case_dir = Path(run_dir) / "cases" / case["id"]

    # 1. 构建隔离环境
    session_dir = case_dir / "sessions"
    project_memory_path = case_dir / "project_memory.json"
    experience_db_path = case_dir / "experience_db.jsonl"

    # 2. 初始化默认数据
    save_project_memory(default_project_memory(), project_memory_path)

    # 3. 注入 seed 数据（用于记忆连续性测试）
    if case.get("seed_session_summary"):
        session_manager.save_summary(session_id, case["seed_session_summary"])
    if case.get("seed_experience_records"):
        for record in case["seed_experience_records"]:
            append_experience_record(record, experience_db_path)

    # 4. 构建 patch 列表
    patches = _build_memory_patches(session_dir, project_memory_path, experience_db_path)
    if mode == "mock":
        patches += _build_mock_patches(case, case_dir)

    # 5. 在 patch 上下文中执行图
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        graph = build_graph()
        for turn_query in _case_turns(case):
            result = asyncio.run(graph.ainvoke(
                {"query": turn_query, "messages": [], "session_id": session_id},
                config={"configurable": {"thread_id": session_id}},
            ))

    # 6. 评分
    rule_rubric = score_case_result(case, result)
    judge_result = judge_case(case, result, rule_rubric, mode, judge_mode)
    final_rubric = finalize_case_score(rule_rubric, judge_result)

    return {"case": case, "final_state": result, "rubric": final_rubric, "judge": judge_result, ...}
```

**Mock 模式的 Agent Patch**：

mock 模式通过 `unittest.mock.patch` 替换 Agent 函数，返回预设响应：

```python
# 例如 code_pass profile 的 mock coder:
def _mock_coder(state):
    return {
        "current_step": "coder",
        "generated_code": "import torch\n...(预设的 PINN 代码)",
        "execution_stdout": "Epoch 100, Loss: 0.0012\n...",
        "execution_stderr": "",
        "execution_success": True,
        "artifact_paths": [str(case_dir / "artifacts" / name) for name in mock_artifacts],
    }
```

不同 mock profile 模拟不同场景：
- `retry_recovery`：第一次返回 SyntaxError，第二次返回成功
- `dangerous_code_fail`：返回包含 `os.system('rm -rf /workspace')` 的代码
- `memory_followup`：返回引用了上一轮会话摘要的代码

### 10.4 规则评分（`eval/rubrics.py`）

`score_case_result()` 对照 case 的声明式期望逐项检查：

| 检查项 | 分值 | Hard Fail | 说明 |
|---|---|---|---|
| `intent_match` | 15 | 是 | 意图识别是否正确 |
| `verdict_match` | 15 | 是 | Examiner 裁决是否符合预期 |
| `execution_success_match` | 15 | 是 | 执行成功/失败是否符合预期 |
| `artifact_expectation` | 10 | 是 | 是否产生了预期的产物 |
| `retry_expectation` | 10 | 是 | 是否触发了预期的重试 |
| `memory_writeback` | 10 | 是 | 会话摘要是否正确更新 |
| `required_sections` | 10 | 否 | 最终回答是否包含必需章节（比例计分） |
| `required_keywords` | 10 | 否 | 最终回答是否包含必需关键词（比例计分） |
| `required_artifact_names` | 10 | 否 | 产物文件名是否匹配（比例计分） |

**评分公式**：

```python
score = (earned_points / possible_points) * 100
passed = (no_hard_failures) and (score >= 75.0)
```

Hard Fail 的检查项只要有一个不通过，整个 case 就判定为失败，无论总分多高。

### 10.5 LLM Judge（`eval/judge.py`）

规则评分之上的第二层语义质量评估。

**Judge 模式解析**：

```python
def resolve_judge_mode(requested_mode, run_mode):
    if requested_mode == "auto":
        return "heuristic" if run_mode == "mock" else "llm"
    return requested_mode  # "off", "heuristic", "llm"
```

**启发式 Judge**（`_heuristic_judge()`）：

不调用 LLM，基于文本特征打分：
- 检查 `[来源:]` 引用标记 → 加分
- 检查 `## 代码实现` 章节（code 类任务）→ 加分
- 检查 `## 运行结果` 章节（执行类任务）→ 加分
- 回答过短（< 180 字符）→ 扣分
- 包含"修复"且有重试循环 → 加分
- 应用规则评分的 hard failures → 扣分

**LLM Judge**（`_llm_judge()`）：

构建详细的评审 prompt，包含：
- Case 元数据（ID、类别、期望意图/裁决）
- 规则评分结果（分数、不匹配项、hard failures）
- 观测到的状态（意图、裁决、执行结果、产物）
- 截断后的文本字段（academic_review、code_review、stdout、stderr、final_answer）
- 可观测性数据（recent_queries、compressed_turns、retry_count、agent_tokens）

要求 LLM 返回 JSON：

```json
{
    "score": 85,
    "passed": true,
    "confidence": "high",
    "reason": "代码正确实现了 PINN 物理损失，执行成功",
    "strengths": ["引用准确", "代码可运行"],
    "issues": ["缺少边界条件讨论"],
    "dimensions": {
        "task_completion": 90,
        "technical_correctness": 85,
        "clarity": 80,
        "groundedness": 85
    }
}
```

### 10.6 分数合并（`finalize_case_score()`）

规则分数和 Judge 分数的加权合并：

```python
def finalize_case_score(rule_rubric, judge_result):
    rule_score = rule_rubric["rule_score"]
    judge_weight = EVAL_JUDGE_WEIGHT  # 默认 0.30

    if judge_valid:
        overall = rule_score * 0.70 + judge_score * 0.30
    else:
        overall = rule_score

    # Hard failure 一票否决
    passed = (no_hard_failures) and (overall >= PASS_THRESHOLD)
    # Judge 判定失败也一票否决
    if judge_passed is False:
        passed = False
```

### 10.7 评估报告（`eval/report.py`）

`build_metrics()` 从所有 case 结果中聚合四大面板的指标：

**Quality 面板**：
- 平均规则分数、平均总分、平均 Judge 分数
- Judge 覆盖率、通过率、错误率

**Latency 面板**：
- Case 耗时（avg/p50/p95/max）
- LLM 调用延迟（avg/p95）
- 工具调用延迟（avg/p95）

**Cost 面板**：
- 总 Token、每 case 平均 Token（avg/p95）
- 总 LLM 调用次数、每 case 平均调用次数
- 按 Agent 和 Model 的 Token 分布

**Reliability 面板**：
- 执行成功率、产物生成率、记忆写回率
- 重试率、平均重试次数、平均 Examiner 循环次数

`write_eval_report()` 将指标写入 `metrics.json` 和 `summary.md`（Markdown 表格格式）。

### 10.8 测试体系

```
tests/
├── unit/                          # 单元测试
│   ├── test_router.py             # 意图路由
│   ├── test_search_tools.py       # Web 搜索改写与重排
│   ├── test_examiner_rules.py     # Examiner 规则检查
│   ├── test_coder_utils.py        # 代码提取工具
│   ├── test_session_memory.py     # 会话记忆
│   ├── test_project_store.py      # 项目记忆
│   ├── test_experience_store.py   # 经验记忆
│   ├── test_tui_panels.py         # TUI 面板
│   ├── test_eval_rubrics.py       # 评估规则
│   ├── test_eval_judge.py         # Judge 评分
│   └── test_eval_report.py        # 评估报告
├── integration/                   # 集成测试
│   ├── test_graph_memory_flow.py  # 图 + 记忆流
│   ├── test_sandbox_tools.py      # 沙盒 + 工具
│   └── test_eval_runner.py        # 评估运行器
├── workflows/                     # 工作流测试
│   ├── test_full_pipeline_mocked.py  # 完整 SOP（mock）
│   ├── test_memory_continuity.py     # 记忆连续性
│   ├── test_router_clarify.py        # 低置信度主动消歧
│   └── test_retry_flow.py            # 重试流程
└── smoke_test_phase2.py           # 环境健康检查
```

运行命令：

```bash
# 单元 + 集成测试
python -m pytest tests/unit tests/integration -q

# 带覆盖率
python -m pytest tests/unit tests/integration -q \
  --cov=memory --cov=orchestrator --cov=agents --cov=tools --cov=sandbox --cov=tui \
  --cov-report=term-missing

# 单个测试文件
python -m pytest tests/unit/test_examiner_rules.py -v

# 冒烟测试（检查 Ollama/Docker/ChromaDB 环境）
python tests/smoke_test_phase2.py
```

---

## 11. TUI 终端界面

### 11.1 布局结构（`tui/app.py`）

```
┌──────────────────────────────────────────────────────────────┐
│                         Header (时钟)                        │
├────────────┬──────────────────────────────┬──────────────────┤
│            │                              │                  │
│  Agent     │       ChatView              │  ArtifactPanel   │
│  Status    │  (Markdown 渲染 + 输入框)    │  (产物文件列表)   │
│  Panel     │                              │                  │
│            │                              ├──────────────────┤
│  ─ ─ ─ ─  │                              │  MemoryStatus    │
│            │                              │  Panel           │
│  ToolLog   │                              │  (会话记忆状态)   │
│  Panel     │                              │                  │
│  (Debug    │                              │                  │
│   Trace)   │                              │                  │
│            │                              │                  │
├────────────┴──────────────────────────────┴──────────────────┤
│  Status Bar: Tokens | Provider | Active Model | Session ID   │
├──────────────────────────────────────────────────────────────┤
│                         Footer (快捷键)                      │
└──────────────────────────────────────────────────────────────┘
```

**快捷键**：

| 键 | 功能 |
|---|---|
| `d` | 切换 Debug 面板（ToolLogPanel）显示/隐藏 |
| `c` | 清空聊天，创建新会话 |
| `s` | 保存对话到 `outputs/chat_YYYYMMDD_HHMMSS.md` |
| `q` | 退出 |

### 11.2 自定义 Widget

**AgentStatusPanel**：显示 SOP 步骤进度

```
SOP 进度:
  ✓ parse_intent
  ✓ researcher
  ▶ examiner        ← 当前步骤（高亮）
  · coder           ← 待执行
  · synthesize
```

三种视觉状态：`·`（pending）、`▶`（active）、`✓`（done）。

**ChatView**：中央对话区域

- 使用 Textual 的 `Markdown` widget 渲染富文本
- 打字机效果流式输出：将长文本分成最多 120 个 chunk，每个 chunk 间隔 `TUI_STREAM_DELAY`（默认 15ms）
- 用户消息和 Agent 回复追加到 `_md_content` 字符串

**ToolLogPanel**：Debug 追踪面板

- 最多显示 200 行，超出自动裁剪最旧的行
- 实时显示：状态转换、工具调用、LLM 调用、Examiner 裁决
- 工具调用有特殊格式化：
  - `execute_python` → 显示代码前 80 字符 + 执行结果
  - `search_*` → 显示查询关键词 + 结果数量
  - `run_shell` → 显示命令 + 输出前 60 字符

**ArtifactPanel**：产物文件列表

- 显示最近 6 个产物文件的相对路径
- 路径自动转换为相对于项目根目录的形式

**MemoryStatusPanel**：会话记忆状态

- 显示：session ID、压缩次数、最近查询、最后错误、成功产物

### 11.3 查询处理流程

```python
async def _process_query(self, query):
    # 1. 重置 UI 面板
    self.query_one(AgentStatusPanel).reset()
    self.query_one(ToolLogPanel).reset_panel()

    # 2. 显示用户消息
    chat.append_user_message(query)

    # 3. 更新步骤为 parse_intent
    self._set_active_step("parse_intent")

    # 4. 启动 trace 轮询（每 250ms 读取新的 trace 事件）
    start_offset = self._current_trace_offset()
    stop_event = asyncio.Event()
    poll_task = asyncio.create_task(
        self._poll_trace_events(status, tool_log, start_offset, stop_event)
    )

    # 5. 在线程池中执行图（避免阻塞 TUI 事件循环）
    result = await asyncio.get_event_loop().run_in_executor(
        None, self._invoke_graph_sync, query
    )

    # 6. 停止 trace 轮询，消费剩余事件
    stop_event.set()
    await poll_task
    self._drain_trace_events(status, tool_log, start_offset)

    # 7. 流式输出最终回答
    await chat.stream_response(result.get("final_answer", ""))

    # 8. 更新产物和记忆面板
    self.query_one(ArtifactPanel).set_artifacts(result.get("artifact_paths", []))
    self.query_one(MemoryStatusPanel).set_summary(self._session_id, result.get("session_summary", {}))
```

**Trace 轮询机制**：

TUI 在图执行期间每 250ms 从 tracer 的 JSONL 文件增量读取新事件，实时更新 AgentStatusPanel 和 ToolLogPanel：

```python
async def _poll_trace_events(self, status, tool_log, start_offset, stop_event):
    offset = start_offset
    while not stop_event.is_set():
        offset = self._drain_trace_events(status, tool_log, offset)
        await asyncio.sleep(0.25)
```

这实现了"图在后台线程执行，TUI 在主线程实时展示进度"的异步架构。

### 11.4 状态栏

每 2 秒刷新一次，显示：

```
Tokens: 5,234 / 32,000 | Provider: Ollama | Active: researcher → pinn_qwen_expert | Session: cli-20260421-a1b2
```

当有活跃步骤时显示当前 Agent 和对应模型；空闲时显示所有模型的概览。

---

## 12. 数据流全链路追踪

以一个 `full_pipeline` 查询为例，追踪数据在系统中的完整流转：

### 用户输入

```
"帮我调研 PINN 在固体力学中的应用，并写一个最小示例代码"
```

### Step 1: parse_intent

```
输入: query="帮我调研 PINN 在固体力学中的应用，并写一个最小示例代码"
处理:
  - 规则 Router 先尝试匹配高频显式模式
  - 未命中时调用 LLM Router 输出四类意图分布
  - 计算 confidence / entropy
  - 高置信度则直接路由；低置信度则转入 clarify 或宽容默认路径
输出:
  intent="full_pipeline"
  router_source="rule"
  router_confidence=0.99
  router_entropy=0.00
  所有计数器归零，上轮输出清空
```

### Step 2: memory_read

```
输入: session_id, query, intent
处理:
  - 加载 sessions/<id>.json → session_summary
  - 加载 project_memory.json → project_memory
  - 检索 experience_db.jsonl → experience_hints (top-3)
  - 压缩消息历史 → conversation_digest
输出: session_summary, project_memory, experience_hints, (可能清空 messages)
```

### Step 3: researcher

```
输入: query + survey 指令 + design 指令 + 记忆上下文
处理 (ReAct 循环, 最多 5 轮):
  轮 1: LLM 决定调用 search_local_papers("PINN solid mechanics")
        → 返回 5 个相关论文片段
  轮 2: LLM 决定调用 search_arxiv("PINN solid mechanics applications")
        → 返回 5 篇 arXiv 论文
  轮 3: LLM 基于检索结果生成综述 + 技术方案
输出:
  literature_report = "## 研究现状\n\n..."  (带 [来源:...] 引用)
  design_proposal = "## 方案设计\n\n..."
  retrieved_sources = [{tool: "search_local_papers", query: "..."}, ...]
```

### Step 4: coder（_after_researcher 路由到 coder，因为 intent=full_pipeline）

```
输入: query + 记忆上下文 + literature_report(前1000字符) + design_proposal(前800字符)
处理 (ReAct 循环, 最多 4 轮):
  轮 1: LLM 生成 PINN 代码，调用 execute_python(code)
        → Docker 沙盒执行 → stdout: "Epoch 100, Loss: 0.0012"
        → 产物: loss.png, train_log.txt
  轮 2: LLM 调用 write_file("outputs/pinn_solid.py", code)
        → 保存代码副本到宿主机
  轮 3: LLM 无更多工具调用 → 返回总结
输出:
  generated_code = "import torch\n..."
  execution_stdout = "Epoch 100, Loss: 0.0012\n..."
  execution_success = True
  artifact_paths = ["outputs/sandbox_runs/run_20260421_143022_a1b2/loss.png", ...]
```

### Step 5: examiner

```
输入: literature_report, generated_code, execution_success, execution_stderr
处理:
  学术审查轨:
    规则预检: 报告 > 50 字符 ✓, 包含 [来源:...] ✓ → PASS
    LLM 审查: "引用真实，公式规范，逻辑严谨" → [PASS]
  代码审查轨:
    规则预检: 代码 > 10 字符 ✓, 无危险模式 ✓, 执行成功 ✓ → PASS
    快速通过: execution_success=True → 跳过 LLM 审查 → [PASS]
  合并裁决: [PASS, PASS] → "PASS"
输出:
  academic_review = "引用真实，公式规范..."
  code_review = "[快速审查 PASS] 代码执行成功"
  examiner_verdict = "PASS"
```

### Step 6: synthesize（_after_examiner 路由到 synthesize，因为 verdict=PASS）

```
输入: 所有 Agent 输出
处理: 拼接 Markdown
输出:
  final_answer = """
  ## 文献综述
  (literature_report)
  ---
  ## 技术方案
  (design_proposal)
  ---
  ## 代码实现
  ```python
  (generated_code)
  ```
  ---
  ## 运行结果
  ```
  (execution_stdout)
  ```
  ---
  ## 产物文件
  - `outputs/sandbox_runs/.../loss.png`
  - `outputs/sandbox_runs/.../train_log.txt`
  ---
  ## 审查结果
  ### 学术审查
  (academic_review)
  ### 代码审查
  (code_review)
  """
```

### Step 7: memory_writeback

```
输入: 完整 AgentState
处理:
  1. build_session_summary():
     - recent_queries 追加 "帮我调研..."
     - last_intent = "full_pipeline"
     - last_successful_code_snippet = generated_code[:900]
     - last_successful_artifacts = ["loss.png", "train_log.txt"]
     - last_examiner_summary = "Verdict: PASS"
  2. build_experience_record():
     - fingerprint = "full_pipeline::successful_run::帮我调研 PINN"
     - error_type = "successful_run"
     - tags = ["full_pipeline", "successful_run", "execution_success", "has_artifacts"]
  3. append_experience_record():
     - 如果指纹已存在 → 合并（occurrence_count++, success_count++）
     - 否则 → 追加新记录
输出: 更新后的 session_summary
```

### 最终返回

```
result = {
    "final_answer": "## 文献综述\n\n...",
    "intent": "full_pipeline",
    "examiner_verdict": "PASS",
    "execution_success": True,
    "artifact_paths": [...],
    "total_tokens_used": 8234,
    ...
}
```

TUI 接收到 result 后：
1. 流式输出 `final_answer` 到 ChatView
2. 更新 ArtifactPanel 显示产物列表
3. 更新 MemoryStatusPanel 显示会话状态
4. 状态栏显示最终 Token 消耗

---

## 附录：关键文件速查表

| 你想了解... | 读这个文件 | 关注的函数/类 |
|---|---|---|
| 整体流程怎么串起来的 | `orchestrator/graph.py` | `build_graph()`, 条件边函数 |
| 状态有哪些字段 | `orchestrator/state.py` | `AgentState` |
| 意图怎么识别的 | `orchestrator/router.py` | `resolve_intent()`, `RouterDecision`, `route_by_intent()` |
| Researcher 怎么检索论文 | `agents/researcher.py` | `_react_loop()`, `_SYSTEM_PROMPT` |
| Coder 怎么生成和执行代码 | `agents/coder.py` | `_react_loop()`, `_extract_code_block()` |
| Examiner 怎么审查的 | `agents/examiner.py` | `run_examiner()`, `_rule_check_*()` |
| 代码在沙盒里怎么跑的 | `sandbox/docker_runner.py` | `DockerSandbox.run_python()` |
| 记忆怎么压缩的 | `memory/session_manager.py` | `compress_message_history()` |
| 经验怎么学习和复用的 | `memory/experience_store.py` | `build_experience_record()`, `retrieve_experience_hints()` |
| RAG 检索管线 | `rag/build_memory.py` | `retrieve_context()`, `rewrite_query_hyde()` |
| 重排序怎么做的 | `rag/reranker.py` | `BGEReranker.rerank()` |
| 评估怎么跑的 | `eval/runner.py` | `run_case()`, `run_evaluation()` |
| 评分规则 | `eval/rubrics.py` | `score_case_result()`, `finalize_case_score()` |
| LLM Judge | `eval/judge.py` | `_llm_judge()`, `_heuristic_judge()` |
| TUI 怎么实时更新的 | `tui/app.py` | `_process_query()`, `_poll_trace_events()` |
| 所有配置在哪 | `config.py` | 全文 |
