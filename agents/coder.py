"""
agents/coder.py — 编程执行 Agent

职责: 将科研思路转化为可运行代码，在 Docker 沙盒中执行并自动 debug。

流程: 生成代码 → execute_python → 分析结果 → 修复错误（最多 MAX_ITER 轮）

工具: execute_python, read_file, write_file, run_shell

输出字段（写入 AgentState）:
    generated_code      最终代码字符串
    execution_stdout    沙盒 stdout
    execution_stderr    沙盒 stderr
    execution_success   执行是否成功
"""

from __future__ import annotations

import json
import re

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
    MODEL_CODER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)
from memory import (
    format_code_memory,
    format_experience_hints,
    format_project_memory,
    format_session_summary,
)
from observability.cost_tracker import cost_tracker
from observability.tracer import timer, tracer
from orchestrator.state import AgentState
from tools.code_tools import execute_python, read_file, run_shell, write_file

# ── 常量 ────────────────────────────────────────────────────
MAX_ITER   = 4          # 1次生成 + 最多3次 debug
AGENT_NAME = "Coder"

_TOOLS    = [execute_python, read_file, write_file, run_shell]
_TOOL_MAP = {t.name: t for t in _TOOLS}

# ── 系统提示词 ───────────────────────────────────────────────
_SYSTEM_PROMPT = """\
你是一个专业的 PINN 代码工程师 Agent。

【核心职责】
1. 根据科研方案编写高质量、可直接运行的 Python 代码。
2. 代码规范：清晰注释、模块化结构、遵循 PEP8。
3. 执行验证：写完代码后必须用 execute_python 运行，确认无错误。
4. 修复错误：若执行失败，分析 stderr，修改代码后重新执行。
5. 保存结果：执行成功后用 write_file 保存到 outputs/ 目录。
6. 产物落盘：图像、日志等运行产物请保存到当前工作目录下的相对路径
   （例如 `artifacts/loss.png`、`train.log`），不要写入 `/tmp` 或宿主机绝对路径。
7. 沙盒边界：`execute_python` 运行在独立的 `/workspace` 中；不要先 `write_file`
   到宿主机项目目录，再在沙盒里用 `subprocess` / `python xxx.py` 去读取那个文件。
   若要执行代码，请把“完整、可独立运行”的源码直接传给 `execute_python`。
8. 文件职责分离：`write_file` 只用于把最终代码副本保存到宿主机 `outputs/`；
   运行期日志、图片等产物则应由 `execute_python` 内的代码直接写入相对路径。

【技术栈偏好】
- PINN 框架：PyTorch（首选）
- 可视化：matplotlib
- 数值计算：numpy, scipy

【代码风格要求】
- 使用 if __name__ == '__main__': 保护入口
- 关键步骤打印进度信息
- 损失曲线最后用 matplotlib 保存为图片

【工作流程】编写 → execute_python 验证 → 修复（如需）→ write_file 保存。\
"""

_RETRY_TEMPLATE = """\
上一次代码执行失败，请分析错误并修复。

【失败代码】
```python
{code}
```

【错误信息（stderr）】
```
{stderr}
```

【标准输出（stdout，可能为空）】
```
{stdout}
```

这是第 {retry_num} 次修复尝试（最多 {max_retries} 次）。
请直接给出修复后的完整代码并重新执行，不要只修改片段。\
"""


# ── LLM 工厂 ────────────────────────────────────────────────
def _build_llm(with_tools: bool = True) -> ChatOpenAI:
    llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        model=MODEL_CODER,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return llm.bind_tools(_TOOLS) if with_tools else llm


def _count_tokens(response: AIMessage) -> int:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return 0
    if isinstance(meta, dict):
        return meta.get("total_tokens", 0)
    return getattr(meta, "total_tokens", 0)


# ── 代码提取 ─────────────────────────────────────────────────
def _extract_code_block(text: str) -> str:
    """
    从 LLM 回答中提取最后一个 Python 代码块。

    设计目标:
    - 优先提取 ```python ... ``` / ```py ... ``` 代码块
    - 兼容少量“直接输出裸 Python 源码”的情况
    - 明确避免把工具调用 JSON 误判为源码
    """
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    for pattern in (
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ):
        matches = re.findall(pattern, stripped, re.DOTALL | re.IGNORECASE)
        if matches:
            candidate = _normalize_code_candidate(matches[-1])
            if candidate:
                return candidate

    # 兼容模型直接输出裸 Python 源码，但拒绝工具调用 JSON / 非源码围栏。
    return _normalize_code_candidate(stripped)


def _normalize_code_candidate(text: str) -> str:
    """
    规范化候选源码，必要时从工具调用 JSON 中剥离出真正的 code 字段。
    """
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    nested = _extract_code_from_tool_payload(stripped)
    if nested:
        return _normalize_code_candidate(nested)

    lines = stripped.splitlines()
    first_code_line = _first_python_line_index(lines)
    if first_code_line is not None and first_code_line > 0:
        stripped = "\n".join(lines[first_code_line:]).strip()

    return stripped if _looks_like_python_source(stripped) else ""


def _extract_code_from_tool_payload(text: str) -> str:
    """从 execute_python 工具调用载荷中提取内层 code 字段。"""
    if not text.startswith("{"):
        return ""

    try:
        payload = json.loads(text)
    except Exception:
        return ""

    if isinstance(payload, dict):
        if isinstance(payload.get("code"), str):
            return payload["code"]

        if payload.get("name") == "execute_python":
            arguments = payload.get("arguments")
            if isinstance(arguments, dict) and isinstance(arguments.get("code"), str):
                return arguments["code"]

    return ""


def _looks_like_python_source(text: str) -> bool:
    """尽量保守地判断文本是否像 Python 源码。"""
    if not text:
        return False

    python_markers = (
        "\nimport ",
        "\nfrom ",
        "\ndef ",
        "\nclass ",
        "\nfor ",
        "\nwhile ",
        "\nif __name__",
        "print(",
        "torch.",
        "plt.",
        "=",
    )
    return any(marker in f"\n{text}" for marker in python_markers)


def _first_python_line_index(lines: list[str]) -> int | None:
    """找到首个明显像 Python 代码的行，用于去掉前置的 shell 输出。"""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith((
            "import ",
            "from ",
            "def ",
            "class ",
            "@",
            "if __name__",
            "#",
        )):
            return idx
    return 0 if lines else None


def _build_memory_context(state: AgentState) -> str:
    """Format reusable memory context for the Coder agent."""
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

    experience_text = format_experience_hints(state.get("experience_hints") or [])
    if experience_text:
        blocks.append("Relevant past experience:\n" + experience_text)

    return "\n\n".join(blocks)


# ── 执行结果解析 ──────────────────────────────────────────────
def _parse_execution_result(tool_output: str) -> tuple[bool, str, str]:
    """
    解析 execute_python 工具的输出字符串。

    Returns:
        (success, stdout, stderr, artifacts)
    """
    success = "[执行成功]" in tool_output
    stdout  = ""
    stderr  = ""
    artifacts: list[str] = []

    if "[stdout]" in tool_output:
        after_stdout = tool_output.split("[stdout]", 1)[1]
        end_candidates = [
            after_stdout.find(marker)
            for marker in ("[stderr]", "[artifacts]")
            if marker in after_stdout
        ]
        if end_candidates:
            stdout = after_stdout[: min(end_candidates)].strip()
        else:
            stdout = after_stdout.strip()

    if "[stderr]" in tool_output:
        after_stderr = tool_output.split("[stderr]", 1)[1]
        if "[artifacts]" in after_stderr:
            stderr = after_stderr.split("[artifacts]", 1)[0].strip()
        else:
            stderr = after_stderr.strip()

    if "[artifacts]" in tool_output:
        after_artifacts = tool_output.split("[artifacts]", 1)[1]
        artifacts = [
            line.strip()
            for line in after_artifacts.splitlines()
            if line.strip()
        ]

    return success, stdout, stderr, artifacts


# ── ReAct 核心循环 ───────────────────────────────────────────
def _react_loop(
    messages: list,
) -> tuple[str, str, str, bool, list[str], list]:
    """
    Coder ReAct 循环。

    Returns:
        generated_code, stdout, stderr, success, artifact_paths, updated_messages
    """
    llm_with_tools = _build_llm(with_tools=True)

    generated_code = ""
    executed_code  = ""
    last_stdout    = ""
    last_stderr    = ""
    last_success   = False
    artifact_paths: list[str] = []

    for _ in range(MAX_ITER):
        with timer() as t:
            response = llm_with_tools.invoke(messages)

        tokens = _count_tokens(response)
        cost_tracker.record(AGENT_NAME, MODEL_CODER, tokens)
        tracer.log_llm_call(
            agent=AGENT_NAME,
            prompt=str(messages[-1].content)[:500],
            response=str(response.content)[:500],
            model=MODEL_CODER,
            tokens_used=tokens,
            duration_ms=t.ms,
        )

        # 从文本内容中尝试提取代码
        if response.content:
            extracted = _extract_code_block(response.content)
            if extracted:
                generated_code = extracted

        # 无工具调用 → 结束循环
        if not response.tool_calls:
            final_code = executed_code or generated_code
            return (
                final_code,
                last_stdout,
                last_stderr,
                last_success,
                sorted(set(artifact_paths)),
                messages + [response],
            )

        # 执行工具
        messages = messages + [response]
        for tc in response.tool_calls:
            name    = tc["name"]
            args    = tc["args"]
            call_id = tc["id"]

            with timer() as tt:
                fn = _TOOL_MAP.get(name)
                if fn is None:
                    result = f"[错误] 未知工具: {name}"
                else:
                    try:
                        result = fn.invoke(args)
                    except Exception as e:
                        result = f"[工具异常] {e}"

            tracer.log_tool_call(
                agent=AGENT_NAME,
                tool_name=name,
                tool_input=args,
                tool_output=str(result)[:500],
                duration_ms=tt.ms,
            )

            # 追踪执行结果 & 捕获代码参数
            if name == "execute_python":
                (
                    last_success,
                    last_stdout,
                    last_stderr,
                    new_artifacts,
                ) = _parse_execution_result(str(result))
                if isinstance(args, dict) and "code" in args:
                    normalized = _normalize_code_candidate(str(args["code"]))
                    generated_code = normalized or str(args["code"])
                    executed_code = generated_code
                artifact_paths.extend(new_artifacts)

            messages = messages + [
                ToolMessage(content=str(result), tool_call_id=call_id)
            ]

    # 迭代上限后强制汇总（不再调工具）
    llm_plain = _build_llm(with_tools=False)
    force_msg = HumanMessage(content="执行尝试已达上限，请总结当前进展和代码状态。")
    with timer() as t:
        final = llm_plain.invoke(messages + [force_msg])

    tokens = _count_tokens(final)
    cost_tracker.record(AGENT_NAME, MODEL_CODER, tokens)
    final_code = executed_code or generated_code
    return (
        final_code,
        last_stdout,
        last_stderr,
        last_success,
        sorted(set(artifact_paths)),
        messages + [force_msg, final],
    )


# ── Agent 节点入口 ───────────────────────────────────────────
def run_coder(state: AgentState) -> dict:
    """
    LangGraph 节点函数。
    初次调用时依据科研方案生成代码；重试时依据错误信息 debug。
    """
    query       = state["query"]
    intent      = state.get("intent", "code")
    retry_count = state.get("code_retry_count", 0)
    history     = list(state.get("messages") or [])

    tracer.log_state_transition(
        from_step=state.get("current_step", "researcher"),
        to_step="coder",
        intent=intent,
    )

    if retry_count == 0:
        # 初次：携带科研背景
        context_parts: list[str] = []
        memory_context = _build_memory_context(state)
        if memory_context:
            context_parts.append(memory_context)
        if state.get("literature_report"):
            context_parts.append(
                f"【文献背景】\n{state['literature_report'][:1000]}"
            )
        if state.get("design_proposal"):
            context_parts.append(
                f"【技术方案】\n{state['design_proposal'][:800]}"
            )
        context = "\n\n".join(context_parts)
        prefix  = f"{context}\n\n" if context else ""
        user_text = (
            f"{prefix}请根据以上背景，为以下需求编写 PINN Python 代码并执行验证：\n\n{query}"
        )
    else:
        # 重试：携带失败信息
        from config import EXAMINER_MAX_RETRIES
        retry_text = _RETRY_TEMPLATE.format(
            code=state.get("generated_code", "（无代码）"),
            stderr=state.get("execution_stderr", ""),
            stdout=state.get("execution_stdout", ""),
            retry_num=retry_count,
            max_retries=EXAMINER_MAX_RETRIES,
        )
        memory_context = _build_memory_context(state)
        user_text = (
            f"{memory_context}\n\n{retry_text}"
            if memory_context else retry_text
        )

    messages = history + [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    code, stdout, stderr, success, artifact_paths, updated_messages = _react_loop(messages)
    merged_artifacts = sorted(set(list(state.get("artifact_paths") or []) + artifact_paths))

    return {
        "current_step":      "coder",
        "generated_code":    code,
        "execution_stdout":  stdout,
        "execution_stderr":  stderr,
        "execution_success": success,
        "artifact_paths":    merged_artifacts,
        "code_retry_count":  retry_count,   # graph.py 中在失败重试时递增
        "messages":          updated_messages,
    }
