"""
PINN Agent V2 — Textual TUI 主应用

布局:
    ┌──────────────────────────────────────────────────┐
    │  左: Agent + Debug    │  中: 对话区               │
    │  (Flow / Trace)       │  (ChatView)               │
    │                       │                           │
    │                       │  右: 本轮产物 / 会话记忆   │
    │                       │  (Artifacts / Memory)     │
    ├──────────────────────────────────────────────────┤
    │  底部状态栏: Token用量 | 模型 | 耗时 | 快捷键提示  │
    └──────────────────────────────────────────────────┘

快捷键:
    d → 切换 Debug 面板（工具日志）
    c → 清空当前会话
    s → 保存对话到文件
    q → 退出
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    Log,
    Markdown,
    Static,
)

from config import (
    LOGS_DIR,
    ROOT_DIR,
    TUI_STREAM_DELAY,
    LLM_PROVIDER_LABEL,
    MODEL_SUMMARY,
    get_model_for_step,
)
from memory import SessionManager
from observability.cost_tracker import cost_tracker
from observability.tracer import tracer


# ── 自定义 Widget ─────────────────────────────────────────────

class AgentStatusPanel(Static):
    """左侧 Agent 状态面板 — 显示当前激活 Agent 和 SOP 进度"""

    DEFAULT_CSS = """
    AgentStatusPanel {
        width: 1fr;
        height: 16;
        border: round #4f8cff;
        background: #141922;
        padding: 1 2;
        margin-bottom: 1;
    }
    AgentStatusPanel Label {
        color: #dbe4ee;
    }
    #agent-kicker {
        color: #93c5fd;
        text-style: bold;
        margin-bottom: 1;
    }
    #agent-note {
        color: #94a3b8;
        margin-bottom: 1;
    }
    #agent-active {
        color: #cbd5e1;
        margin-top: 1;
    }
    #agent-hint {
        color: #64748b;
        margin-top: 1;
    }
    #agent-shortcuts {
        color: #64748b;
    }
    .step-done {
        color: #86efac;
    }
    .step-pending {
        color: #94a3b8;
    }
    """

    SOP_STEPS = [
        "parse_intent",
        "researcher",
        "examiner",
        "coder",
        "synthesize",
    ]

    STEP_LABELS = {
        "parse_intent": "🔍 意图识别",
        "researcher":   "📚 文献研究",
        "examiner":     "🔬 质量审查",
        "coder":        "💻 代码编写",
        "synthesize":   "✅ 汇总输出",
    }

    def __init__(self):
        super().__init__()
        self._current_step = ""
        self._completed_steps: set[str] = set()
        self.border_title = " Agent Flow "

    def compose(self) -> ComposeResult:
        yield Label("多 Agent 协作流", id="agent-kicker")
        yield Label("等待任务输入后启动。", id="agent-note")
        for step in self.SOP_STEPS:
            yield Label(f"· {self.STEP_LABELS[step]}", id=f"step-{step}", classes="step-pending")
        yield Label("当前阶段: 待命", id="agent-active")
        yield Label("快捷键", id="agent-hint")
        yield Label("d 调试 | c 清空", id="agent-shortcuts")

    def reset(self) -> None:
        """重置当前查询的步骤状态。"""
        self._current_step = ""
        self._completed_steps.clear()
        for step in self.SOP_STEPS:
            label = self.query_one(f"#step-{step}", Label)
            self._set_step_visual(label, "pending", step)
        self.query_one("#agent-active", Label).update("当前阶段: 待命")

    def update_step(self, step: str) -> None:
        """高亮当前激活步骤"""
        if step not in self.STEP_LABELS:
            return
        if self._current_step and self._current_step != step:
            self._completed_steps.add(self._current_step)
        self._current_step = step
        for s in self.SOP_STEPS:
            label = self.query_one(f"#step-{s}", Label)
            if s == step:
                self._set_step_visual(label, "active", s)
            elif s in self._completed_steps:
                self._set_step_visual(label, "done", s)
            else:
                self._set_step_visual(label, "pending", s)

        active_label = self.query_one("#agent-active", Label)
        active_label.update(f"当前阶段: {self.STEP_LABELS.get(step, step)}")

    def _set_step_visual(self, label: Label, state: str, step: str) -> None:
        label.remove_class("active")
        label.remove_class("step-done")
        label.remove_class("step-pending")

        if state == "active":
            label.update(f"▶ {self.STEP_LABELS[step]}")
            label.add_class("active")
            return
        if state == "done":
            label.update(f"✓ {self.STEP_LABELS[step]}")
            label.add_class("step-done")
            return

        label.update(f"· {self.STEP_LABELS[step]}")
        label.add_class("step-pending")


class ChatView(Vertical):
    """中部对话区 — Markdown 渲染 + 流式打字机输出"""

    DEFAULT_CSS = """
    ChatView {
        border: round #4f8cff;
        background: #111722;
        padding: 1;
    }
    #chat-output {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }
    #chat-input {
        height: 3;
        dock: bottom;
        margin: 1 1 0 1;
        border: round #334155;
        background: #0b1220;
        color: #e2e8f0;
    }
    """

    def __init__(self):
        super().__init__()
        self.border_title = " Conversation "
        self.border_subtitle = " Research / Code / Review "
        self._md_content: str = self._empty_state_markdown()

    def compose(self) -> ComposeResult:
        yield Markdown(self._md_content, id="chat-output")
        yield Input(
            placeholder="输入研究任务、代码需求或调试问题，按 Enter 发送...",
            id="chat-input",
        )

    async def stream_response(self, text: str, animate: bool = True) -> None:
        """流式输出回答；debug 场景可关闭动画以减少卡顿。"""
        output = self.query_one("#chat-output", Markdown)
        if not text:
            self._md_content += "\n\n"
            return

        if not animate:
            self._md_content += text + "\n\n"
            output.update(self._md_content)
            return

        # 长文本限制刷新次数，避免 debug 模式下逐字渲染导致明显卡顿。
        max_updates = 120
        chunk_size = max(1, math.ceil(len(text) / min(len(text), max_updates)))
        streamed = ""
        for idx in range(0, len(text), chunk_size):
            streamed = text[: idx + chunk_size]
            output.update(self._md_content + streamed)
            await asyncio.sleep(TUI_STREAM_DELAY)
        self._md_content += streamed + "\n\n"

    def append_user_message(self, msg: str) -> None:
        if self._md_content == self._empty_state_markdown():
            self._md_content = ""
        self._md_content += f"**You**: {msg}\n\n**Agent**: "
        self.query_one("#chat-output", Markdown).update(self._md_content)

    def clear(self) -> None:
        self._md_content = self._empty_state_markdown()
        self.query_one("#chat-output", Markdown).update(self._md_content)

    def _empty_state_markdown(self) -> str:
        return (
            "## 欢迎使用 PINN Research Agent V2\n\n"
            "- 在这里输入科研问题、代码任务或 debug 请求。\n"
            "- 左下会展示 debug trace，右侧展示本轮产物和 session memory。\n"
            "- 推荐示例：`先综述 PINN 损失函数，再写一个最小可运行示例并执行。`\n"
        )


class ToolLogPanel(Log):
    """右侧工具调用日志面板（可折叠）"""

    MAX_LINES = 200

    DEFAULT_CSS = """
    ToolLogPanel {
        width: 1fr;
        height: 1fr;
        border: round #f59e0b;
        background: #141922;
        color: #f8fafc;
        padding: 0 1;
        display: none;
    }
    ToolLogPanel.visible {
        display: block;
    }
    """

    def __init__(self):
        super().__init__()
        self.border_title = " Debug Trace "
        self.border_subtitle = " toggle: d "
        self._has_placeholder = False

    def on_mount(self) -> None:
        self.reset_panel()

    def log_tool(self, agent: str, tool: str, summary: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._append_line(f"[{ts}] [{agent}] {tool}")
        self._append_line(f"  └ {summary[:60]}")

    def reset_panel(self) -> None:
        """清空当前查询的调试日志。"""
        clear_fn = getattr(self, "clear", None)
        if callable(clear_fn):
            clear_fn()
        self._has_placeholder = False
        self._write_placeholder()

    def log_record(self, record: dict[str, Any]) -> None:
        """按 trace record 类型写入易读日志。"""
        ts = self._format_ts(record.get("ts", ""))
        event_type = record.get("type", "")

        if event_type == "state_transition":
            from_step = record.get("from_step", "?")
            to_step = record.get("to_step", "?")
            intent = record.get("intent", "")
            self._append_line(f"[{ts}] [SOP] {from_step} -> {to_step}")
            if intent:
                self._append_line(f"  └ intent={intent}")
            return

        if event_type == "tool_call":
            agent = record.get("agent", "Agent")
            tool = record.get("tool", "tool")
            label = "SANDBOX" if tool in {"execute_python", "run_shell"} else "TOOL"
            summary = self._summarize_tool_call(
                tool,
                record.get("input") or {},
                str(record.get("output", "")),
            )
            self._append_line(f"[{ts}] [{agent}] {label}:{tool}")
            self._append_line(f"  └ {summary}")
            return

        if event_type == "examiner_verdict":
            verdict = record.get("verdict", "?")
            retry = record.get("retry_count", 0)
            review = str(record.get("review", "")).replace("\n", " ")
            self._append_line(f"[{ts}] [Examiner] VERDICT {verdict}")
            self._append_line(f"  └ retry={retry}; {review[:90]}")
            return

        if event_type == "llm_call":
            agent = record.get("agent", "Agent")
            tokens = record.get("tokens_used", 0)
            duration_ms = record.get("duration_ms", 0)
            model = record.get("model", "")
            self._append_line(f"[{ts}] [{agent}] LLM")
            self._append_line(
                f"  └ model={model}; tokens={tokens}; duration={duration_ms}ms"
            )

    def _append_line(self, line: str) -> None:
        """限制日志面板保留行数，避免长会话时控件越来越慢。"""
        if self._has_placeholder:
            clear_fn = getattr(self, "clear", None)
            if callable(clear_fn):
                clear_fn()
            self._has_placeholder = False
        self.write_line(line)
        try:
            lines = getattr(self, "lines", None)
            line_count = len(lines) if lines is not None else 0
            overflow = line_count - self.MAX_LINES
            if overflow > 0:
                self._trim_lines(overflow)
        except Exception:
            # 裁剪失败时保持可用性，宁可不裁剪也不影响主流程。
            pass

    def _trim_lines(self, count: int) -> None:
        """移除最旧的若干行，控制 Log widget 体量。"""
        lines = getattr(self, "lines", None)
        if lines is None:
            return
        retained = list(lines)[count:]
        clear_fn = getattr(self, "clear", None)
        if callable(clear_fn):
            clear_fn()
        for line in retained:
            self.write_line(str(line))

    def _format_ts(self, ts: str) -> str:
        try:
            return datetime.fromisoformat(ts).strftime("%H:%M:%S")
        except Exception:
            return datetime.now().strftime("%H:%M:%S")

    def _summarize_tool_call(
        self,
        tool: str,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> str:
        if tool == "run_shell":
            cmd = str(tool_input.get("cmd", "")).strip()
            return f"cmd={cmd or 'N/A'}"
        if tool == "execute_python":
            timeout = tool_input.get("timeout", "default")
            status = tool_output.splitlines()[0] if tool_output else "无输出"
            return f"python timeout={timeout}; {status[:70]}"
        if tool in {"search_local_papers", "search_arxiv", "web_search"}:
            query = str(tool_input.get("query", "")).strip()
            return f"query={query[:80]}"
        if tool in {"read_file", "write_file"}:
            path = str(tool_input.get("path", "")).strip()
            return f"path={path[:80]}"
        return str(tool_input)[:90] if tool_input else tool_output[:90]

    def _write_placeholder(self) -> None:
        self.write_line("等待本轮 trace / tool 调用...")
        self.write_line("这里会显示 SOP 跳转、LLM 调用、沙盒执行摘要。")
        self._has_placeholder = True


class ArtifactPanel(Static):
    """显示本轮查询导出的宿主机产物文件。"""

    DEFAULT_CSS = """
    ArtifactPanel {
        width: 1fr;
        height: 9;
        border: round #fbbf24;
        background: #141922;
        padding: 1 2;
        margin-bottom: 1;
    }
    """

    def __init__(self):
        self._artifact_paths: list[str] = []
        super().__init__(self._build_content())
        self.border_title = " Current Run Outputs "
        self.border_subtitle = " 本轮导出 "

    def set_artifacts(self, artifact_paths: list[str]) -> None:
        self._artifact_paths = artifact_paths
        self._refresh_content()

    def reset(self) -> None:
        self._artifact_paths = []
        self._refresh_content()

    def _build_content(self) -> str:
        lines = ["导出目录: outputs/sandbox_runs", ""]
        if not self._artifact_paths:
            lines.append("本轮暂无产物")
            lines.append("运行代码后会显示这一次生成的文件。")
        else:
            lines.append(f"本次产物: {len(self._artifact_paths[-6:])} 个")
            for path in self._artifact_paths[-6:]:
                lines.append(f"• {self._clip(self._format_path(path), 30)}")
        return "\n".join(lines)

    def _refresh_content(self) -> None:
        self.update(self._build_content())

    def _format_path(self, path: str) -> str:
        try:
            rel = Path(path).resolve().relative_to(Path(ROOT_DIR).resolve())
            return str(rel)
        except Exception:
            return path

    def _clip(self, text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return "…" + text[-(limit - 1) :]


class MemoryStatusPanel(Static):
    """显示当前 session 的短期记忆状态，便于 demo 展示 Agent 不是无状态聊天。"""

    DEFAULT_CSS = """
    MemoryStatusPanel {
        width: 1fr;
        height: 1fr;
        border: round #22c55e;
        background: #141922;
        padding: 1 2;
    }
    """

    def __init__(self, session_id: str = "", summary: dict[str, Any] | None = None):
        self._session_id = session_id
        self._summary = summary or {}
        super().__init__(self._build_content())
        self.border_title = " Session Memory "
        self.border_subtitle = " 会话上下文 "

    def set_summary(self, session_id: str, summary: dict[str, Any] | None) -> None:
        self._session_id = session_id
        self._summary = summary or {}
        self._refresh_content()

    def reset(self, session_id: str, summary: dict[str, Any] | None = None) -> None:
        self.set_summary(session_id, summary or {})

    def _build_content(self) -> str:
        lines = [
            f"会话    {self._format_session_id()}",
            f"压缩    {self._summary.get('compressed_turns', 0)} 次",
            "",
        ]

        recent_query = self._latest_query()
        lines.append(f"查询    {self._clip(recent_query or '暂无', 24)}")

        error = (
            str(self._summary.get("last_failure_error_summary", "")).strip()
            or str(self._summary.get("last_error_summary", "")).strip()
        )
        lines.append(f"错误    {self._clip(error or '暂无', 24)}")

        artifacts = self._memory_artifacts()
        if artifacts:
            lines.append("")
            lines.append("最近成功产物")
            for path in artifacts[-3:]:
                lines.append(f"• {self._clip(self._format_path(path), 30)}")
        else:
            lines.append("成功产物    暂无")

        return "\n".join(lines)

    def _refresh_content(self) -> None:
        self.update(self._build_content())

    def _format_session_id(self) -> str:
        if not self._session_id:
            return "N/A"
        return self._session_id.split("-")[-1]

    def _latest_query(self) -> str:
        queries = [
            str(item).strip()
            for item in self._summary.get("recent_queries") or []
            if str(item).strip()
        ]
        return queries[-1] if queries else ""

    def _memory_artifacts(self) -> list[str]:
        successful_artifacts = [
            str(item).strip()
            for item in self._summary.get("last_successful_artifacts") or []
            if str(item).strip()
        ]
        if successful_artifacts:
            return successful_artifacts

        return [
            str(item).strip()
            for item in self._summary.get("last_artifacts") or []
            if str(item).strip()
        ]

    def _format_path(self, path: str) -> str:
        try:
            rel = Path(path).resolve().relative_to(Path(ROOT_DIR).resolve())
            return str(rel)
        except Exception:
            return path

    def _clip(self, text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return "…" + text[-(limit - 1) :]


# ── 主 App ────────────────────────────────────────────────────

class PINNAgentApp(App):
    """PINN Research Agent V2 TUI"""

    TITLE = "PINN Research Agent V2"
    SUB_TITLE = "Multi-Agent 科研助手"

    CSS = """
    Screen {
        background: #0f141b;
        color: #e5e7eb;
    }
    Horizontal {
        height: 1fr;
        padding: 1 1 0 1;
    }
    #left-stack {
        width: 38;
        margin-right: 1;
    }
    #right-stack {
        width: 34;
        margin-left: 1;
    }
    .active {
        color: #f8fafc;
        background: #1e3a8a;
        text-style: bold;
    }
    #status-bar {
        height: 1;
        background: #111827;
        color: #cbd5e1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("d",     "toggle_debug",   "Debug面板"),
        Binding("c",     "clear_chat",     "清空会话"),
        Binding("s",     "save_chat",      "保存对话"),
        Binding("q",     "quit",           "退出"),
        Binding("ctrl+c","quit",           "退出", show=False),
    ]

    def __init__(self, show_debug: bool = False):
        super().__init__()
        self._show_debug = show_debug
        self._chat_history: list[dict] = []
        self._active_step = ""
        self._session_manager = SessionManager()
        self._session_id = self._session_manager.reset_session(prefix="tui")
        self._graph = self._build_graph()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left-stack"):
                yield AgentStatusPanel()
                yield ToolLogPanel()
            yield ChatView()
            with Vertical(id="right-stack"):
                yield ArtifactPanel()
                yield MemoryStatusPanel(
                    self._session_id,
                    self._session_manager.load_summary(self._session_id),
                )
        yield Static(self._build_status_bar_text(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        if self._show_debug:
            self.action_toggle_debug()
        # 刷新状态栏
        self.set_interval(2.0, self._refresh_status_bar)

    def _refresh_status_bar(self) -> None:
        self.query_one("#status-bar", Static).update(self._build_status_bar_text())

    def _build_status_bar_text(self) -> str:
        """生成底部状态栏摘要，显示 Token / Provider / 当前模型。"""
        token_summary = cost_tracker.summary
        session_label = self._session_id.split("-")[-1] if self._session_id else "N/A"
        if self._active_step:
            model_name = get_model_for_step(self._active_step)
            return (
                f"{token_summary} | Provider: {LLM_PROVIDER_LABEL}"
                f" | Active: {self._active_step} -> {model_name}"
                f" | Session: {session_label}"
            )
        return (
            f"{token_summary} | Provider: {LLM_PROVIDER_LABEL}"
            f" | Models: {MODEL_SUMMARY}"
            f" | Session: {session_label}"
        )

    def _set_active_step(self, step: str) -> None:
        """更新当前步骤，并同步刷新状态栏。"""
        self._active_step = step
        self._refresh_status_bar()

    # ── 快捷键动作 ────────────────────────────────────────────

    def action_toggle_debug(self) -> None:
        panel = self.query_one(ToolLogPanel)
        panel.toggle_class("visible")

    def action_clear_chat(self) -> None:
        self.query_one(ChatView).clear()
        self.query_one(AgentStatusPanel).reset()
        self.query_one(ToolLogPanel).reset_panel()
        self.query_one(ArtifactPanel).reset()
        cost_tracker.reset_session()
        self._chat_history.clear()
        self._active_step = ""
        self._session_id = self._session_manager.reset_session(prefix="tui")
        self._graph = self._build_graph()
        self.query_one(MemoryStatusPanel).reset(
            self._session_id,
            self._session_manager.load_summary(self._session_id),
        )
        self._refresh_status_bar()

    def action_save_chat(self) -> None:
        log_dir = Path(LOGS_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"chat_{ts}.md"
        content = "\n\n".join(
            f"**{m['role']}**: {m['content']}"
            for m in self._chat_history
        )
        path.write_text(content, encoding="utf-8")
        self.notify(f"💾 已保存至 {path.name}", severity="information")

    # ── 用户输入处理 ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self._chat_history.append({"role": "user", "content": query})
        self.run_worker(self._process_query(query), exclusive=True)

    async def _process_query(self, query: str) -> None:
        """异步调用 LangGraph 图并流式输出结果"""
        chat_view  = self.query_one(ChatView)
        status     = self.query_one(AgentStatusPanel)
        tool_log   = self.query_one(ToolLogPanel)
        artifacts  = self.query_one(ArtifactPanel)
        memory     = self.query_one(MemoryStatusPanel)

        status.reset()
        tool_log.reset_panel()
        artifacts.reset()
        memory.set_summary(
            self._session_id,
            self._session_manager.load_summary(self._session_id),
        )
        chat_view.append_user_message(query)
        status.update_step("parse_intent")
        self._set_active_step("parse_intent")

        trace_start_offset = self._current_trace_offset()
        stop_trace_poll = asyncio.Event()
        trace_task = asyncio.create_task(
            self._poll_trace_events(
                status,
                tool_log,
                start_offset=trace_start_offset,
                stop_event=stop_trace_poll,
            )
        )
        trace_drained = False

        try:
            result = await asyncio.to_thread(self._invoke_graph_sync, query)

            # 先收尾 trace 轮询，避免晚到的 examiner 事件把最终步骤覆盖回去。
            stop_trace_poll.set()
            try:
                await trace_task
                trace_drained = True
            except Exception:
                pass

            # 更新状态
            status.update_step("synthesize")
            self._set_active_step("synthesize")
            final = result.get("final_answer", "（无结果）")
            artifact_paths = list(result.get("artifact_paths") or [])
            artifacts.set_artifacts(artifact_paths)
            memory.set_summary(
                self._session_id,
                result.get("session_summary")
                or self._session_manager.load_summary(self._session_id),
            )
            self._chat_history.append({"role": "agent", "content": final})

            # 流式输出
            await chat_view.stream_response(final, animate=not self._show_debug)

        except Exception as exc:
            await chat_view.stream_response(
                f"❌ 错误: {exc}",
                animate=not self._show_debug,
            )
            tool_log.log_tool("System", "ERROR", str(exc))

        finally:
            if not stop_trace_poll.is_set():
                stop_trace_poll.set()
            if not trace_drained:
                try:
                    await trace_task
                except Exception:
                    pass
            self._refresh_status_bar()

    def _invoke_graph_sync(self, query: str) -> dict[str, Any]:
        """在线程中运行图，避免阻塞 TUI 事件循环。"""
        config = {"configurable": {"thread_id": self._session_id}}
        return self._graph.invoke(
            {"query": query, "messages": [], "session_id": self._session_id},
            config=config,
        )

    def _build_graph(self):
        """Create a graph instance whose in-memory checkpoints survive across the current session."""
        from orchestrator.graph import build_graph

        return build_graph()

    async def _poll_trace_events(
        self,
        status: AgentStatusPanel,
        tool_log: ToolLogPanel,
        start_offset: int,
        stop_event: asyncio.Event,
    ) -> None:
        """轮询 tracer JSONL，实时把当前查询的 trace 显示到 TUI。"""
        next_offset = start_offset
        while not stop_event.is_set():
            next_offset = self._drain_trace_events(status, tool_log, next_offset)
            await asyncio.sleep(0.25)

        self._drain_trace_events(status, tool_log, next_offset)

    def _drain_trace_events(
        self,
        status: AgentStatusPanel,
        tool_log: ToolLogPanel,
        start_offset: int,
    ) -> int:
        """消费当前会话新增 trace 记录，并同步到 UI。"""
        next_offset, records = tracer.read_session_records_from_offset(
            start_offset=start_offset
        )
        if not records:
            return next_offset

        for record in records:
            tool_log.log_record(record)
            if record.get("type") == "state_transition":
                step = record.get("to_step", "")
                status.update_step(step)
                self._set_active_step(step)

        return next_offset

    def _current_trace_offset(self) -> int:
        """读取当前 trace 文件大小，作为本次查询的增量读取起点。"""
        path = tracer.log_path
        if path is None or not path.exists():
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0
