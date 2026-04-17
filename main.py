"""
PINN Research Agent V2 — 主启动入口

用法:
    python main.py              # 启动 TUI 模式（默认）
    python main.py --debug      # 启动 TUI 并默认展开 Debug 面板
    python main.py --cli        # 纯 CLI 模式（无 TUI，适合脚本调试）
    python main.py --query "PINN 损失函数"   # 单次问答后退出
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import TUI_SHOW_DEBUG, LOGS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="PINN Research Agent V2 — Multi-Agent 科研助手"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="启动时展开 Debug 面板（显示 Agent 思考链）"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="纯 CLI 模式，不启动 TUI（用于调试）"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="单次问答模式：传入问题字符串，输出结果后退出"
    )
    return parser.parse_args()


async def run_cli_once(query: str):
    """单次问答模式（无 TUI）"""
    from orchestrator.graph import build_graph
    from memory import SessionManager

    print(f"\n🔬 PINN Agent V2 — 单次查询模式")
    print(f"📝 问题: {query}\n")
    print("─" * 60)

    graph = build_graph()
    session_id = SessionManager().reset_session(prefix="cli-once")
    # MemorySaver checkpointer 必须提供 thread_id，否则抛 ValueError
    result = await graph.ainvoke(
        {"query": query, "messages": [], "session_id": session_id},
        config={"configurable": {"thread_id": session_id}},
    )

    print("\n─" * 60)
    print("✅ 最终回答:\n")
    print(result.get("final_answer", "（无结果）"))


def run_cli_interactive():
    """纯 CLI 交互模式（无 TUI）"""
    from orchestrator.graph import build_graph
    from memory import SessionManager

    print("🔬 PINN Agent V2 — CLI 调试模式（输入 'exit' 退出）\n")
    graph = build_graph()
    session_manager = SessionManager()
    session_id = session_manager.reset_session(prefix="cli")

    while True:
        try:
            query = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("👋 再见！")
            break
        if query.lower() in {"clear", "/clear"}:
            session_id = session_manager.reset_session(prefix="cli")
            graph = build_graph()
            print(f"🧠 已切换到新会话: {session_id}\n")
            continue
        if not query:
            continue

        result = asyncio.run(
            graph.ainvoke(
                {"query": query, "messages": [], "session_id": session_id},
                config={"configurable": {"thread_id": session_id}},
            )
        )
        print(f"\nAgent > {result.get('final_answer', '（无结果）')}\n")


def run_tui(debug: bool = False):
    """启动 Textual TUI"""
    from tui.app import PINNAgentApp

    app = PINNAgentApp(show_debug=debug)
    app.run()


def main():
    # 确保日志目录存在
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    args = parse_args()

    if args.query:
        # 单次问答
        asyncio.run(run_cli_once(args.query))
    elif args.cli:
        # 纯 CLI 交互
        run_cli_interactive()
    else:
        # TUI 模式（默认）
        show_debug = args.debug or TUI_SHOW_DEBUG
        run_tui(debug=show_debug)


if __name__ == "__main__":
    main()
