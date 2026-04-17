"""
PINN Agent V2 — 安装验证脚本

运行: python verify_install.py
逐项检查各依赖是否可正常导入，并验证核心模块的基础逻辑。
不需要 Ollama / Docker / CUDA 在线，纯离线结构检查。
"""

import sys
import importlib

# ── 颜色输出（兼容 Windows / WSL）───────────────────────────
def ok(msg):  print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}");
def section(title): print(f"\n{'─'*50}\n  {title}\n{'─'*50}")

results = {"pass": 0, "fail": 0}

def check_import(pkg_name, import_as=None, attr=None):
    """尝试 import 包，可选检查特定属性是否存在"""
    try:
        mod = importlib.import_module(import_as or pkg_name)
        if attr and not hasattr(mod, attr):
            raise AttributeError(f"缺少属性: {attr}")
        ok(f"{pkg_name}")
        results["pass"] += 1
    except Exception as e:
        fail(f"{pkg_name}: {e}")
        results["fail"] += 1


# ══════════════════════════════════════════════
section("1. 编排框架")
# ══════════════════════════════════════════════
check_import("langgraph",           "langgraph.graph",               "StateGraph")
check_import("langgraph.checkpoint","langgraph.checkpoint.memory",  "MemorySaver")
# langchain 0.3+ 将核心类型全部迁移到 langchain_core，不再从 langchain.schema 导入
check_import("langchain_core (messages)",  "langchain_core.messages",  "BaseMessage")
check_import("langchain_core (documents)", "langchain_core.documents", "Document")
check_import("langchain_openai",    "langchain_openai",              "ChatOpenAI")
check_import("openai",              "openai",                        "OpenAI")

# ══════════════════════════════════════════════
section("2. RAG / 向量库")
# ══════════════════════════════════════════════
check_import("chromadb",               "chromadb")
check_import("sentence_transformers",  "sentence_transformers",    "CrossEncoder")
check_import("pypdf",                  "pypdf",                    "PdfReader")

# ══════════════════════════════════════════════
section("3. TUI")
# ══════════════════════════════════════════════
check_import("textual",   "textual.app",     "App")
check_import("textual",   "textual.widgets", "Markdown")
check_import("rich",      "rich",            "print")

# ══════════════════════════════════════════════
section("4. 工具层")
# ══════════════════════════════════════════════
check_import("arxiv",             "arxiv")
check_import("duckduckgo_search", "duckduckgo_search", "DDGS")
check_import("sympy",             "sympy",             "symbols")
check_import("docker",            "docker",            "from_env")

# ══════════════════════════════════════════════
section("5. 工具库")
# ══════════════════════════════════════════════
check_import("pydantic",      "pydantic",      "BaseModel")
check_import("dotenv",        "dotenv",        "load_dotenv")
check_import("tiktoken",      "tiktoken")
check_import("httpx",         "httpx",         "AsyncClient")
check_import("tenacity",      "tenacity",      "retry")
check_import("langsmith",     "langsmith",     "Client")

# ══════════════════════════════════════════════
section("6. 科学计算")
# ══════════════════════════════════════════════
check_import("numpy",      "numpy",      "ndarray")
check_import("matplotlib", "matplotlib")
check_import("scipy",      "scipy")
check_import("torch",      "torch",      "Tensor")

# ══════════════════════════════════════════════
section("7. 项目核心模块（不依赖第三方的部分）")
# ══════════════════════════════════════════════
sys.path.insert(0, ".")

try:
    import config
    assert hasattr(config, "OLLAMA_BASE_URL")
    assert hasattr(config, "TOKEN_BUDGET_PER_SESSION")
    ok("config.py — 字段完整")
    results["pass"] += 1
except Exception as e:
    fail(f"config.py: {e}")
    results["fail"] += 1

try:
    from observability.cost_tracker import cost_tracker
    cost_tracker.record("test_agent", "qwen2.5", 500)
    assert cost_tracker.session_tokens == 500
    assert cost_tracker.usage_percent > 0
    cost_tracker.reset_session()
    ok("observability.cost_tracker — 逻辑正确")
    results["pass"] += 1
except Exception as e:
    fail(f"observability.cost_tracker: {e}")
    results["fail"] += 1

try:
    from observability.tracer import timer
    import time
    with timer() as t:
        time.sleep(0.01)
    assert t.ms >= 8
    ok("observability.tracer.timer — 计时正常")
    results["pass"] += 1
except Exception as e:
    fail(f"observability.tracer: {e}")
    results["fail"] += 1

try:
    from orchestrator.state import AgentState
    # TypedDict 只是类型注解，能导入即可
    ok("orchestrator.state.AgentState — 导入正常")
    results["pass"] += 1
except Exception as e:
    fail(f"orchestrator.state: {e}")
    results["fail"] += 1

try:
    from orchestrator.router import _rule_based_intent
    assert _rule_based_intent("帮我写一段 PINN 代码") == "code"
    assert _rule_based_intent("综述最新 PINN 进展") == "survey"
    assert _rule_based_intent("PINN 的损失函数是什么") == "qa"
    ok("orchestrator.router — 规则路由逻辑正确")
    results["pass"] += 1
except Exception as e:
    fail(f"orchestrator.router: {e}")
    results["fail"] += 1

try:
    from orchestrator.graph import build_graph
    ok("orchestrator.graph.build_graph — 可导入")
    results["pass"] += 1
except Exception as e:
    fail(f"orchestrator.graph: {e}")
    results["fail"] += 1

# ══════════════════════════════════════════════
section("结果汇总")
# ══════════════════════════════════════════════
total = results["pass"] + results["fail"]
print(f"\n  通过: {results['pass']} / {total}")
if results["fail"] > 0:
    print(f"  失败: {results['fail']} 项 — 请对照上方 [FAIL] 条目安装对应依赖")
    sys.exit(1)
else:
    print("  全部通过！可以开始 Phase 1 开发。")
