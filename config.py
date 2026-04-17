"""
PINN Research Agent V2 — 统一配置中心
所有模块从此处读取配置，禁止在其他文件中硬编码路径/参数。
"""

import os
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.resolve()


def _load_env_files() -> None:
    """
    自动加载仓库根目录下的 .env / .env.local。

    规则:
    - .env 作为基础配置
    - .env.local 作为本机覆盖配置（若存在则覆盖 .env）
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    dotenv_path = _CONFIG_DIR / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    dotenv_local_path = _CONFIG_DIR / ".env.local"
    if dotenv_local_path.exists():
        load_dotenv(dotenv_path=dotenv_local_path, override=True)


_load_env_files()


def _getenv_first(*keys: str, default: str = "") -> str:
    """返回首个非空环境变量值。"""
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return default


def _getenv_bool(*keys: str, default: bool = False) -> bool:
    """解析布尔环境变量。"""
    raw = _getenv_first(*keys, default="")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_torch_device(preferred: str) -> str:
    """
    根据当前 Torch 环境解析最终设备。

    规则:
    - 显式环境变量 `PINN_AGENT_TORCH_DEVICE` 优先
    - 请求 cuda/mps 但当前不可用时，自动回退到 cpu
    - Torch 不可导入时，保守回退到 cpu
    """
    preferred = os.getenv("PINN_AGENT_TORCH_DEVICE", preferred).strip().lower()
    if preferred == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        return "cpu"

    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if preferred == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        return "mps" if mps_backend and mps_backend.is_available() else "cpu"

    return preferred


def _resolve_hf_model_source(model_name: str) -> str:
    """
    优先返回本地 Hugging Face 缓存快照目录，离线环境下更稳定。

    若本地无缓存，则回退为原始 repo id，交给下游自行下载。
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_files_only=True,
        )
        return str(Path(config_path).parent)
    except Exception:
        return model_name

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
ROOT_DIR = _CONFIG_DIR

CHROMA_DB_PATH   = str(ROOT_DIR / "rag" / "chroma_db")
PAPERS_DIR       = str(ROOT_DIR / "papers")
LOGS_DIR         = str(ROOT_DIR / "logs")
EVAL_DIR         = str(ROOT_DIR / "eval")
OUTPUTS_DIR      = str(ROOT_DIR / "outputs")
MEMORY_DIR       = str(ROOT_DIR / "memory")
SESSION_MEMORY_DIR = str(Path(MEMORY_DIR) / "sessions")
PROJECT_MEMORY_PATH = str(Path(MEMORY_DIR) / "project_memory.json")
EXPERIENCE_DB_PATH  = str(Path(MEMORY_DIR) / "experience_db.jsonl")
SANDBOX_WORKDIR  = str(ROOT_DIR / "sandbox" / "workspace")   # 挂载到容器的工作目录
SANDBOX_ARTIFACTS_DIR = str(Path(OUTPUTS_DIR) / "sandbox_runs")

# ─────────────────────────────────────────────
# LLM Provider
# ─────────────────────────────────────────────
# 兼容两类后端:
# 1. 本地 Ollama（默认）→ 适合离线开发 / 演示
# 2. Qwen / DashScope OpenAI-Compatible API → 适合先用强模型把 Agent 做稳
#
# 切换方式:
#   export PINN_AGENT_LLM_PROVIDER=qwen
#   export DASHSCOPE_API_KEY=sk-xxx
#   export PINN_AGENT_QWEN_MODEL_CODER=qwen3-coder-plus
#
# 说明:
# - 国内百炼常用: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 国际站可改:   https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_PROVIDER = _getenv_first(
    "PINN_AGENT_LLM_PROVIDER",
    "PINN_AGENT_MODEL_PROVIDER",
    default="ollama",
).lower()

_OLLAMA_BASE_URL_LOCAL = _getenv_first(
    "PINN_AGENT_OLLAMA_BASE_URL",
    "OLLAMA_BASE_URL",
    default="http://localhost:11434/v1",
)
_OLLAMA_API_KEY_LOCAL = _getenv_first(
    "PINN_AGENT_OLLAMA_API_KEY",
    "OLLAMA_API_KEY",
    default="ollama-local",
)

QWEN_BASE_URL = _getenv_first(
    "PINN_AGENT_QWEN_BASE_URL",
    "QWEN_BASE_URL",
    "DASHSCOPE_BASE_URL",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_API_KEY = _getenv_first(
    "PINN_AGENT_QWEN_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
)

_LOCAL_MODEL_RESEARCHER = _getenv_first(
    "PINN_AGENT_LOCAL_MODEL_RESEARCHER",
    default="pinn_qwen_expert",
)
_LOCAL_MODEL_CODER = _getenv_first(
    "PINN_AGENT_LOCAL_MODEL_CODER",
    default="qwen2.5:7b",
)
_LOCAL_MODEL_EXAMINER = _getenv_first(
    "PINN_AGENT_LOCAL_MODEL_EXAMINER",
    default="qwen2.5:7b",
)
_LOCAL_MODEL_ROUTER = _getenv_first(
    "PINN_AGENT_LOCAL_MODEL_ROUTER",
    default="qwen2.5:7b",
)

_QWEN_MODEL_RESEARCHER = _getenv_first(
    "PINN_AGENT_QWEN_MODEL_RESEARCHER",
    "QWEN_MODEL_RESEARCHER",
    "PINN_AGENT_QWEN_MODEL",
    "QWEN_MODEL",
    default="qwen-plus",
)
_QWEN_MODEL_CODER = _getenv_first(
    "PINN_AGENT_QWEN_MODEL_CODER",
    "QWEN_MODEL_CODER",
    "PINN_AGENT_QWEN_MODEL",
    "QWEN_MODEL",
    default="qwen3-coder-plus",
)
_QWEN_MODEL_EXAMINER = _getenv_first(
    "PINN_AGENT_QWEN_MODEL_EXAMINER",
    "QWEN_MODEL_EXAMINER",
    "PINN_AGENT_QWEN_MODEL",
    "QWEN_MODEL",
    default="qwen-plus",
)
_QWEN_MODEL_ROUTER = _getenv_first(
    "PINN_AGENT_QWEN_MODEL_ROUTER",
    "QWEN_MODEL_ROUTER",
    "PINN_AGENT_QWEN_MODEL",
    "QWEN_MODEL",
    default="qwen-plus",
)

if LLM_PROVIDER in {"qwen", "dashscope", "bailian"}:
    OLLAMA_BASE_URL = QWEN_BASE_URL
    OLLAMA_API_KEY  = QWEN_API_KEY or "missing-qwen-api-key"
    MODEL_RESEARCHER = _QWEN_MODEL_RESEARCHER
    MODEL_CODER      = _QWEN_MODEL_CODER
    MODEL_EXAMINER   = _QWEN_MODEL_EXAMINER
    MODEL_ROUTER     = _QWEN_MODEL_ROUTER
else:
    OLLAMA_BASE_URL = _OLLAMA_BASE_URL_LOCAL
    OLLAMA_API_KEY  = _OLLAMA_API_KEY_LOCAL
    MODEL_RESEARCHER = _LOCAL_MODEL_RESEARCHER
    MODEL_CODER      = _LOCAL_MODEL_CODER
    MODEL_EXAMINER   = _LOCAL_MODEL_EXAMINER
    MODEL_ROUTER     = _LOCAL_MODEL_ROUTER

LLM_PROVIDER_LABEL = {
    "ollama":    "Ollama",
    "qwen":      "Qwen API",
    "dashscope": "Qwen API",
    "bailian":   "Qwen API",
}.get(LLM_PROVIDER, LLM_PROVIDER)

MODEL_BY_STEP = {
    "parse_intent": MODEL_ROUTER,
    "researcher":   MODEL_RESEARCHER,
    "coder":        MODEL_CODER,
    "examiner":     MODEL_EXAMINER,
    "synthesize":   "(no-llm)",
}

MODEL_SUMMARY = (
    f"R={MODEL_RESEARCHER} | "
    f"C={MODEL_CODER} | "
    f"E={MODEL_EXAMINER} | "
    f"Router={MODEL_ROUTER}"
)


def get_model_for_step(step: str) -> str:
    """返回给定 SOP 步骤对应的当前模型名。"""
    return MODEL_BY_STEP.get(step, MODEL_CODER)


# ─────────────────────────────────────────────
# Eval / LLM-as-Judge
# ─────────────────────────────────────────────
EVAL_JUDGE_ENABLED = _getenv_bool(
    "PINN_AGENT_EVAL_ENABLE_JUDGE",
    default=True,
)
EVAL_JUDGE_MODE = _getenv_first(
    "PINN_AGENT_EVAL_JUDGE_MODE",
    default="auto",
).lower()
EVAL_JUDGE_MODEL = _getenv_first(
    "PINN_AGENT_EVAL_JUDGE_MODEL",
    default=MODEL_EXAMINER,
)
EVAL_JUDGE_TIMEOUT_SEC = int(
    _getenv_first("PINN_AGENT_EVAL_JUDGE_TIMEOUT_SEC", default="20")
)
EVAL_JUDGE_MAX_TOKENS = int(
    _getenv_first("PINN_AGENT_EVAL_JUDGE_MAX_TOKENS", default="800")
)
EVAL_JUDGE_PASS_THRESHOLD = float(
    _getenv_first("PINN_AGENT_EVAL_JUDGE_PASS_THRESHOLD", default="75")
)
EVAL_JUDGE_WEIGHT = float(
    _getenv_first("PINN_AGENT_EVAL_JUDGE_WEIGHT", default="0.30")
)

# 生成参数
LLM_TEMPERATURE  = 0.1
LLM_MAX_TOKENS   = 2048

# Examiner 深度审查时的输入截断阈值（避免误判半截代码）
EXAMINER_REVIEW_MAX_REPORT_CHARS = int(
    _getenv_first("PINN_AGENT_EXAMINER_REPORT_CHARS", default="4000")
)
EXAMINER_REVIEW_MAX_CODE_CHARS = int(
    _getenv_first("PINN_AGENT_EXAMINER_CODE_CHARS", default="12000")
)
EXAMINER_REVIEW_MAX_STDIO_CHARS = int(
    _getenv_first("PINN_AGENT_EXAMINER_STDIO_CHARS", default="1200")
)

# ─────────────────────────────────────────────
# RAG / 嵌入
# ─────────────────────────────────────────────
EMBED_MODEL_NAME      = "BAAI/bge-m3"           # V2 升级：多语言 1024-dim
EMBED_MODEL_SOURCE    = _resolve_hf_model_source(EMBED_MODEL_NAME)
EMBED_DEVICE          = _resolve_torch_device("cuda")
RERANKER_MODEL_NAME   = "BAAI/bge-reranker-base"
RERANKER_MODEL_SOURCE = _resolve_hf_model_source(RERANKER_MODEL_NAME)
RERANKER_DEVICE       = _resolve_torch_device("cuda")

CHROMA_COLLECTION     = "pinn_papers_v2"
RAG_CHUNK_SIZE        = 512
RAG_CHUNK_OVERLAP     = 64
RAG_TOP_K_COARSE      = 20                       # 粗召回数
RAG_TOP_K_FINAL       = 5                        # 精排后保留数

# ─────────────────────────────────────────────
# Token 成本控制
# ─────────────────────────────────────────────
TOKEN_BUDGET_PER_SESSION  = 32_000               # 每次会话最大 Token 消耗
TOKEN_BUDGET_PER_CALL     = 4_096                # 单次 LLM 调用上限
TOKEN_WARN_THRESHOLD      = 0.80                 # 达到 80% 时 TUI 警告

# 超出预算时的降级策略
# "skip_hyde"   → 关闭 HyDE 查询重写，直接检索
# "skip_rerank" → 关闭重排，仅粗召回
# "abort"       → 直接拒绝本次请求
TOKEN_OVERBUDGET_STRATEGY = "skip_hyde"

# ─────────────────────────────────────────────
# 代码沙盒 (Docker)
# ─────────────────────────────────────────────
SANDBOX_IMAGE         = "pinn_agent_sandbox:latest"
SANDBOX_CPU_LIMIT     = "2"                      # 最多 2 核
SANDBOX_MEM_LIMIT     = "2g"                     # 最多 2 GB
SANDBOX_TIMEOUT_SEC   = 30                       # 执行超时（秒）
SANDBOX_NETWORK       = "none"                   # 完全网络隔离

# 白名单 Shell 命令（Coder Agent run_shell 工具可用）
SANDBOX_SHELL_WHITELIST = {
    "ls", "cat", "head", "tail", "pwd",
    "python", "python3", "pip", "pip3",
    "mkdir", "cp", "mv", "echo",
}

# ─────────────────────────────────────────────
# 外部搜索工具
# ─────────────────────────────────────────────
ARXIV_MAX_RESULTS     = 5
ARXIV_SORT_BY         = "relevance"              # "relevance" | "lastUpdatedDate"

# DuckDuckGo / SerpAPI（可选）
SERPAPI_KEY           = ""                       # 留空则降级用 DuckDuckGo
WEB_SEARCH_MAX_RESULTS = 5

# ─────────────────────────────────────────────
# Examiner Agent 审查
# ─────────────────────────────────────────────
EXAMINER_MAX_RETRIES  = 3                        # 最多循环修订次数
EXAMINER_STRICT_MODE  = True                     # True → 引用必须可验证
EXAMINER_LLM_TIMEOUT_SEC = int(
    _getenv_first("PINN_AGENT_EXAMINER_LLM_TIMEOUT_SEC", default="25")
)
EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS = _getenv_bool(
    "PINN_AGENT_EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS",
    default=False,
)

# ─────────────────────────────────────────────
# 可观测性 / Tracing
# ─────────────────────────────────────────────
TRACE_ENABLED         = True
TRACE_LOG_DIR         = LOGS_DIR
LANGSMITH_API_KEY     = ""                       # 留空则只写本地 JSONL
LANGSMITH_PROJECT     = "pinn-agent-v2"

# ─────────────────────────────────────────────
# TUI
# ─────────────────────────────────────────────
TUI_THEME             = "dark"                   # "dark" | "light"
TUI_STREAM_DELAY      = 0.015                    # 打字机效果间隔（秒）
TUI_SHOW_DEBUG        = False                    # 默认隐藏 Debug 面板（按 d 切换）
