#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# PINN Agent V2 — 分阶段依赖安装脚本
#
# 用法: bash install.sh
# 说明: 按功能分组安装，便于定位失败点；
#       Step 6 会自动检查 torch，缺失时补装 CPU 版。
# ─────────────────────────────────────────────────────────────

set -e   # 任意步骤失败立即停止

echo "========================================"
echo " PINN Agent V2 — 依赖安装"
echo "========================================"

# ── Step 1: 编排框架（最关键）───────────────────────────────
echo ""
echo "[1/6] 安装 LangGraph / LangChain..."
pip install \
    "langgraph>=0.2.0" \
    "langchain>=0.3.0" \
    "langchain-community>=0.3.0" \
    "langchain-openai>=0.2.0" \
    "openai>=1.30.0"

# ── Step 2: RAG 相关 ─────────────────────────────────────────
echo ""
echo "[2/6] 安装 RAG 相关（chromadb / sentence-transformers / pypdf）..."
pip install \
    "chromadb>=0.5.0" \
    "sentence-transformers>=3.0.0" \
    "pypdf>=4.0.0"

# ── Step 3: TUI ──────────────────────────────────────────────
echo ""
echo "[3/6] 安装 TUI（textual / rich）..."
pip install \
    "textual>=0.70.0" \
    "rich>=13.7.0"

# ── Step 4: 工具层 ───────────────────────────────────────────
echo ""
echo "[4/6] 安装工具层（arxiv / duckduckgo-search / sympy / docker）..."
pip install \
    "arxiv>=2.1.0" \
    "duckduckgo-search>=6.0.0" \
    "sympy>=1.12" \
    "docker>=7.0.0"

# ── Step 5: 可观测性 + 工具库 ────────────────────────────────
echo ""
echo "[5/6] 安装工具库（pydantic / tiktoken / httpx / tenacity / langsmith）..."
pip install \
    "pydantic>=2.6.0" \
    "python-dotenv>=1.0.0" \
    "tiktoken>=0.7.0" \
    "httpx>=0.27.0" \
    "tenacity>=8.3.0" \
    "langsmith>=0.1.0"

# ── Step 6: 科学计算（自动补 torch）────────────────────────
echo ""
echo "[6/6] 检查科学计算库（numpy/matplotlib/scipy/torch）..."
pip install \
    "numpy>=1.26.0" \
    "matplotlib>=3.8.0" \
    "scipy>=1.12.0"

if python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
then
    echo "  [OK] torch 已存在，跳过安装"
else
    echo "  [INFO] 当前环境缺少 torch，开始安装 CPU 版..."
    pip install \
        "torch>=2.2.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "========================================"
echo " 安装完成！运行验证脚本检查结果："
echo "   python verify_install.py"
echo "========================================"
