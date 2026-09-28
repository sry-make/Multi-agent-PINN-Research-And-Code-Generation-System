"""
Multi-Agent PINN Research Assistant — Publication-Quality Architecture Diagram
Generates a high-resolution figure suitable for academic paper submission.
Version 2: Cleaner layout, no cross-figure arrows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ── Color Palette (academic-friendly, colorblind-safe) ────────────────
C = {
    "bg":           "#FAFBFC",
    "node_fill":    "#E8F0FE",
    "agent_fill":   "#FFF3E0",
    "tool_fill":    "#E8F5E9",
    "memory_fill":  "#F3E5F5",
    "sandbox_fill": "#FBE9E7",
    "examiner_fill":"#FFFDE7",
    "edge":         "#455A64",
    "retry_edge":   "#D32F2F",
    "data_edge":    "#1565C0",
    "border":       "#37474F",
    "text":         "#212121",
    "text_light":   "#616161",
    "highlight":    "#1976D2",
}


def rounded_box(ax, xy, w, h, label, sublabel=None,
                fc="#E8F0FE", ec="#37474F", lw=1.0,
                fontsize=9, fontweight="bold", text_color="#212121",
                sublabel_size=7, radius=0.02, zorder=2, alpha=1.0):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, alpha=alpha,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.28, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color=C["text_light"], zorder=zorder + 1, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
    return box


def arrow(ax, start, end, color="#455A64", lw=1.2, style="-|>",
          connectionstyle="arc3,rad=0", zorder=1, linestyle="-"):
    a = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle,
        zorder=zorder, linestyle=linestyle,
        mutation_scale=12,
    )
    ax.add_patch(a)
    return a


def edge_label(ax, pos, text, fontsize=6.5, color="#616161", bg="#FAFBFC"):
    ax.text(pos[0], pos[1], text, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none", alpha=0.9))


# ── Figure Setup ──────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(13, 11))
ax.set_xlim(-0.5, 13.0)
ax.set_ylim(-1.2, 11.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ── Title ─────────────────────────────────────────────────────────────
ax.text(6.25, 10.6, "Multi-Agent PINN Research Assistant",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=C["text"])
ax.text(6.25, 10.2, "LangGraph-Orchestrated SOP Workflow with Three-Tier Memory & Sandboxed Execution",
        ha="center", va="center", fontsize=8.5, color=C["text_light"], style="italic")

# ══════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW (center column, x ≈ 5–8)
# ══════════════════════════════════════════════════════════════════════
cx = 6.25  # center x

# ── User Query (top) ──
rounded_box(ax, (cx - 1.25, 9.4), 2.5, 0.5, "User Query",
            fc="#BBDEFB", ec=C["highlight"], lw=1.4, fontsize=10)

# ── Parse Intent ──
rounded_box(ax, (cx - 1.25, 8.5), 2.5, 0.55, "Parse Intent",
            sublabel="LLM + Regex Fallback",
            fc=C["node_fill"], ec=C["border"])

# ── Memory Read ──
rounded_box(ax, (cx - 1.25, 7.55), 2.5, 0.55, "Memory Read",
            sublabel="Load 3-Tier Context",
            fc=C["memory_fill"], ec="#7B1FA2")

# ── Router (diamond) ──
router_cy = 6.7
ds = 0.35
diamond = plt.Polygon([
    (cx, router_cy + ds),
    (cx + ds * 1.5, router_cy),
    (cx, router_cy - ds),
    (cx - ds * 1.5, router_cy),
], closed=True, fc="#E3F2FD", ec=C["highlight"], lw=1.2, zorder=3)
ax.add_patch(diamond)
ax.text(cx, router_cy, "Route",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color=C["highlight"], zorder=4)

# ── Researcher Agent ──
res_x, res_y = 2.5, 5.2
rounded_box(ax, (res_x, res_y), 3.0, 0.7, "Researcher Agent",
            sublabel="ReAct Loop · max 5 iter",
            fc=C["agent_fill"], ec="#E65100", lw=1.2, fontsize=10)

# ── Coder Agent ──
cod_x, cod_y = 7.5, 5.2
rounded_box(ax, (cod_x, cod_y), 3.0, 0.7, "Coder Agent",
            sublabel="ReAct Loop · max 4 iter",
            fc=C["agent_fill"], ec="#E65100", lw=1.2, fontsize=10)

# ── Examiner Agent ──
exam_x, exam_y = 4.75, 3.5
rounded_box(ax, (exam_x, exam_y), 3.0, 0.7, "Examiner Agent",
            sublabel="Rule Precheck → LLM Review",
            fc=C["examiner_fill"], ec="#F9A825", lw=1.2, fontsize=10)

# ── Verdict diamond ──
vd_cy = 2.5
vd = plt.Polygon([
    (cx, vd_cy + 0.3),
    (cx + 0.45, vd_cy),
    (cx, vd_cy - 0.3),
    (cx - 0.45, vd_cy),
], closed=True, fc="#FFFDE7", ec="#F9A825", lw=1.0, zorder=3)
ax.add_patch(vd)
ax.text(cx, vd_cy, "Pass?",
        ha="center", va="center", fontsize=7.5, fontweight="bold",
        color="#F57F17", zorder=4)

# ── Synthesize ──
rounded_box(ax, (cx - 1.25, 1.3), 2.5, 0.55, "Synthesize",
            sublabel="Merge All Outputs",
            fc=C["node_fill"], ec=C["border"])

# ── Memory Writeback ──
rounded_box(ax, (cx - 1.25, 0.35), 2.5, 0.55, "Memory Writeback",
            sublabel="Session + Experience",
            fc=C["memory_fill"], ec="#7B1FA2")

# ── Final Answer ──
rounded_box(ax, (cx - 1.0, -0.55), 2.0, 0.45, "Final Answer",
            fc="#C8E6C9", ec="#2E7D32", lw=1.2, fontsize=9)

# ══════════════════════════════════════════════════════════════════════
# ARROWS — Main Flow
# ══════════════════════════════════════════════════════════════════════

# User Query → Parse Intent
arrow(ax, (cx, 9.4), (cx, 9.05))
# Parse Intent → Memory Read
arrow(ax, (cx, 8.5), (cx, 8.1))
# Memory Read → Router
arrow(ax, (cx, 7.55), (cx, 7.05))

# Router → Researcher (left)
arrow(ax, (cx - 0.52, router_cy), (5.5, 5.9),
      connectionstyle="arc3,rad=0.15")
edge_label(ax, (4.3, 6.45), "qa / survey /\nfull_pipeline", fontsize=6)

# Router → Coder (right)
arrow(ax, (cx + 0.52, router_cy), (7.5, 5.9),
      connectionstyle="arc3,rad=-0.15")
edge_label(ax, (8.1, 6.45), "code", fontsize=6)

# Researcher → Coder (full_pipeline, horizontal)
arrow(ax, (5.5, 5.55), (7.5, 5.55),
      color=C["data_edge"], lw=1.0, style="-|>")
edge_label(ax, (6.5, 5.75), "full_pipeline", fontsize=6, color=C["highlight"])

# Researcher → Examiner
arrow(ax, (4.0, 5.2), (5.5, 4.2),
      connectionstyle="arc3,rad=0.1")
edge_label(ax, (4.15, 4.65), "qa/survey", fontsize=6)

# Coder → Examiner
arrow(ax, (9.0, 5.2), (7.1, 4.2),
      connectionstyle="arc3,rad=-0.1")

# Examiner → Verdict
arrow(ax, (cx, 3.5), (cx, 2.8))

# Verdict → Synthesize (PASS)
arrow(ax, (cx, 2.2), (cx, 1.85), color="#2E7D32")
edge_label(ax, (cx + 0.6, 2.0), "PASS", fontsize=6.5, color="#2E7D32")

# Verdict → Coder (FAIL retry — right arc)
arrow(ax, (cx + 0.45, vd_cy), (10.0, 5.2),
      color=C["retry_edge"], lw=1.0, style="-|>",
      connectionstyle="arc3,rad=-0.35", linestyle="--")
edge_label(ax, (10.3, 3.6), "FAIL\n(retry ≤ 3)", fontsize=6, color=C["retry_edge"])

# Verdict → Researcher (FAIL retry — left arc)
arrow(ax, (cx - 0.45, vd_cy), (3.0, 5.2),
      color=C["retry_edge"], lw=1.0, style="-|>",
      connectionstyle="arc3,rad=0.35", linestyle="--")
edge_label(ax, (2.2, 3.6), "FAIL\n(retry ≤ 3)", fontsize=6, color=C["retry_edge"])

# Synthesize → Memory Writeback
arrow(ax, (cx, 1.3), (cx, 0.9))

# Memory Writeback → Final Answer
arrow(ax, (cx, 0.35), (cx, -0.1))

# ══════════════════════════════════════════════════════════════════════
# TOOL PANELS
# ══════════════════════════════════════════════════════════════════════

# ── Researcher Tools (left) ──
tr_x, tr_y = 0.0, 4.85
rounded_box(ax, (tr_x, tr_y), 2.2, 1.55, "",
            fc=C["tool_fill"], ec="#2E7D32", lw=0.8, radius=0.03, alpha=0.9)
ax.text(tr_x + 1.1, tr_y + 1.35, "Researcher Tools",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1B5E20")
tools_r = [
    "search_local_papers",
    "search_arxiv",
    "web_search",
    "simplify_formula",
    "latex_to_sympy",
]
for i, t in enumerate(tools_r):
    ax.text(tr_x + 0.12, tr_y + 1.08 - i * 0.2, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# Arrow: Tools ↔ Researcher
arrow(ax, (2.2, 5.55), (2.5, 5.55), color="#2E7D32", lw=0.8)

# ── Coder Tools (right) ──
tc_x, tc_y = 10.8, 5.2
rounded_box(ax, (tc_x, tc_y), 2.0, 1.1, "",
            fc=C["tool_fill"], ec="#2E7D32", lw=0.8, radius=0.03, alpha=0.9)
ax.text(tc_x + 1.0, tc_y + 0.9, "Coder Tools",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1B5E20")
tools_c = ["execute_python", "read_file / write_file", "run_shell (whitelist)"]
for i, t in enumerate(tools_c):
    ax.text(tc_x + 0.12, tc_y + 0.62 - i * 0.2, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# Arrow: Tools ↔ Coder
arrow(ax, (10.8, 5.7), (10.5, 5.7), color="#2E7D32", lw=0.8)

# ══════════════════════════════════════════════════════════════════════
# DOCKER SANDBOX (below coder tools)
# ══════════════════════════════════════════════════════════════════════
sb_x, sb_y = 10.8, 3.85
rounded_box(ax, (sb_x, sb_y), 2.0, 1.1, "",
            fc=C["sandbox_fill"], ec="#BF360C", lw=0.8, radius=0.03)
ax.text(sb_x + 1.0, sb_y + 0.9, "Docker Sandbox",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#BF360C")
sandbox_items = ["network = none", "CPU / Mem limits", "Timeout (60s)"]
for i, t in enumerate(sandbox_items):
    ax.text(sb_x + 0.12, sb_y + 0.6 - i * 0.2, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# Arrow: Sandbox ↔ Coder Tools
arrow(ax, (11.8, 5.2), (11.8, 4.95), color="#BF360C", lw=0.8)

# ══════════════════════════════════════════════════════════════════════
# THREE-TIER MEMORY (left panel, top)
# ══════════════════════════════════════════════════════════════════════
mem_x, mem_y = 0.0, 7.0
rounded_box(ax, (mem_x, mem_y), 2.5, 2.7, "",
            fc="#F3E5F5", ec="#7B1FA2", lw=1.0, radius=0.04, alpha=0.25)
ax.text(mem_x + 1.25, mem_y + 2.45, "Three-Tier Memory",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#4A148C")

# Tier 1
rounded_box(ax, (mem_x + 0.1, mem_y + 1.6), 2.3, 0.6, "Session Memory",
            sublabel="messages · queries · errors",
            fc="#E1BEE7", ec="#7B1FA2", lw=0.6, fontsize=7.5,
            sublabel_size=6, radius=0.015)
# Tier 2
rounded_box(ax, (mem_x + 0.1, mem_y + 0.85), 2.3, 0.6, "Project Memory",
            sublabel="rules · decisions · stack",
            fc="#CE93D8", ec="#7B1FA2", lw=0.6, fontsize=7.5,
            sublabel_size=6, radius=0.015)
# Tier 3
rounded_box(ax, (mem_x + 0.1, mem_y + 0.1), 2.3, 0.6, "Experience Memory",
            sublabel="success/fail · hints · score",
            fc="#BA68C8", ec="#7B1FA2", lw=0.6, fontsize=7.5,
            sublabel_size=6, radius=0.015, text_color="#FFFFFF")

# Arrows: Memory ↔ Memory Read node
arrow(ax, (2.5, 8.3), (5.0, 7.95), color="#7B1FA2", lw=0.9,
      connectionstyle="arc3,rad=0.05", linestyle="--")
edge_label(ax, (3.5, 8.3), "read", fontsize=6, color="#7B1FA2")

# Arrows: Memory ↔ Memory Writeback node
arrow(ax, (5.0, 0.5), (2.5, 7.0), color="#7B1FA2", lw=0.9,
      connectionstyle="arc3,rad=0.25", linestyle="--")
edge_label(ax, (2.6, 3.5), "write", fontsize=6, color="#7B1FA2")

# ══════════════════════════════════════════════════════════════════════
# RAG PIPELINE (below researcher tools)
# ══════════════════════════════════════════════════════════════════════
rag_x, rag_y = 0.0, 3.3
rounded_box(ax, (rag_x, rag_y), 2.2, 1.3, "",
            fc="#E0F7FA", ec="#00695C", lw=0.8, radius=0.03)
ax.text(rag_x + 1.1, rag_y + 1.1, "RAG Pipeline",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#004D40")
rag_items = ["ChromaDB (vectors)", "HyDE expansion", "BGE-M3 + Reranker", "Local PINN papers"]
for i, t in enumerate(rag_items):
    ax.text(rag_x + 0.12, rag_y + 0.78 - i * 0.18, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# Arrow: RAG → Researcher Tools (short, vertical)
arrow(ax, (1.1, 4.85), (1.1, 4.6), color="#00695C", lw=0.7)

# ══════════════════════════════════════════════════════════════════════
# LLM BACKEND (top right)
# ══════════════════════════════════════════════════════════════════════
llm_x, llm_y = 10.5, 7.8
rounded_box(ax, (llm_x, llm_y), 2.3, 1.2, "",
            fc="#FFF8E1", ec="#FF8F00", lw=0.8, radius=0.03)
ax.text(llm_x + 1.15, llm_y + 1.0, "LLM Backend",
        ha="center", va="center", fontsize=8, fontweight="bold", color="#E65100")
llm_items = ["Ollama (local, default)", "DashScope / Qwen API", "Per-role model config"]
for i, t in enumerate(llm_items):
    ax.text(llm_x + 0.12, llm_y + 0.68 - i * 0.2, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# Dashed arrow: LLM → Agents area (conceptual)
arrow(ax, (10.5, 8.0), (8.0, 7.0), color="#FF8F00", lw=0.7,
      linestyle=":", connectionstyle="arc3,rad=-0.15")
edge_label(ax, (9.6, 7.7), "LLM calls", fontsize=6, color="#E65100")

# ══════════════════════════════════════════════════════════════════════
# OBSERVABILITY (bottom right)
# ══════════════════════════════════════════════════════════════════════
obs_x, obs_y = 10.8, 2.5
rounded_box(ax, (obs_x, obs_y), 2.0, 1.0, "",
            fc="#ECEFF1", ec="#546E7A", lw=0.7, radius=0.03, alpha=0.85)
ax.text(obs_x + 1.0, obs_y + 0.8, "Observability",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#37474F")
obs_items = ["Tracer (JSONL)", "Cost Tracker", "Token Budget"]
for i, t in enumerate(obs_items):
    ax.text(obs_x + 0.12, obs_y + 0.52 - i * 0.18, f"• {t}",
            fontsize=6, color=C["text"], va="center")

# ══════════════════════════════════════════════════════════════════════
# LEGEND (bottom)
# ══════════════════════════════════════════════════════════════════════
leg_y = -0.85
legend_items = [
    (C["node_fill"],    C["border"],   "Workflow Node"),
    (C["agent_fill"],   "#E65100",     "Agent (ReAct)"),
    (C["tool_fill"],    "#2E7D32",     "Tool Layer"),
    (C["memory_fill"],  "#7B1FA2",     "Memory System"),
    (C["sandbox_fill"], "#BF360C",     "Docker Sandbox"),
]
for i, (fc, ec, label) in enumerate(legend_items):
    bx = 1.5 + i * 2.2
    rounded_box(ax, (bx, leg_y), 0.3, 0.2, "", fc=fc, ec=ec, lw=0.7, radius=0.005)
    ax.text(bx + 0.4, leg_y + 0.1, label, fontsize=6.5, va="center", color=C["text"])

# Retry arrow legend
arrow(ax, (1.5, leg_y - 0.35), (1.8, leg_y - 0.35),
      color=C["retry_edge"], lw=1.0, linestyle="--")
ax.text(1.9, leg_y - 0.35, "Retry on FAIL (max 3)",
        fontsize=6.5, va="center", color=C["text"])

# Data flow arrow legend
arrow(ax, (5.5, leg_y - 0.35), (5.8, leg_y - 0.35),
      color=C["data_edge"], lw=1.0)
ax.text(5.9, leg_y - 0.35, "Data Flow (full_pipeline)",
        fontsize=6.5, va="center", color=C["text"])

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig("figures/architecture_diagram.pdf", format="pdf")
fig.savefig("figures/architecture_diagram.png", format="png")
print("Saved: figures/architecture_diagram.pdf & .png")
plt.close(fig)
