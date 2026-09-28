"""
Multi-Agent PINN — Horizontal PPT-style flow diagram.
Left-to-right layout, modern flat design with icon badges.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Microsoft YaHei", "Arial",
                         "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.4,
})

BG      = "#F8FAFC"
WHITE   = "#FFFFFF"
TEXT    = "#1E293B"
TEXT2   = "#475569"
TEXT3   = "#94A3B8"
SHADOW  = "#CBD5E1"

BLUE    = "#3B82F6";  BLUE_L  = "#EFF6FF";  BLUE_D  = "#1D4ED8"
ORANGE  = "#F59E0B";  ORANGE_L= "#FFFBEB";  ORANGE_D= "#B45309"
GREEN   = "#10B981";  GREEN_L = "#ECFDF5";  GREEN_D = "#047857"
RED     = "#EF4444";  RED_L   = "#FEF2F2"
VIOLET  = "#8B5CF6";  VIOLET_L= "#F5F3FF";  VIOLET_D= "#6D28D9"
INDIGO  = "#6366F1";  INDIGO_L= "#EEF2FF"
AMBER_L = "#FEF3C7";  AMBER_D = "#92400E"
CYAN    = "#06B6D4"
ROSE    = "#F43F5E";  ROSE_L  = "#FFF1F2"
TEAL    = "#14B8A6"
SLATE   = "#64748B"


def card(ax, x, y, w, h, fc=WHITE, ec="#E2E8F0", lw=1.2,
         r=0.12, z=2, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x+0.03, y-0.03), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=SHADOW, ec="none", lw=0, zorder=z-1, alpha=0.3))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z))


def icon(ax, x, y, r=0.28, fc=BLUE, char="?", fs=13, z=5):
    ax.add_patch(Circle((x, y), r+0.05, fc=fc, ec="none",
                         zorder=z-1, alpha=0.15))
    ax.add_patch(Circle((x, y), r, fc=fc, ec="none", zorder=z))
    ax.text(x, y, char, ha="center", va="center", fontsize=fs,
            color=WHITE, fontweight="bold", zorder=z+1)


def arr(ax, s, e, c=SLATE, lw=2.0, cs="arc3,rad=0", ls="-", ms=14, z=1):
    ax.add_patch(FancyArrowPatch(
        s, e, arrowstyle="-|>", color=c, lw=lw,
        connectionstyle=cs, zorder=z, linestyle=ls, mutation_scale=ms))


def tag(ax, x, y, t, fc=BLUE_L, tc=BLUE, fs=7.5, z=6):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold", zorder=z,
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec=tc,
                      lw=0.5, alpha=0.95))


def T(ax, x, y, s, fs=11, c=TEXT, fw="bold", ha="center", z=4):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=fw,
            ha=ha, va="center", zorder=z)


def S(ax, x, y, s, fs=8, c=TEXT2, ha="center", z=4):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va="center", zorder=z)


# ── Canvas (wide) ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(-0.5, 22)
ax.set_ylim(-1.5, 9.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)

cy = 4.5  # center y for main flow

# ══════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════
T(ax, 11, 9.0, "Multi-Agent PINN Research Assistant", fs=18, c=TEXT)
S(ax, 11, 8.5, "Full-Pipeline Workflow:  User Query → Researcher → Coder → Examiner → Answer",
  fs=10.5, c=TEXT2)
ax.plot([3, 19], [8.15, 8.15], color=BLUE, lw=1.5, alpha=0.25, zorder=0)

# ══════════════════════════════════════════════════════════════════════
#  NODE: USER QUERY (left)
# ══════════════════════════════════════════════════════════════════════
qx = 0.3
card(ax, qx, cy-0.6, 2.4, 1.2, fc=BLUE_L, ec=BLUE, lw=1.5)
icon(ax, qx+0.45, cy+0.25, r=0.25, fc=BLUE, char="Q", fs=12)
T(ax, qx+1.5, cy+0.25, "User", fs=11, c=BLUE_D)
T(ax, qx+1.5, cy-0.1, "Query", fs=11, c=BLUE_D)

# ══════════════════════════════════════════════════════════════════════
#  NODE: INTENT ROUTER
# ══════════════════════════════════════════════════════════════════════
rx = 3.5
card(ax, rx, cy-0.6, 2.4, 1.2, fc=INDIGO_L, ec=INDIGO, lw=1.5)
icon(ax, rx+0.45, cy+0.25, r=0.25, fc=INDIGO, char="R", fs=12)
T(ax, rx+1.5, cy+0.25, "Intent", fs=11, c=INDIGO)
T(ax, rx+1.5, cy-0.1, "Router", fs=11, c=INDIGO)

arr(ax, (qx+2.4, cy), (rx, cy), c=BLUE, lw=2.2)

# ══════════════════════════════════════════════════════════════════════
#  AGENT 1: RESEARCHER (container)
# ══════════════════════════════════════════════════════════════════════
a1x = 6.8
card(ax, a1x, cy-1.8, 3.6, 3.6, fc="#FFFDF5", ec=ORANGE, lw=2.0, r=0.15)

icon(ax, a1x+0.4, cy+1.4, r=0.22, fc=ORANGE, char="1", fs=11)
T(ax, a1x+1.8, cy+1.4, "Researcher Agent", fs=12, c=ORANGE_D)
S(ax, a1x+1.8, cy+1.0, "Literature Search & Analysis", fs=8, c=TEXT2)

# mini steps (vertical inside)
steps1 = [
    ("Search",     TEAL,   "S"),
    ("Analyze",    ORANGE, "A"),
    ("Synthesize", GREEN,  "Y"),
]
for i, (name, sc, ch) in enumerate(steps1):
    sy = cy + 0.3 - i * 0.85
    card(ax, a1x+0.3, sy, 2.9, 0.6, fc=WHITE, ec=sc, lw=0.7,
         r=0.06, shadow=False)
    icon(ax, a1x+0.7, sy+0.3, r=0.16, fc=sc, char=ch, fs=8)
    T(ax, a1x+1.8, sy+0.3, name, fs=9, c=TEXT, fw="normal")

# iterate arrow (loop back)
arr(ax, (a1x+3.2, cy-0.85), (a1x+3.2, cy+0.6),
    c=ORANGE, lw=0.8, ls="--", ms=8)
S(ax, a1x+3.45, cy-0.1, "iter", fs=6, c=ORANGE)

arr(ax, (rx+2.4, cy), (a1x, cy), c=INDIGO, lw=2.2)

# output tag below
tag(ax, a1x+1.8, cy-2.2, "literature_report\ndesign_proposal",
    fc=ORANGE_L, tc=ORANGE_D, fs=7)

# ══════════════════════════════════════════════════════════════════════
#  AGENT 2: CODER (container)
# ══════════════════════════════════════════════════════════════════════
a2x = 11.2
card(ax, a2x, cy-1.8, 3.6, 3.6, fc="#F0F7FF", ec=BLUE, lw=2.0, r=0.15)

icon(ax, a2x+0.4, cy+1.4, r=0.22, fc=BLUE, char="2", fs=11)
T(ax, a2x+1.8, cy+1.4, "Coder Agent", fs=12, c=BLUE_D)
S(ax, a2x+1.8, cy+1.0, "Code Generation & Execution", fs=8, c=TEXT2)

steps2 = [
    ("Generate", BLUE, "G"),
    ("Execute",  ROSE, "E"),
    ("Debug",    CYAN, "D"),
]
for i, (name, sc, ch) in enumerate(steps2):
    sy = cy + 0.3 - i * 0.85
    card(ax, a2x+0.3, sy, 2.9, 0.6, fc=WHITE, ec=sc, lw=0.7,
         r=0.06, shadow=False)
    icon(ax, a2x+0.7, sy+0.3, r=0.16, fc=sc, char=ch, fs=8)
    T(ax, a2x+1.8, sy+0.3, name, fs=9, c=TEXT, fw="normal")

arr(ax, (a2x+3.2, cy-0.85), (a2x+3.2, cy+0.6),
    c=BLUE, lw=0.8, ls="--", ms=8)
S(ax, a2x+3.45, cy-0.1, "iter", fs=6, c=BLUE)

arr(ax, (a1x+3.6, cy), (a2x, cy), c=SLATE, lw=2.2)

# Docker badge (below coder)
card(ax, a2x+0.5, cy-2.6, 2.5, 0.6, fc=ROSE_L, ec=ROSE, lw=0.8, r=0.06)
icon(ax, a2x+0.9, cy-2.3, r=0.16, fc=ROSE, char="D", fs=8)
T(ax, a2x+1.9, cy-2.3, "Docker Sandbox", fs=8, c=ROSE)
S(ax, a2x+1.75, cy-2.55, "isolated · no network", fs=6)

# ══════════════════════════════════════════════════════════════════════
#  AGENT 3: EXAMINER (container)
# ══════════════════════════════════════════════════════════════════════
a3x = 15.6
card(ax, a3x, cy-1.8, 3.0, 3.6, fc="#FEFFF5", ec="#CA8A04", lw=2.0, r=0.15)

icon(ax, a3x+0.4, cy+1.4, r=0.22, fc="#CA8A04", char="3", fs=11)
T(ax, a3x+1.5, cy+1.4, "Examiner", fs=12, c=AMBER_D)
S(ax, a3x+1.5, cy+1.0, "Quality Gate", fs=8, c=TEXT2)

# Two review cards
card(ax, a3x+0.2, cy+0.05, 2.5, 0.65, fc=ORANGE_L, ec=ORANGE, lw=0.7,
     r=0.06, shadow=False)
icon(ax, a3x+0.55, cy+0.37, r=0.16, fc=ORANGE, char="A", fs=8)
T(ax, a3x+1.5, cy+0.37, "Academic", fs=9, c=TEXT, fw="normal")

card(ax, a3x+0.2, cy-0.85, 2.5, 0.65, fc=BLUE_L, ec=BLUE, lw=0.7,
     r=0.06, shadow=False)
icon(ax, a3x+0.55, cy-0.53, r=0.16, fc=BLUE, char="C", fs=8)
T(ax, a3x+1.5, cy-0.53, "Code", fs=9, c=TEXT, fw="normal")

arr(ax, (a2x+3.6, cy), (a3x, cy), c=SLATE, lw=2.2)

# ══════════════════════════════════════════════════════════════════════
#  VERDICT + FINAL ANSWER
# ══════════════════════════════════════════════════════════════════════
vx = 19.3
ds = 0.35
diamond = plt.Polygon([
    (vx, cy+ds*1.1), (vx+ds*1.6, cy),
    (vx, cy-ds*1.1), (vx-ds*1.6, cy),
], closed=True, fc=AMBER_L, ec="#CA8A04", lw=1.5, zorder=3)
ax.add_patch(diamond)
T(ax, vx, cy, "?", fs=13, c=AMBER_D, z=5)

arr(ax, (a3x+3.0, cy), (vx-0.56, cy), c="#CA8A04", lw=2.2)

# Final Answer
fx = 20.3
card(ax, fx, cy-0.5, 1.5, 1.0, fc="#D1FAE5", ec=GREEN_D, lw=2.0)
icon(ax, fx+0.35, cy+0.15, r=0.22, fc=GREEN_D, char="F", fs=11)
T(ax, fx+0.75, cy-0.2, "Final", fs=9, c=GREEN_D)
T(ax, fx+0.75, cy-0.45, "Answer", fs=8, c=GREEN_D, fw="normal")

arr(ax, (vx+0.56, cy), (fx, cy), c=GREEN_D, lw=2.5)
tag(ax, vx+0.8, cy+0.45, "PASS", fc=GREEN_L, tc=GREEN_D, fs=7)

# ══════════════════════════════════════════════════════════════════════
#  FAIL RETRY LOOP (below, U-shape back to Coder)
# ══════════════════════════════════════════════════════════════════════
fail_y = cy - 3.2

# down from diamond
arr(ax, (vx, cy-ds*1.1), (vx, fail_y+0.3), c=RED, lw=2.0)
# horizontal left
ax.plot([a2x+1.8, vx], [fail_y, fail_y], color=RED, lw=2.0, zorder=1)
# up into coder
arr(ax, (a2x+1.8, fail_y), (a2x+1.8, cy-1.8), c=RED, lw=2.0)

tag(ax, 16.0, fail_y, "FAIL  ·  retry ≤ 3", fc=RED_L, tc=RED, fs=8)

# ══════════════════════════════════════════════════════════════════════
#  MEMORY (top-left, compact)
# ══════════════════════════════════════════════════════════════════════
card(ax, 0.3, 6.8, 2.4, 1.2, fc=VIOLET_L, ec=VIOLET, lw=1.0)
icon(ax, 0.75, 7.65, r=0.22, fc=VIOLET, char="M", fs=11)
T(ax, 1.7, 7.65, "Memory", fs=10, c=VIOLET_D)
for i, t in enumerate(["Session", "Project", "Experience"]):
    S(ax, 1.5, 7.25 - i*0.22, t, fs=7, c=VIOLET)

# dashed lines to agents
arr(ax, (1.5, 6.8), (a1x+1.8, cy+1.8),
    c=VIOLET, lw=0.8, ls="--", ms=8, cs="arc3,rad=-0.1")
arr(ax, (2.7, 7.0), (a2x+1.8, cy+1.8),
    c=VIOLET, lw=0.8, ls="--", ms=8, cs="arc3,rad=-0.15")

# ══════════════════════════════════════════════════════════════════════
#  LEGEND (bottom)
# ══════════════════════════════════════════════════════════════════════
ly = -1.0
items = [
    (ORANGE_L, ORANGE,  "Researcher"),
    (BLUE_L,   BLUE,    "Coder"),
    (AMBER_L,  "#CA8A04","Examiner"),
    (GREEN_L,  GREEN_D, "Output"),
    (VIOLET_L, VIOLET,  "Memory"),
    (RED_L,    RED,     "Retry"),
]
for i, (fc, ec, name) in enumerate(items):
    bx = 4.0 + i * 2.5
    card(ax, bx, ly, 0.3, 0.22, fc=fc, ec=ec, lw=0.7, r=0.01, shadow=False)
    ax.text(bx+0.4, ly+0.11, name, fontsize=7.5, va="center",
            color=TEXT, zorder=6)

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig("figures/full_pipeline_flow.pdf", format="pdf")
fig.savefig("figures/full_pipeline_flow.png", format="png")
print("Saved OK")
plt.close(fig)
