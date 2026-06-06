#!/usr/bin/env python3
"""Social-media decode-throughput chart for the Qwen3.6-27B MLX engine benchmark.
Horizontal bars: every engine's autoregressive decode + the two MTP speculative-decode
results (afm, mlx_vlm). afm bars are brand-highlighted; the winner gets a ★.
Output: results-latest/social-decode.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

OUT = os.environ.get("BENCH_OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "results-latest"))
OUT = os.path.abspath(OUT)

# (label, tok/s, category)  category: afm_mtp | other_mtp | afm | other
DATA = [
    ("afm  ·  MTP",        23.3, "afm_mtp"),
    ("mlx-vlm  ·  MTP",    23.0, "other_mtp"),
    ("LM Studio",          15.74, "other"),
    ("afm",                15.68, "afm"),
    ("rapid-mlx",          15.38, "other"),
    ("oMLX",               15.35, "other"),
    ("mlx-vlm",            15.21, "other"),
    ("ollama",             13.87, "other"),
]
DATA.sort(key=lambda r: r[1])  # ascending -> biggest on top in barh

BG     = "#0e1117"
GRID   = "#2a2f3a"
TXT    = "#e6e8eb"
MUTED  = "#9aa0a6"
COLORS = {
    "afm_mtp":   "#ffcb2e",   # hero gold
    "other_mtp": "#36c2a6",   # teal — also speculative, not afm
    "afm":       "#4f8cff",   # afm brand blue
    "other":     "#5b616e",   # muted slate
}

labels = [d[0] for d in DATA]
vals   = [d[1] for d in DATA]
cats   = [d[2] for d in DATA]
colors = [COLORS[c] for c in cats]
y = range(len(DATA))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG, "axes.facecolor": BG,
})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)  # 2040x1148 ~ 16:9
fig.subplots_adjust(left=0.205, right=0.965, top=0.80, bottom=0.135)

bars = ax.barh(list(y), vals, color=colors, height=0.66,
               edgecolor=BG, linewidth=1.5, zorder=3)

# value labels + speedup annotations
ar_afm = next(v for l, v, c in DATA if c == "afm")
for i, (lab, v, c) in enumerate(DATA):
    ax.text(v + 0.18, i, f"{v:.1f}", va="center", ha="left",
            color=TXT, fontsize=13, fontweight="bold", zorder=4)
    if c == "afm_mtp":
        ax.text(v - 0.25, i, "+47%  with MTP", va="center", ha="right",
                color="#1a1d23", fontsize=11, fontweight="bold", zorder=5)
        ax.annotate("★", (v, i), textcoords="offset points", xytext=(34, 0),
                    va="center", ha="center", fontsize=17, color="#ffcb2e", zorder=6)

ax.set_yticks(list(y))
ax.set_yticklabels(labels, color=TXT, fontsize=13)
# bold the afm tick labels
for t, c in zip(ax.get_yticklabels(), cats):
    if c.startswith("afm"):
        t.set_color("#ffd24a" if c == "afm_mtp" else "#7fb0ff")
        t.set_fontweight("bold")

ax.set_xlim(0, max(vals) * 1.16)
ax.set_xlabel("decode throughput  (tokens / sec, single stream)", color=MUTED, fontsize=11)
ax.tick_params(axis="x", colors=MUTED, labelsize=10)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# titles
fig.text(0.205, 0.945, "Qwen3.6-27B  ·  4-bit  ·  Apple M4 Pro (64 GB)",
         ha="left", color=TXT, fontsize=20, fontweight="bold")
fig.text(0.205, 0.885,
         "MLX local-inference decode speed  ·  single stream",
         ha="left", color=MUTED, fontsize=12.5)

# legend chips (right-aligned on the subtitle row)
chips = [("MTP speculative", "#ffcb2e"), ("autoregressive", "#5b616e")]
x0 = 0.965
for name, col in reversed(chips):
    fig.text(x0, 0.885, name, ha="right", va="center", color=MUTED, fontsize=10)
    x0 -= (0.010 + 0.0058 * len(name))
    fig.text(x0, 0.886, "■", ha="right", va="center", color=col, fontsize=12)
    x0 -= 0.018

fig.text(0.205, 0.045,
         "afm (Apple Foundation Models server, MLX/Swift)  ·  all engines latest, same weights  ·  "
         "probe.py canonical decode  ·  2026-06-06",
         ha="left", color="#6b7280", fontsize=8.5)

path = os.path.join(OUT, "social-decode.png")
fig.savefig(path, facecolor=BG)
print("wrote", path)
