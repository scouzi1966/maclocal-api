#!/usr/bin/env python3
"""EAGLE3 speculative-decode comparison on dense Gemma4-31B.
Engines that implement EAGLE3: afm (--eagle3) and mlx-vlm (--draft-kind eagle3).
Grouped bars: autoregressive vs EAGLE3 per engine. Decode-only tok/s (excludes prefill),
matching mlx-vlm's "Generation" metric and afm's in-server decode loop.
Output: results-latest/eagle3-decode.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("BENCH_OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "results-latest"))
OUT = os.path.abspath(OUT)

# decode-only tok/s (afm: AR=[STATS] tg, EAGLE3=generateSpeculative loop; mlx-vlm: Generation line)
ENGINES = ["afm", "mlx-vlm"]
AR     = [13.16, 13.18]
EAGLE3 = [17.21, 17.89]
ACCEPT = ["66.7% accept", "68.6% accept"]

BG, GRID, TXT, MUTED = "#0e1117", "#2a2f3a", "#e6e8eb", "#9aa0a6"
C_AR  = "#5b616e"   # autoregressive (muted slate)
C_E3  = ["#4f8cff", "#36c2a6"]  # afm blue, mlx-vlm teal

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": BG, "axes.facecolor": BG})
fig, ax = plt.subplots(figsize=(11, 6.6), dpi=170)
fig.subplots_adjust(left=0.10, right=0.96, top=0.78, bottom=0.16)

import numpy as np
x = np.arange(len(ENGINES)); w = 0.34
b_ar = ax.bar(x - w/2, AR, w, color=C_AR, edgecolor=BG, linewidth=1.5,
              zorder=3, label="autoregressive")
b_e3 = [ax.bar(x[i] + w/2, EAGLE3[i], w, color=C_E3[i], edgecolor=BG, linewidth=1.5, zorder=3)
        for i in range(len(ENGINES))]

ymax = max(EAGLE3) * 1.22
ax.set_ylim(0, ymax)
for i in range(len(ENGINES)):
    ax.text(x[i]-w/2, AR[i]+ymax*0.015, f"{AR[i]:.1f}", ha="center", va="bottom",
            color=TXT, fontsize=13, fontweight="bold")
    ax.text(x[i]+w/2, EAGLE3[i]+ymax*0.015, f"{EAGLE3[i]:.1f}", ha="center", va="bottom",
            color=TXT, fontsize=14, fontweight="bold")
    spd = (EAGLE3[i]/AR[i]-1)*100
    ax.annotate(f"+{spd:.0f}%", (x[i]+w/2, EAGLE3[i]), textcoords="offset points",
                xytext=(0, 26), ha="center", color=C_E3[i], fontsize=15, fontweight="bold")
    ax.text(x[i]+w/2, EAGLE3[i]*0.5, ACCEPT[i], ha="center", va="center",
            color="#0e1117", fontsize=9.5, fontweight="bold", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(["afm\n(MLX / Swift)", "mlx-vlm 0.6.2\n(MLX / Python)"],
                   color=TXT, fontsize=14, fontweight="bold")
ax.set_ylabel("decode throughput  (tok/s, decode-only)", color=MUTED, fontsize=11)
ax.tick_params(axis="y", colors=MUTED, labelsize=10)
ax.tick_params(axis="x", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)

# legend chips (top-right, clear of the shortened subtitle)
fig.text(0.96, 0.885, "EAGLE3", ha="right", va="center", color=MUTED, fontsize=10.5)
fig.text(0.878, 0.886, "■", ha="right", va="center", color="#4f8cff", fontsize=13)
fig.text(0.864, 0.885, "autoregressive", ha="right", va="center", color=MUTED, fontsize=10.5)
fig.text(0.752, 0.886, "■", ha="right", va="center", color=C_AR, fontsize=13)

fig.text(0.10, 0.94, "EAGLE3 speculative decode  ·  dense Gemma4-31B (4-bit)",
         ha="left", color=TXT, fontsize=20, fontweight="bold")
fig.text(0.10, 0.882, "Apple M4 Pro (64 GB)  ·  greedy  ·  same verifier + drafter",
         ha="left", color=MUTED, fontsize=11.5)
fig.text(0.10, 0.04,
         "afm --eagle3   vs   mlx-vlm --draft-kind eagle3   ·   RedHatAI/gemma-4-31B-it-speculator.eagle3 drafter  ·  "
         "greedy · decode-only · 2026-06-07",
         ha="left", color="#6b7280", fontsize=8.5)

path = os.path.join(OUT, "eagle3-decode.png")
fig.savefig(path, facecolor=BG)
print("wrote", path)
