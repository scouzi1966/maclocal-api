#!/usr/bin/env python3
"""EAGLE3 speculative-decode comparison on dense Gemma4-31B — afm vs mlx-vlm.
The two MLX engines that implement EAGLE3. Grouped bars: autoregressive vs EAGLE3 per engine,
with a verify-fidelity badge: afm verifies against the FULL vocab (lossless / bit-exact greedy),
mlx-vlm verifies against a ~32k hot-vocab subset (approximate, ~4% faster but can diverge from
greedy). Decode-only tok/s (matched-thermal medians, 200 tok, M4 Pro).
Output: results-latest/eagle3-decode.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("BENCH_OUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "results-latest"))
OUT = os.path.abspath(OUT)

ENGINES = ["afm", "mlx-vlm 0.6.2"]
AR      = [13.16, 13.18]
EAGLE3  = [17.13, 17.88]          # matched-thermal decode-only medians
VERIFY  = ["✓ lossless\n(full-vocab verify ·\nbit-exact greedy)",
           "≈ approximate\n(hot-vocab verify)"]

BG, GRID, TXT, MUTED = "#0e1117", "#2a2f3a", "#e6e8eb", "#9aa0a6"
C_AR = "#5b616e"
C_E3 = ["#4f8cff", "#36c2a6"]     # afm blue (lossless), mlx-vlm teal (approx)

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": BG, "axes.facecolor": BG})
fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=170)
fig.subplots_adjust(left=0.095, right=0.965, top=0.74, bottom=0.135)

x = np.arange(len(ENGINES)); w = 0.34
ax.bar(x - w/2, AR, w, color=C_AR, edgecolor=BG, linewidth=1.5, zorder=3)
for i in range(len(ENGINES)):
    ax.bar(x[i] + w/2, EAGLE3[i], w, color=C_E3[i], edgecolor=BG, linewidth=1.5, zorder=3)

ymax = max(EAGLE3) * 1.20
ax.set_ylim(0, ymax)
for i in range(len(ENGINES)):
    ax.text(x[i]-w/2, AR[i]+ymax*0.015, f"{AR[i]:.1f}", ha="center", va="bottom",
            color=TXT, fontsize=13, fontweight="bold")
    ax.text(x[i]+w/2, EAGLE3[i]+ymax*0.015, f"{EAGLE3[i]:.1f}", ha="center", va="bottom",
            color=TXT, fontsize=15, fontweight="bold")
    spd = (EAGLE3[i]/AR[i]-1)*100
    ax.annotate(f"+{spd:.0f}%", (x[i]+w/2, EAGLE3[i]), textcoords="offset points",
                xytext=(0, 26), ha="center", color=C_E3[i], fontsize=14, fontweight="bold")
    # verify-fidelity badge inside the EAGLE3 bar
    ax.text(x[i]+w/2, EAGLE3[i]*0.46, VERIFY[i], ha="center", va="center",
            color="#0e1117", fontsize=10, fontweight="bold", linespacing=1.25)
    ax.text(x[i]-w/2, AR[i]*0.5, "AR", ha="center", va="center",
            color="#cfd3da", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["afm  (MLX / Swift)", "mlx-vlm 0.6.2  (MLX / Python)"],
                   color=TXT, fontsize=14, fontweight="bold")
ax.set_ylabel("decode throughput  (tok/s, decode-only)", color=MUTED, fontsize=11)
ax.tick_params(axis="y", colors=MUTED, labelsize=10); ax.tick_params(axis="x", length=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)

fig.text(0.095, 0.935, "EAGLE3 speculative decode  ·  dense Gemma4-31B (4-bit)",
         ha="left", color=TXT, fontsize=20, fontweight="bold")
fig.text(0.095, 0.875,
         "Apple M4 Pro · same verifier + drafter · matched-thermal · the two MLX engines with EAGLE3",
         ha="left", color=MUTED, fontsize=11.5)
fig.text(0.095, 0.815,
         "afm's verify is LOSSLESS (full-vocab argmax = bit-exact greedy AR);  mlx-vlm trades exactness "
         "for ~4% speed via a ~32k hot-vocab verify.",
         ha="left", color="#ffcb2e", fontsize=10.5)

fig.text(0.095, 0.04,
         "afm --eagle3  vs  mlx-vlm --draft-kind eagle3  ·  RedHatAI/gemma-4-31B-it-speculator.eagle3  ·  "
         "greedy · decode-only · 2026-06-07",
         ha="left", color="#6b7280", fontsize=8.5)

path = os.path.join(OUT, "eagle3-decode.png")
fig.savefig(path, facecolor=BG)
print("wrote", path)
