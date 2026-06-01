#!/usr/bin/env python3
"""Plot the SDPA 0.31.3 adaptive-block backport vs the 0.30.3 baseline.
Reads the pipeline JSONs (afm-base303-full.json, afm-sdpa311-full.json) and writes
results/plots/sdpa_backport_depth.png. Style matches plot_common."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plot_common import caption, star_bar  # noqa

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)

def load(tag):
    with open(os.path.join(RESULTS, f"afm-{tag}-full.json")) as f:
        d = json.load(f)
    return {r["depth"]: r["decode_tps"] for r in d["depth_sweep"]}

base = load("base303")
back = load("sdpa311")
depths = sorted(set(base) | set(back))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# Left: decode tok/s vs context depth (line)
ax1.plot(depths, [base[d] for d in depths], "o-", color="#888", lw=2, label="0.30.3 baseline (blocks=32)")
ax1.plot(depths, [back[d] for d in depths], "o-", color="#1f77b4", lw=2, label="0.31.3 backport (adaptive blocks)")
ax1.set_xlabel("context depth (tokens)")
ax1.set_ylabel("decode tok/s")
ax1.set_title("Decode throughput vs context depth")
ax1.set_xticks(depths)
ax1.set_xticklabels([str(d) for d in depths])
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc="lower left")
for d in depths:
    if base.get(d) and back.get(d):
        pct = 100 * (back[d] - base[d]) / base[d]
        if abs(pct) >= 1:
            ax1.annotate(f"{pct:+.0f}%", (d, back[d]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")

# Right: decode@16k bar
vals = [base[16384], back[16384]]
names = ["0.30.3\nbaseline", "0.31.3\nbackport"]
bars = ax2.bar(names, vals, color=["#888", "#1f77b4"], width=0.55)
ax2.set_ylabel("decode tok/s")
ax2.set_title("decode@16k (the contested metric)")
ax2.set_ylim(0, max(vals) * 1.18)
for b, v in zip(bars, vals):
    ax2.annotate(f"{v:.2f}", (b.get_x() + b.get_width()/2, v), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
star_bar(ax2, vals)
gain = 100 * (vals[1] - vals[0]) / vals[0]
ax2.annotate(f"+{gain:.1f}%", (1, vals[1]), textcoords="offset points", xytext=(0, 22),
             ha="center", fontsize=11, color="#1a7f37", fontweight="bold")

fig.suptitle("afm SDPA backport — mlx-swift 0.31.3 adaptive-block 2-pass (decode@16k)", fontsize=12)
caption(fig)
out = os.path.join(PLOTS, "sdpa_backport_depth.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", os.path.abspath(out))
