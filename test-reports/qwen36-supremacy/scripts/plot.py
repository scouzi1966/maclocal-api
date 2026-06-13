#!/usr/bin/env python3
"""Render visuals from the aggregated dataset + timeseries CSVs into ../results/plots/.
Requires: matplotlib, pandas  (uvx --from matplotlib --with pandas python plot.py  also works)

Outputs:
  decode_bar.png        : decode tok/s per engine (afm highlighted), error bars
  decode_vs_prefill.png : grouped bars, decode + cold prefill per engine
  prefill_cache.png     : afm cold vs prefix-cached prefill (the 8x)
  decode_timeseries.png : instantaneous decode tok/s over time, one line per engine
"""
import os, csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

OUTDIR = os.path.abspath(os.environ.get("BENCH_OUT_DIR") or os.path.join(os.path.dirname(__file__), "..", "results"))
PLOTS = os.path.join(OUTDIR, "plots"); os.makedirs(PLOTS, exist_ok=True)
AFM = "#d6336c"; OTHER = "#4c6ef5"; GREY = "#adb5bd"

def read_csv(name):
    p = os.path.join(OUTDIR, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

data = json.load(open(os.path.join(OUTDIR, "benchmark-data.json")))
eng = [e for e in data["engines"] if e.get("decode_tps") is not None]
eng.sort(key=lambda e: -e["decode_tps"])
names = [e["engine"] for e in eng]
colors = [AFM if n == "afm" else OTHER for n in names]

# 1) decode bar
fig, ax = plt.subplots(figsize=(8, 4.5))
vals = [e["decode_tps"] for e in eng]
ax.bar(names, vals, color=colors)
for i, v in enumerate(vals): ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("decode tok/s"); ax.set_title("Decode throughput (higher is better)")
ax.margins(y=0.18); pc.star_bar(ax, vals); plt.tight_layout(); pc.caption(fig)
plt.savefig(f"{PLOTS}/decode_bar.png", dpi=150); plt.close()

# 2) decode vs cold prefill grouped
fig, ax = plt.subplots(figsize=(9, 4.5))
import numpy as np
x = np.arange(len(names)); w = 0.38
dec = [e["decode_tps"] for e in eng]
pre = [e.get("prefill_cold_tps") or 0 for e in eng]
ax.bar(x - w/2, dec, w, label="decode tok/s", color=colors)
ax.bar(x + w/2, pre, w, label="cold prefill tok/s", color=GREY)
ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()
ax.set_title("Decode vs cold prefill"); pc.star_bar(ax, dec); plt.tight_layout(); pc.caption(fig)
plt.savefig(f"{PLOTS}/decode_vs_prefill.png", dpi=150); plt.close()

# 3) afm prefix-cache 8x
afm = next((e for e in data["engines"] if e["engine"] == "afm"), None)
if afm and afm.get("prefill_cached_tps"):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    vals = [afm["prefill_cold_tps"], afm["prefill_cached_tps"]]
    mult = afm["prefill_cached_tps"] / afm["prefill_cold_tps"] if afm["prefill_cold_tps"] else 0
    ax.bar(["cold prefill", f"cached prefill\n({afm.get('prefill_cache_hit_pct')}% hit)"], vals, color=[GREY, AFM])
    ax.set_ylabel("prefill tok/s")
    ax.set_title(f"afm prefix cache: {mult:.1f}× faster context reload", pad=12)
    ax.set_ylim(0, max(vals) * 1.18)
    for i, v in enumerate(vals): ax.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.annotate("★", (1, vals[1]), textcoords="offset points", xytext=(28, -4), fontsize=15, color="#f1a700")
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/prefill_cache.png", dpi=150); plt.close()

# 4) decode timeseries (series 0, depth 0) one line per engine
ts = read_csv("timeseries-long.csv")
if ts:
    fig, ax = plt.subplots(figsize=(10, 5))
    byeng = {}
    for r in ts:
        if int(r["depth"]) == 0 and int(r["series"]) == 0:
            byeng.setdefault(r["engine"], []).append((float(r["t_seconds"]), float(r["tps"])))
    ymax = 0
    for name in sorted(byeng, key=lambda n: -sorted(p[1] for p in byeng[n])[len(byeng[n])//2]):
        pts = sorted(byeng[name])
        # drop degenerate end-window spikes: cap at 2.5x the engine's median instantaneous rate
        med = sorted(p[1] for p in pts)[len(pts)//2]
        pts = [(t, v) for t, v in pts if v <= 2.5 * med]
        ymax = max(ymax, max(v for _, v in pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=name, lw=1.8, alpha=0.9, color=AFM if name == "afm" else None)
    ax.set_xlabel("seconds"); ax.set_ylabel("decode tok/s (scaled to verified mean)")
    ax.set_ylim(0, ymax * 1.1)
    ax.set_title("Decode throughput over time — Qwen3.6-27B, concurrency=1\n(curves scaled so mean = verified decode; LM Studio excluded: llama-benchy warmup incompatible)")
    ax.legend(loc="upper right")
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/decode_timeseries.png", dpi=150); plt.close()

print("wrote PNGs to", PLOTS)
for f in sorted(os.listdir(PLOTS)): print("  ", f)
