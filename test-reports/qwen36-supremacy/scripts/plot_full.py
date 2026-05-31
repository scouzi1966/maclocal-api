#!/usr/bin/env python3
"""Comprehensive visuals from comprehensive*.csv into ../results/plots/.
Requires matplotlib (uvx --with matplotlib python plot_full.py)."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

OUTDIR = os.path.abspath(os.environ.get("BENCH_OUT_DIR") or os.path.join(os.path.dirname(__file__), "..", "results"))
PLOTS = os.path.join(OUTDIR, "plots"); os.makedirs(PLOTS, exist_ok=True)
AFM = "#d6336c"
def rd(n):
    p=os.path.join(OUTDIR,n); return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def col(name): return AFM if name=="afm" else None

wide = rd("comprehensive.csv")
order = [r["engine"] for r in wide]  # decode-desc

# 1) TTFT bar (mean)
rows=[r for r in wide if r.get("ttft_mean_ms")]
if rows:
    fig,ax=plt.subplots(figsize=(8,4.5))
    names=[r["engine"] for r in rows]; vals=[float(r["ttft_mean_ms"]) for r in rows]
    ax.bar(names,vals,color=[AFM if n=="afm" else "#4c6ef5" for n in names])
    for i,v in enumerate(vals): ax.text(i,v,f"{v:.0f}",ha="center",va="bottom",fontsize=9)
    ax.set_ylabel("TTFT (ms)"); ax.set_title("Time to first token (short prompt) — lower is better")
    ax.margins(y=0.18); pc.star_bar(ax, vals, lower_is_better=True); plt.tight_layout(); pc.caption(fig)
    plt.savefig(f"{PLOTS}/ttft_bar.png",dpi=150); plt.close()

# 2) ITL distribution (p50/p95/p99 grouped)
rows=[r for r in wide if r.get("itl_p50_ms")]
if rows:
    import numpy as np
    fig,ax=plt.subplots(figsize=(9,4.5))
    names=[r["engine"] for r in rows]; x=np.arange(len(names)); w=0.27
    for j,(k,lab) in enumerate([("itl_p50_ms","p50"),("itl_p95_ms","p95"),("itl_p99_ms","p99")]):
        ax.bar(x+(j-1)*w,[float(r[k]) for r in rows],w,label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("inter-token latency (ms)")
    ax.set_title("Decode inter-token latency distribution — lower & flatter is better"); ax.legend()
    p95s = [float(r["itl_p95_ms"]) for r in rows]
    win = min(range(len(p95s)), key=lambda i: p95s[i])
    ax.annotate("★", (win, p95s[win]), textcoords="offset points", xytext=(0,12), ha="center", fontsize=15, color="#f1a700")
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/itl_distribution.png",dpi=150); plt.close()

# 3) prefill scaling curve
pf=rd("comprehensive-prefill.csv")
if pf:
    fig,ax=plt.subplots(figsize=(8.5,5)); byeng={}
    for r in pf: byeng.setdefault(r["engine"],[]).append((int(r["prompt_tokens"]),float(r["prefill_tps"])))
    win = max(byeng, key=lambda n: max(v for _,v in byeng[n]))  # best prefill overall
    for n in order:
        if n in byeng:
            pts=sorted(byeng[n]); ax.plot([p[0] for p in pts],[p[1] for p in pts],marker="o",lw=2,
                                          label=(n+" ★" if n==win else n),color=col(n))
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("prefill tok/s"); ax.set_xscale("log")
    ax.set_title("Prefill throughput vs prompt size (higher is better)"); ax.legend()
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/prefill_scaling.png",dpi=150); plt.close()

# 4) depth sweep: decode tok/s vs context depth
dp=rd("comprehensive-depth.csv")
if dp:
    fig,ax=plt.subplots(figsize=(8.5,5)); byeng={}
    for r in dp: byeng.setdefault(r["engine"],[]).append((int(r["depth"]),float(r["decode_tps"])))
    win = max(byeng, key=lambda n: sorted(byeng[n])[-1][1])  # highest decode at deepest ctx
    for n in order:
        if n in byeng:
            pts=sorted(byeng[n]); ax.plot([p[0] for p in pts],[p[1] for p in pts],marker="o",lw=2,
                                          label=(n+" ★" if n==win else n),color=col(n))
    ax.set_xlabel("context depth (tokens)"); ax.set_ylabel("decode tok/s")
    ax.set_title("Decode throughput vs context depth — flatter = better long-context (★ = best at deepest)"); ax.legend()
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/depth_decode.png",dpi=150); plt.close()

# 5) depth sweep: TTFT (prefill-at-depth) vs depth
if dp:
    fig,ax=plt.subplots(figsize=(8.5,5)); byeng={}
    for r in dp:
        if r.get("ttft_s"): byeng.setdefault(r["engine"],[]).append((int(r["depth"]),float(r["ttft_s"])))
    win = min(byeng, key=lambda n: sorted(byeng[n])[-1][1])  # lowest TTFT at deepest ctx
    for n in order:
        if n in byeng:
            pts=sorted(byeng[n]); ax.plot([p[0] for p in pts],[p[1] for p in pts],marker="o",lw=2,
                                          label=(n+" ★" if n==win else n),color=col(n))
    ax.set_xlabel("context depth (tokens)"); ax.set_ylabel("TTFT (s) — prefill at depth")
    ax.set_title("Time-to-first-token vs context depth — lower is better (★ = lowest at deepest)"); ax.legend()
    plt.tight_layout(); pc.caption(fig); plt.savefig(f"{PLOTS}/depth_ttft.png",dpi=150); plt.close()

# 6) realistic prefix-cache speedup
cr=rd("cache-realistic.csv")
cr=[r for r in cr if r.get("speedup") not in ("","None",None)]
if cr:
    fig,ax=plt.subplots(figsize=(8.5,5))
    names=[r["engine"] for r in cr]; vals=[float(r["speedup"]) for r in cr]
    ax.bar(names,vals,color=[AFM if n=="afm" else "#4c6ef5" for n in names])
    for i,v in enumerate(vals): ax.text(i,v,f"{v:g}x",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel("prefill speedup (cold TTFT / hit TTFT)")
    ax.set_title("Realistic prefix cache: shared context + NEW question (multi-turn/agentic) — higher is better")
    ax.margins(y=0.18); pc.star_bar(ax, vals); plt.tight_layout(); pc.caption(fig)
    plt.savefig(f"{PLOTS}/cache_realistic.png",dpi=150); plt.close()

print("wrote comprehensive PNGs to", PLOTS)
for f in sorted(os.listdir(PLOTS)): print("  ", f)
