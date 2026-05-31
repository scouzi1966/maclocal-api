#!/usr/bin/env python3
"""Consolidate the comprehensive probe (*-full.json) into wide + long CSVs and a printed table.
Writes into $BENCH_OUT_DIR (default ../results):
  comprehensive.csv          : one row per engine, all headline metrics
  comprehensive-prefill.csv  : engine,prompt_tokens,prefill_tps   (for the prefill-scaling plot)
  comprehensive-depth.csv    : engine,depth,prompt_tokens,decode_tps,ttft_s  (for the depth plots)
"""
import os, json, glob, csv

OUTDIR = os.path.abspath(os.environ.get("BENCH_OUT_DIR") or os.path.join(os.path.dirname(__file__), "..", "results"))
NAME = {"afm":"afm","rapidmlx":"rapid-mlx","mlx_vlm":"mlx_vlm","omlx":"omlx",
        "lmstudio":"lmstudio","ollama":"ollama","llamacpp":"llama.cpp"}

def depth_of(d, depth):
    for x in d.get("depth_sweep", []):
        if x.get("depth") == depth: return x
    return {}

wide, prefill_long, depth_long = [], [], []
for f in sorted(glob.glob(os.path.join(OUTDIR, "*-full.json"))):
    stem = os.path.basename(f)[:-len("-full.json")]
    eng = NAME.get(stem, stem)
    d = json.load(open(f))
    L = d.get("latency", {}); C = d.get("cache", {})
    ttft = L.get("ttft_ms", {}); itl = L.get("itl_ms", {})
    ps = {p["target"]: p for p in d.get("prefill_scaling", [])}
    g = lambda depth, k: depth_of(d, depth).get(k)
    wide.append({
        "engine": eng,
        "decode_tps": L.get("decode_tps"),
        "ttft_mean_ms": ttft.get("mean"), "ttft_p95_ms": ttft.get("p95"),
        "tpot_ms": L.get("tpot_ms"),
        "itl_p50_ms": itl.get("p50"), "itl_p95_ms": itl.get("p95"), "itl_p99_ms": itl.get("p99"),
        "e2e_256tok_s": L.get("e2e_s"),
        "prefill_512": (ps.get(512) or {}).get("prefill_tps"),
        "prefill_2048": (ps.get(2048) or {}).get("prefill_tps"),
        "prefill_8192": (ps.get(8192) or {}).get("prefill_tps"),
        "cache_speedup": C.get("speedup"),
        "decode_d0": g(0,"decode_tps"), "decode_d4k": g(4096,"decode_tps"),
        "decode_d8k": g(8192,"decode_tps"), "decode_d16k": g(16384,"decode_tps"),
        "ttft_d16k_s": g(16384,"ttft_s"),
    })
    for p in d.get("prefill_scaling", []):
        if p.get("prefill_tps"): prefill_long.append([eng, p.get("prompt_tokens"), p["prefill_tps"]])
    for x in d.get("depth_sweep", []):
        if x.get("decode_tps") is not None:
            depth_long.append([eng, x.get("depth"), x.get("prompt_tokens"), x.get("decode_tps"), x.get("ttft_s")])

wide.sort(key=lambda r: (r["decode_tps"] is None, -(r["decode_tps"] or 0)))
cols = list(wide[0].keys()) if wide else []
with open(os.path.join(OUTDIR,"comprehensive.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(wide)
with open(os.path.join(OUTDIR,"comprehensive-prefill.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["engine","prompt_tokens","prefill_tps"]); w.writerows(prefill_long)
with open(os.path.join(OUTDIR,"comprehensive-depth.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["engine","depth","prompt_tokens","decode_tps","ttft_s"]); w.writerows(depth_long)

# printed table
def fmt(v): return "—" if v is None else (f"{v:g}" if isinstance(v,(int,float)) else str(v))
hdr = ["engine","decode","TTFTms","ITLp95","TPOT","e2e_s","pf512","pf2k","pf8k","cache_x","d0","d4k","d8k","d16k"]
keys= ["engine","decode_tps","ttft_mean_ms","itl_p95_ms","tpot_ms","e2e_256tok_s","prefill_512","prefill_2048","prefill_8192","cache_speedup","decode_d0","decode_d4k","decode_d8k","decode_d16k"]
print(f"wrote comprehensive.csv ({len(wide)} engines), -prefill.csv, -depth.csv in {OUTDIR}\n")
print(" | ".join(f"{h:>8}" for h in hdr))
for r in wide: print(" | ".join(f"{fmt(r.get(k)):>8}" for k in keys))
