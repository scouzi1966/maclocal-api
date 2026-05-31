#!/usr/bin/env python3
"""Flatten llama-benchy *-ts.json (json + --save-*-throughput-timeseries) into tidy CSVs
for plotting. Writes into $BENCH_OUT_DIR (default ../results):

  timeseries-long.csv : engine,depth,prompt_size,response_size,series,point,t_seconds,tps
                        (one row per [elapsed_seconds, instantaneous_tok/s] sample)
  scalars.csv         : engine,depth,prompt_size,response_size,pp_tps,tg_tps_mean,tg_tps_std,
                        peak_tps,ttfr_s,e2e_ttft_ms

Plot example (matplotlib):
  df = pd.read_csv('timeseries-long.csv')
  for eng,g in df[df.depth==0].groupby('engine'):
      s = g[g.series==0]; plt.plot(s.t_seconds, s.tps, label=eng)
"""
import json, os, glob, csv

OUTDIR = os.path.abspath(os.environ.get("BENCH_OUT_DIR") or os.path.join(os.path.dirname(__file__), "..", "results"))
LABEL = {  # filename stem -> display engine
    "afm": "afm", "mlx_vlm": "mlx_vlm", "llamacpp": "llama.cpp",
    "lmstudio": "lmstudio", "omlx": "omlx", "ollama": "ollama-mlx",
}

def is_point(x):
    return isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)

def find_series(node, acc):
    """Collect every list-of-[t,tps] found anywhere under node."""
    if isinstance(node, list):
        if node and is_point(node[0]) and all(is_point(p) for p in node):
            acc.append(node)
        else:
            for x in node:
                find_series(x, acc)

# Authoritative decode tok/s per engine (probe / server-anchored) from benchmark-data.json.
# llama-benchy's instantaneous tok/s is inflated for engines without token_ids (afm, oMLX),
# so we scale each engine's curve by (verified_mean / benchy_mean) — preserves SHAPE, fixes SCALE.
verified = {}
try:
    bd = json.load(open(os.path.join(OUTDIR, "benchmark-data.json")))
    verified = {e["engine"]: e.get("decode_tps") for e in bd.get("engines", [])}
except Exception:
    pass

long_rows, scalar_rows = [], []
for path in sorted(glob.glob(os.path.join(OUTDIR, "*-ts.json"))):
    stem = os.path.basename(path)[:-len("-ts.json")]
    engine = LABEL.get(stem, stem)
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"skip {path}: {e}"); continue
    for b in d.get("benchmarks", []):
        depth, ps, rs = b.get("context_size"), b.get("prompt_size"), b.get("response_size")
        def m(k):
            v = b.get(k); return v.get("mean") if isinstance(v, dict) else v
        benchy_tg = m("tg_throughput") or 0
        # benchmark-data.json keys ollama as "ollama" (we display "ollama-mlx")
        vdec = verified.get(engine) or verified.get({"ollama-mlx": "ollama"}.get(engine, engine))
        scale = (vdec / benchy_tg) if (vdec and benchy_tg) else 1.0
        scalar_rows.append([engine, depth, ps, rs,
                            round(m("pp_throughput") or 0, 1),
                            round(vdec if vdec else benchy_tg, 2),     # verified decode
                            round((b.get("tg_throughput") or {}).get("std", 0) or 0, 2),
                            round((m("peak_throughput") or 0) * scale, 1),  # peak, scaled to verified
                            round((m("ttfr") or 0), 3),
                            round((m("e2e_ttft") or 0), 1),
                            round(scale, 4)])
        series = []
        find_series(b.get("throughput_over_time"), series)
        for si, ser in enumerate(series):
            for pi, (t, tps) in enumerate(ser):
                long_rows.append([engine, depth, ps, rs, si, pi, round(t, 4),
                                  round(tps * scale, 3), round(tps, 3), round(scale, 4)])

with open(os.path.join(OUTDIR, "timeseries-long.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["engine","depth","prompt_size","response_size","series","point","t_seconds","tps","tps_raw_benchy","scale"]); w.writerows(long_rows)
with open(os.path.join(OUTDIR, "scalars.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["engine","depth","prompt_size","response_size","pp_tps","tg_tps_verified","tg_tps_std","peak_tps_scaled","ttfr_s","e2e_ttft_ms","benchy_scale"]); w.writerows(scalar_rows)

print(f"wrote timeseries-long.csv ({len(long_rows)} points) and scalars.csv ({len(scalar_rows)} cells) in {OUTDIR}")
print("  (timeseries tps = llama-benchy instantaneous x scale, so its mean matches the verified decode; tps_raw_benchy kept)")
for r in scalar_rows:
    print(f"  {r[0]:<12} verified_tg={r[5]} tok/s  scale={r[10]}  ttft={r[9]}ms")
