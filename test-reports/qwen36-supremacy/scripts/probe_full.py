#!/usr/bin/env python3
"""Comprehensive single-stream performance probe for OpenAI-compatible engines.
All token counts come from the server's standard `usage` (immune to tokenizer mismatch);
all timings from streamed SSE delta timestamps.

Captures three suites:
  1. Latency      : TTFT (mean/p50/p95), decode tok/s, ITL p50/p95/p99 (ms), TPOT, e2e latency
  2. Prefill+cache: prefill tok/s at 512/2048/8192-token prompts; cold-vs-warm cached prefill speedup
  3. Depth sweep  : decode tok/s + TTFT at context depth 0/4096/8192/16384

Usage: probe_full.py BASE_URL MODEL LABEL OUTFILE.json
Progress is printed to stderr; full JSON written to OUTFILE and echoed to stdout.
"""
import sys, json, time, random, statistics as st, urllib.request

BASE, MODEL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CHARS_PER_TOK = 4.45            # calibrated for the filler below on Qwen tokenizer
FILLER = "The quick brown fox jumps over the lazy dog. "   # ~45 chars

def log(m): sys.stderr.write(f"[{LABEL}] {m}\n"); sys.stderr.flush()

def pctl(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    f = int(k); return round(xs[f] + (xs[min(f+1, len(xs)-1)] - xs[f]) * (k - f), 3)

def make_prompt(approx_tokens, unique=True):
    """Build a prompt of ~approx_tokens tokens; unique prefix avoids cross-test prefix-cache hits."""
    pre = (f"[req-{random.randint(0,10**9)}] " if unique else "")
    if approx_tokens <= 8:
        return pre + "Write a detailed paragraph about the Pacific Ocean and marine biology."
    reps = max(1, int(approx_tokens * CHARS_PER_TOK / len(FILLER)))
    return pre + (FILLER * reps) + "\nSummarize the text above in one short sentence."

def stream(messages, max_tokens, timeout=900):
    """Return dict: t0, ttft, last, n_deltas, delta_times[], usage."""
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); times = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            if obj.get("usage"): usage = obj["usage"]
            ch = obj.get("choices") or []
            if not ch: continue
            d = ch[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                times.append(time.time())
    return {"t0": t0, "ttft": (times[0]-t0) if times else None,
            "last": times[-1] if times else None, "n": len(times),
            "delta_times": times, "usage": usage or {}}

def decode_tps(s):
    ct = s["usage"].get("completion_tokens")
    if ct and s["delta_times"] and s["last"] and s["delta_times"][0] and s["last"] > s["delta_times"][0]:
        return round(ct / (s["last"] - s["delta_times"][0]), 2)
    return None

def prefill_tps(s):
    pt = s["usage"].get("prompt_tokens")
    return round(pt / s["ttft"], 1) if (pt and s["ttft"]) else None

report = {"label": LABEL, "base": BASE, "model": MODEL}

# ---------- 1. LATENCY SUITE ----------
log("latency suite: 5x TTFT…")
ttfts = []
for i in range(5):
    s = stream([{"role": "user", "content": f"[{i}] Reply with the single word: ready."}], 8)
    if s["ttft"]: ttfts.append(s["ttft"] * 1000)
log("latency suite: 256-token decode (ITL/TPOT/e2e)…")
d = stream([{"role": "user", "content": make_prompt(8)}], 256)
itls = [(d["delta_times"][i] - d["delta_times"][i-1]) * 1000 for i in range(1, d["n"])]
report["latency"] = {
    "ttft_ms": {"mean": round(st.mean(ttfts),1) if ttfts else None,
                "p50": pctl(ttfts,50), "p95": pctl(ttfts,95), "n": len(ttfts)},
    "decode_tps": decode_tps(d),
    "decode_tokens": d["usage"].get("completion_tokens"),
    "tpot_ms": round((d["last"]-d["delta_times"][0])/(d["n"]-1)*1000, 2) if d["n"] > 1 else None,
    "itl_ms": {"p50": pctl(itls,50), "p95": pctl(itls,95), "p99": pctl(itls,99),
               "mean": round(st.mean(itls),2) if itls else None},
    "e2e_s": round(d["last"]-d["t0"], 2) if d["last"] else None,
}
log(f"  decode={report['latency']['decode_tps']} tok/s  TTFT={report['latency']['ttft_ms']['mean']}ms  ITL p95={report['latency']['itl_ms']['p95']}ms")

# ---------- 2. PREFILL SCALING + CACHE ----------
report["prefill_scaling"] = []
for size in (512, 2048, 8192):
    log(f"prefill scaling: ~{size} tokens…")
    s = stream([{"role": "user", "content": make_prompt(size)}], 1)
    report["prefill_scaling"].append({"target": size,
        "prompt_tokens": s["usage"].get("prompt_tokens"), "ttft_s": round(s["ttft"],3) if s["ttft"] else None,
        "prefill_tps": prefill_tps(s)})
log("cache: cold vs warm (same ~2048-tok prompt twice)…")
cache_prompt = make_prompt(2048)
cold = stream([{"role": "user", "content": cache_prompt}], 1)
warm = stream([{"role": "user", "content": cache_prompt}], 1)
report["cache"] = {
    "prompt_tokens": cold["usage"].get("prompt_tokens"),
    "cold_ttft_s": round(cold["ttft"],3) if cold["ttft"] else None,
    "warm_ttft_s": round(warm["ttft"],3) if warm["ttft"] else None,
    "cold_prefill_tps": prefill_tps(cold), "warm_prefill_tps": prefill_tps(warm),
    "speedup": round(cold["ttft"]/warm["ttft"],1) if (cold["ttft"] and warm["ttft"] and warm["ttft"]>0) else None,
}
log(f"  cache speedup={report['cache']['speedup']}x  (cold {report['cache']['cold_ttft_s']}s -> warm {report['cache']['warm_ttft_s']}s)")

# ---------- 3. CONTEXT-DEPTH SWEEP ----------
report["depth_sweep"] = []
for depth in (0, 4096, 8192, 16384):
    log(f"depth sweep: ~{depth} ctx + 64-token decode…")
    try:
        s = stream([{"role": "user", "content": make_prompt(depth if depth else 8)}], 64)
        report["depth_sweep"].append({"depth": depth, "prompt_tokens": s["usage"].get("prompt_tokens"),
            "ttft_s": round(s["ttft"],3) if s["ttft"] else None, "decode_tps": decode_tps(s),
            "decode_tokens": s["usage"].get("completion_tokens")})
        log(f"  depth {depth}: TTFT={report['depth_sweep'][-1]['ttft_s']}s decode={report['depth_sweep'][-1]['decode_tps']} tok/s")
    except Exception as e:
        report["depth_sweep"].append({"depth": depth, "error": str(e)[:120]})
        log(f"  depth {depth}: FAILED {str(e)[:80]}")

# backward-compat top-level keys (so the simple aggregate.py still works)
report["decode_tps_streamed"] = report["latency"]["decode_tps"]
report["decode_deltas"] = d["n"]; report["server_completion_tokens"] = report["latency"]["decode_tokens"]
report["prefill_tps"] = report["prefill_scaling"][0]["prefill_tps"] if report["prefill_scaling"] else None
report["prefill_prompt_tokens"] = report["prefill_scaling"][0]["prompt_tokens"] if report["prefill_scaling"] else None

json.dump(report, open(OUT, "w"), indent=2)
print(json.dumps(report, indent=2))
log("DONE")
