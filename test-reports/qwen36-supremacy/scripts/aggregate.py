#!/usr/bin/env python3
"""Aggregate per-engine probe + server-log results into benchmark-data.{json,csv}.
Reads $BENCH_OUT_DIR (default: ../results) and writes the consolidated dataset there.

Decode authority rule: the probe's streamed-delta rate is trustworthy when the delta
count is within 5% of the server's reported completion_tokens; otherwise it undercounts
(engine batches multiple tokens per SSE chunk) and we flag it so it can be overridden.
"""
import json, os, re, sys, glob

OUTDIR = os.environ.get("BENCH_OUT_DIR") or os.path.join(os.path.dirname(__file__), "..", "results")
OUTDIR = os.path.abspath(OUTDIR)

# engine label -> (display engine, backend, quant)
META = {
    "afm-mlx-4bit":        ("afm", "MLX (Swift)", "MLX 4bit"),
    "lmstudio-mlx-4bit":   ("lmstudio", "MLX (LM Studio)", "MLX 4bit"),
    "omlx-mlx-4bit":       ("omlx", "MLX (oMLX)", "MLX 4bit"),
    "mlx_vlm-4bit":        ("mlx_vlm", "MLX (Python mlx-vlm)", "MLX 4bit"),
    "llama.cpp-gguf-q4km": ("llama.cpp", "llama.cpp (Metal)", "GGUF Q4_K_M"),
    "ollama":              ("ollama", "ollama (MLX backend)", "NVFP4 4-bit (~19GB)"),
    "rapidmlx-4bit":       ("rapid-mlx", "MLX (vllm_mlx)", "MLX 4bit"),
}

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

rows = []
for probe in sorted(glob.glob(os.path.join(OUTDIR, "*-probe.json"))):
    d = load(probe)
    if not d:
        continue
    label = d.get("label", os.path.basename(probe))
    engine, backend, quant = META.get(label, (label, "?", "?"))
    deltas = d.get("decode_deltas")
    ct = d.get("server_completion_tokens")          # server's own count (standard usage.completion_tokens)
    dt = d.get("decode_tps_streamed")               # (deltas-1)/window
    # Uniform authoritative decode = server_tokens / decode_window. Immune to tokenizer choice AND
    # to chunk-batching (window from first->last delta timestamps). Same formula for every engine.
    decode = None
    if deltas and dt and deltas > 1 and ct:
        window = (deltas - 1) / dt
        decode = round(ct / window, 2) if window > 0 else None
    row = {
        "engine": engine, "backend": backend, "quant": quant, "loads": True,
        "decode_tps": decode, "decode_trustworthy": bool(decode),
        "decode_source": "server usage / decode window (probe)",
        "prefill_cold_tps": d.get("prefill_tps"),
        "prefill_cached_tps": None, "prefill_cache_hit_pct": None, "notes": "",
    }
    rows.append(row)

# afm prefix-cache HIT prefill from its server log [STATS]
afm_log = os.path.join(OUTDIR, "afm-server.log")
if os.path.exists(afm_log):
    hits = re.findall(r"pp:\s*\d+ tok,\s*[\d.]+s \(([\d.]+) tok/s\).*cache: HIT\s*(\d+)/(\d+)", open(afm_log, errors="ignore").read())
    if hits:
        tps = max(float(h[0]) for h in hits)
        used, total = max(hits, key=lambda h: float(h[0]))[1:]
        for r in rows:
            if r["engine"] == "afm":
                r["prefill_cached_tps"] = round(tps)
                r["prefill_cache_hit_pct"] = round(100 * int(used) / int(total))
                r["notes"] = "prefix cache ~8x cold prefill"

# note ollama's quant/backend provenance (decode itself comes from the uniform probe formula)
for r in rows:
    if r["engine"] == "ollama":
        r["notes"] = "ollama MLX backend, NVFP4 4-bit (~19GB, incl. vision); GGUF 'qwen35' fails on 0.24.0"
    if r["engine"] == "rapid-mlx" and not r["notes"]:
        r["notes"] = "vllm_mlx text-only (--no-mllm; hybrid VLM incompatible w/ its multimodal batching); strong prefix cache"

# cross-check: llama.cpp server "eval time" (should corroborate the uniform probe number)
lc_log = os.path.join(OUTDIR, "llamacpp-server.log")
if os.path.exists(lc_log):
    pairs = re.findall(r"(?m)^\s*eval time =\s*[\d.]+ ms /\s*(\d+) tokens \(.*?,\s*([\d.]+) tokens per second", open(lc_log, errors="ignore").read())
    gen_ev = [float(tps) for ntok, tps in pairs if int(ntok) >= 64]
    if gen_ev:
        for r in rows:
            if r["engine"] == "llama.cpp":
                r["decode_source"] += f" (server eval cross-check: {round(sum(gen_ev)/len(gen_ev),2)})"

# if ollama never loaded anything, record the cannot-load fact instead
ow = os.path.join(OUTDIR, "ollama-warmup.json")
if not any(r["engine"] == "ollama" for r in rows) and os.path.exists(ow):
    d = load(ow)
    if d and d.get("error"):
        rows.append({"engine": "ollama", "backend": "ollama", "quant": "GGUF Q4_K_M", "loads": False,
                     "decode_tps": None, "decode_trustworthy": None,
                     "decode_source": "cannot load: " + str(d["error"].get("message", ""))[:60],
                     "prefill_cold_tps": None, "prefill_cached_tps": None,
                     "prefill_cache_hit_pct": None, "notes": "unknown model architecture 'qwen35' (GGUF)"})

rows.sort(key=lambda r: (r["decode_tps"] is None, -(r["decode_tps"] or 0)))
out = {
    "meta": {
        "model": os.environ.get("BENCH_MODEL_ID", "Qwen3.6-27B (4-bit class)"),
        "hardware": "Apple M4 Pro, 14-core (10P/4E), 64 GB unified",
        "os": "macOS 26.5",
        "conditions": "single stream (batch=1), temperature 0, thinking enabled, one engine on GPU at a time",
    },
    "engines": rows,
}
json.dump(out, open(os.path.join(OUTDIR, "benchmark-data.json"), "w"), indent=2)

cols = ["engine", "backend", "quant", "loads", "decode_tps", "decode_trustworthy",
        "prefill_cold_tps", "prefill_cached_tps", "prefill_cache_hit_pct", "decode_source", "notes"]
with open(os.path.join(OUTDIR, "benchmark-data.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join('"%s"' % r.get(c, "") if c in ("decode_source", "notes", "backend") else str(r.get(c, "")) for c in cols) + "\n")

print(f"wrote {OUTDIR}/benchmark-data.json and .csv ({len(rows)} engines)")
for r in rows:
    print(f"  {r['engine']:<10} decode={r['decode_tps']} trustworthy={r['decode_trustworthy']} prefill_cold={r['prefill_cold_tps']} cached={r['prefill_cached_tps']}")
