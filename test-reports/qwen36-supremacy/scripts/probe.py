#!/usr/bin/env python3
"""Consistent decode/prefill probe across OpenAI-compatible engines.
Counts streamed SSE deltas (content+reasoning) and times first->last delta:
decode tok/s is immune to client-side tokenizer mismatch.
Usage: probe.py BASE_URL MODEL LABEL [PREFILL_CHARS]
"""
import sys, json, time, urllib.request

BASE, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
PREFILL_CHARS = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
DECODE_MAXTOK = 200

def stream(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
            if piece:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
    return t0, t_first, t_last, n, usage

# --- decode test: short prompt, generate DECODE_MAXTOK tokens ---
t0, tf, tl, n, usage = stream([{"role": "user", "content": "Write a detailed paragraph about the Pacific Ocean and marine biology."}], DECODE_MAXTOK)
ttft = (tf - t0) if tf else None
decode_tps = (n - 1) / (tl - tf) if (tf and tl and tl > tf and n > 1) else None
ct = usage.get("completion_tokens") if usage else None

# --- prefill test: long prompt, 1 token; ttft ~ prefill time ---
longp = ("The quick brown fox jumps over the lazy dog. " * (PREFILL_CHARS // 45))
t0b, tfb, tlb, nb, ub = stream([{"role": "user", "content": longp + "\nReply with one word."}], 8)
prefill_ttft = (tfb - t0b) if tfb else None
pt = ub.get("prompt_tokens") if ub else None
prefill_tps = (pt / prefill_ttft) if (pt and prefill_ttft) else None

print(json.dumps({
    "label": LABEL,
    "decode_tps_streamed": round(decode_tps, 2) if decode_tps else None,
    "decode_deltas": n,
    "server_completion_tokens": ct,
    "decode_ttft_s": round(ttft, 3) if ttft else None,
    "prefill_ttft_s": round(prefill_ttft, 3) if prefill_ttft else None,
    "prefill_prompt_tokens": pt,
    "prefill_tps": round(prefill_tps, 1) if prefill_tps else None,
}, indent=2))
