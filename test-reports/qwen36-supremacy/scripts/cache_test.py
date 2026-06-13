#!/usr/bin/env python3
"""Realistic prefix-cache test (what a radix cache is actually for):
a long SHARED context reused across turns, with a DIFFERENT short question each turn.

  cold  : fresh unique context + question         -> full prefill (baseline TTFT)
  prime : shared context + question A             -> populates the cache
  hit   : shared context + question B (new suffix)-> should reuse the shared prefix

Reports TTFT for each + speedup (cold/hit) + server-reported cached_tokens (usage.prompt_tokens_details).
Usage: cache_test.py BASE_URL MODEL LABEL [CTX_TOKENS]
"""
import sys, json, time, random, urllib.request
BASE, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
CTX = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
FILLER = "The quick brown fox jumps over the lazy dog. "

def ctx_block(tag):
    reps = max(1, int(CTX * 4.45 / len(FILLER)))
    return f"[{tag}] " + FILLER * reps

def send(content, max_tokens=4, timeout=600):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "ignore").strip()
            if not ln.startswith("data:"): continue
            dt = ln[5:].strip()
            if dt == "[DONE]": break
            try: o = json.loads(dt)
            except Exception: continue
            if o.get("usage"): usage = o["usage"]
            ch = o.get("choices") or []
            if ch and ttft is None:
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    ttft = time.time() - t0
    return ttft, (usage or {})

run = random.randint(0, 10**9)
shared = ctx_block(f"shared-{run}")

# baseline: a different, never-before-seen context
ttft_cold, u_cold = send(ctx_block(f"cold-{run}") + "\nQuestion: Summarize in one word.")
# prime the shared prefix
ttft_prime, u_prime = send(shared + "\nQuestion: What is two plus two? One word.")
# reuse shared prefix with a NEW question (new suffix)
ttft_hit, u_hit = send(shared + "\nQuestion: Name one ocean. One word.")

def cached(u):
    d = u.get("prompt_tokens_details") or {}
    return d.get("cached_tokens")

out = {
    "label": LABEL, "ctx_tokens_req": CTX,
    "prompt_tokens": u_prime.get("prompt_tokens"),
    "cold_ttft_s": round(ttft_cold, 3) if ttft_cold else None,
    "prime_ttft_s": round(ttft_prime, 3) if ttft_prime else None,
    "hit_ttft_s": round(ttft_hit, 3) if ttft_hit else None,
    "speedup_cold_over_hit": round(ttft_cold / ttft_hit, 1) if (ttft_cold and ttft_hit and ttft_hit > 0) else None,
    "server_cached_tokens_on_hit": cached(u_hit),
}
print(json.dumps(out, indent=2))
