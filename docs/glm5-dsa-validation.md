# GLM-5.2 DSA validation runbook (M3 Ultra)

Validates the two fixes on branch `feature/glm5-dsa-1454` (port of mlx-lm PR #1454) against
the real model. Requires ≥512 GB unified memory and `mlx-community/GLM-5.2-mxfp4` (368 GB)
in the model cache. Everything below assumes the repo root as CWD and:

```bash
export CACHE=<your-model-cache-dir>          # dir containing mlx-community/GLM-5.2-mxfp4
export MODEL=mlx-community/GLM-5.2-mxfp4
```

## 0. Build both binaries

```bash
git fetch origin && git checkout main && ./build.sh
cp .build/arm64-apple-macosx/release/afm /tmp/afm-baseline

git checkout feature/glm5-dsa-1454 && ./build.sh
cp .build/arm64-apple-macosx/release/afm /tmp/afm-dsa
cp .build/arm64-apple-macosx/release/MacLocalAPI_MacLocalAPI.bundle/default.metallib /tmp/
```

Both stashed binaries find the metallib because it sits next to them in /tmp.

## 1. Needle harness (shared by tests 2–4)

Save as `/tmp/needle.sh`. Generates a haystack of `CTX` tokens with a secret code buried at
5 depths (10/30/50/70/90 %), asks for each code at temp=0, prints recall n/5.

```bash
#!/bin/bash
# usage: needle.sh <binary> <ctx-tokens> <port> [extra afm args...]
set -uo pipefail
BIN="$1"; CTX="$2"; PORT="$3"; shift 3
MACAFM_MLX_MODEL_CACHE=$CACHE "$BIN" mlx -m "$MODEL" --port $PORT "$@" \
  > /tmp/needle-server.log 2>&1 &
PID=$!; trap "kill $PID 2>/dev/null" EXIT
until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -5 /tmp/needle-server.log; exit 1; }
  sleep 5
done
python3 - "$CTX" "$PORT" <<'EOF'
import json, urllib.request, sys, time
ctx, port = int(sys.argv[1]), sys.argv[2]
para = ("The lighthouse keeper logged the wind speed, the wave height, the visibility, "
        "and the number of ships passing the strait every single morning without fail. ")
codes = {0.1:"HORIZON-4417", 0.3:"MARMOT-8212", 0.5:"GLACIER-3038", 0.7:"THISTLE-6564", 0.9:"LANTERN-1901"}
n_paras = ctx // 22  # ~22 tokens per paragraph
body = []
marks = {int(d*n_paras): c for d, c in codes.items()}
for i in range(n_paras):
    body.append(para if i not in marks
                else f"The secret code for checkpoint {list(marks.keys()).index(i)+1} is {marks[i]}. ")
hay = "".join(body)
ok = 0
for idx, (d, code) in enumerate(sorted(codes.items()), 1):
    q = hay + f"\n\nWhat is the secret code for checkpoint {idx}? Answer with the code only."
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps({"model": "glm", "temperature": 0, "max_tokens": 500,
                    "messages": [{"role": "user", "content": q}]}).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time()
    # 6h/question: baseline dense prefill at 128k runs ~9.7 tok/s => >5h/question.
    r = json.load(urllib.request.urlopen(req, timeout=21600))
    m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    hit = code in text
    ok += hit
    # flush: stdout is block-buffered when redirected — without it, hours of results
    # sit in the buffer and are lost if the run is killed (learned the hard way).
    print(f"  depth {int(d*100)}%: {'HIT ' if hit else 'MISS'} ({time.time()-t0:.0f}s)", flush=True)
print(f"RECALL {ok}/5", flush=True)
EOF
kill $PID 2>/dev/null; wait $PID 2>/dev/null
```

`chmod +x /tmp/needle.sh`. Note: pass `--enable-prefix-caching` to the server and
questions 2-5 reuse the shared-haystack prefix (~1s prompt_time instead of a full
prefill). Without the flag every question re-pays the full prefill (baseline at 128k:
>5h per question). CacheList prefix caching requires the stateArity restore fix
(2026-07-14); on older binaries the flag is unsafe for GLM-5 (silent empty-cache
restore).

## 2. Recall at 32k and 128k (default prefill step)

```bash
/tmp/needle.sh /tmp/afm-dsa      32000 9999
/tmp/needle.sh /tmp/afm-baseline 32000 9999
/tmp/needle.sh /tmp/afm-dsa      128000 9999
/tmp/needle.sh /tmp/afm-baseline 128000 9999
```

**Expected:** both binaries ~5/5 at both sizes (at the default 1024-token prefill step the
mlx#3784 zeroing is not triggered; this establishes the quality baseline and that the sparse
path doesn't hurt recall). Any DSA-branch result *below* baseline is a red flag — stop and
compare outputs.

## 3. Prefill speed A/B (the perf claim)

Time the first needle question at 40k (dominated by prefill). Alternate binaries, ≥2 pairs —
watch the per-question seconds printed by needle.sh (the depth-10% line of a FRESH server):

```bash
for pair in 1 2; do
  /tmp/needle.sh /tmp/afm-dsa      40000 9999 | head -2
  /tmp/needle.sh /tmp/afm-baseline 40000 9999 | head -2
done
```

**Expected:** DSA-branch first-question latency ≥1.3× faster (upstream: 1.39× end-to-end at
41k, and the gap widens with context — optionally repeat at 128k).

## 4. The mlx#3784 trigger (correctness claim)

The zeroing bug needs indexer score tensors ≥2³⁴ elements: 32 index heads × 4096-chunk ×
131072 context. Force the 4096 prefill step:

**Memory prerequisite (512 GB machine):** the trigger-sized score transients (~34-52 GB
each, 2-3 live under lazy eval) on top of 342 GB resident weights exceed the default MLX
wired limit (0.9 × recommendedMaxWorkingSetSize ≈ 417 GiB) — both binaries die mid-prefill
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Raise the OS ceiling first
(reversible, resets on reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=507904   # 496 GB -> afm wired ~446 GB
```

Even then the baseline leg (fused sum holds ~3 copies live) may not fit. A reduced
variant keeps the same 2³⁴-element trigger with far less pressure: ~65k tokens at
`--prefill-step-size 8192` (32 × 8192 × 65536 = 2³⁴):

```bash
/tmp/needle.sh /tmp/afm-baseline 46000 9999 --prefill-step-size 8192
/tmp/needle.sh /tmp/afm-dsa      46000 9999 --prefill-step-size 8192
```

Original full-scale form:

```bash
/tmp/needle.sh /tmp/afm-baseline 131072 9999 --prefill-step-size 4096
/tmp/needle.sh /tmp/afm-dsa      131072 9999 --prefill-step-size 4096
```

**Expected:** baseline collapses (upstream repro: 0/5 — attention silently degrades to the
last ~2048 tokens, so early-depth needles MISS while the 90% needle may still HIT); DSA
branch recalls ~5/5. This is the dramatic before/after.

Caveat: if the baseline does NOT collapse here, afm's pinned mlx 0.30.1 column-reduce may
not share the upstream bug (its int64 guard keys on total size) — that would demote fix #1
to insurance-only. Record the result either way; the sparse-prefill perf win (test 3) is
independent of it.

## 5. Reporting

Record per test: binary, context size, recall n/5, first-question seconds. Update the
`glm5-dsa-1454-port` memory / PR with results. If all pass: merge the branch. If upstream
mlx-lm #1454 changed since 2026-07-04, diff against the port first
(`gh pr diff 1454 --repo ml-explore/mlx-lm`).

## Results — 2026-07-05..07, M3 Ultra 512 GB

Binaries: dsa = branch @ 2136f8a; baseline = tmp-baseline (main + bdc658b + 2136f8a
cherry-picked, i.e. all correctness fixes but NO #1454 port). "ctx" is the needle
parameter; real token counts are ~1.41× (32000 → 45k, 128000 → 180k).

| Test | dsa | baseline | verdict |
|---|---|---|---|
| 2. recall 32k | **5/5** (~575 s/q) | **5/5** (~1170 s/q) | pass |
| 2. recall 128k | **5/5** (~4040 s/q, 45 tok/s pp) | incomplete¹ | dsa pass |
| 3. prefill A/B 40k, pair 1 | first-q 712 s | first-q 1745 s | **2.45×** |
| 3. prefill A/B 40k, pair 2 | first-q 716 s | first-q 1828 s | **2.55×** |
| 4. mlx#3784 trigger 131k/4096 | OOM² | OOM² | not runnable at default wired limit |

¹ Baseline 128k: first attempt hit the harness's old 2 h/question timeout; rerun (6 h
timeout) measured **pp 9.7 tok/s, ~5.1 h/question** (server STATS), i.e. a ~4.6× dsa
prefill advantage at 128k — the gap widens with context as upstream claimed — but the
host rebooted during question 5 and the buffered recall lines were lost (hence the
flush=True fix above). Re-run pending.

² Both legs died mid-prefill on the Metal wired limit (417 GiB default); see the memory
prerequisite in section 4. Retry with the raised limit / reduced variant pending. Fix #1
(indexer-sum workaround) currently rests on testIndexerHeadReductionEquivalence and
upstream's repro rather than an on-box collapse demonstration.

Bugs found and fixed by this campaign (all required before ANY GLM-5.2-mxfp4 inference
worked at depth): mxfp4 MultiLinear biases crash + qmv_fast_wide affine gate (bdc658b);
rope_theta .int decode (1M-vs-8M) + indexer_types cross-layer sharing (2136f8a). Known
afm follow-ups: server hard-aborts (std::runtime_error) instead of failing the request
on Metal OOM; Swift Jinja parser rejects `content.0` numeric attribute access (worked
around in the model cache); DSA decode drops ~8x once context exceeds index_topk
(per-token indexer + gather — separate investigation).

## Full-feature enablement — 2026-07-14, M3 Ultra 512 GB

GLM-5.2 (CacheList models generally) now works with afm's prefix cache and the
concurrent/batch path:

- **Prefix cache** (`--enable-prefix-caching`): CacheList restore-into-fresh was a
  silent no-op (CacheList.state split by current — empty — child counts); fixed via
  stateArity. Radix tree also gained a subtree fallback (edge-split branch nodes hold
  no entry — sibling conversations sharing a system prompt always missed) and an
  exact-duplicate insert dedupe (~5s wasted snapshot per repeat request). Measured:
  4k-token shared prefix, cold 23.9s prompt_time -> warm 0.9s (26x), retrieval from
  restored KV correct, exact replay correct (hasRecurrentLayers now recurses:
  CacheList-of-attention is replay-safe; Mamba children still bypass).
- **Concurrent/batch** (`--concurrent N`): two zero-width crashes fixed
  (BatchKVCacheSimple.filter take on [B,H,S,0] values at slot finish;
  BatchCacheList.extract rank-1 empties into KVCacheSimple.state for never-updated
  shared-layer indexer sub-caches). Measured: batch endpoint B=4 4/4 known-answer
  correct; 3 simultaneous chat completions 3/3, wall time = slowest request (true
  parallel decode); batched prefill B=3 in scheduler log.
