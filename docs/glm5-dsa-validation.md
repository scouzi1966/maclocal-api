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
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    hit = code in text
    ok += hit
    print(f"  depth {int(d*100)}%: {'HIT ' if hit else 'MISS'} ({time.time()-t0:.0f}s)")
print(f"RECALL {ok}/5")
EOF
kill $PID 2>/dev/null; wait $PID 2>/dev/null
```

`chmod +x /tmp/needle.sh`. Note: the FIRST question pays full prefill; later ones reuse the
prompt cache (shared prefix), so a 5-question round is much cheaper than 5 full prefills.

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
