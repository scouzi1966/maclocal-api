# Merge Cliff Investigation — BatchScheduler Throughput Stalls

Date: 2026-04-08 (updated 2026-04-08 with Codex follow-up findings)
Status: **Diagnosis complete. Direction decided: unified slot-resident KV cache with bulk cohort admissions.** Paged attention has been evaluated and declined for this workload — see §4 and §10 for the full reasoning and the two Codex consultations that informed the decision.
Branch context: `feature/batch-fixed-slab-kv` (post-revert), baseline `6a4179d`, diagnostic probes `66bbbac`. Reference branch `slab-reference` pinned at `9fe46ae` preserves the prior slab attempt for reference.
Author note: This document is written for an AI coding agent picking up work on this problem after a session break. Read the "Quick Orientation" first; the rest of the document is reference material with citations.

---

## Quick Orientation (read this first)

**Problem**: When new requests join an already-running concurrent decode batch in `BatchScheduler`, the next decode step stalls for several seconds. GPU goes idle. Stalls grow linearly with batch size — measured at 22 s at B=199 on Qwen3.5-35B-A3B-4bit.

**Root cause**: `BatchKVCacheSimple.merge()` / `extend()` calls `MLX.concatenated([existing, new], axis: 0)` along the batch dim for K and V on every layer. With 60 layers × 2 (K+V) = 120 separate `concatenated` calls per merge event, the work materializes serially through the Metal command queue on the next `model()` forward pass. Cost is dominated by 120-op submit/wait serialization, not raw memcpy bandwidth.

**Failed approach**: A "slab" KV cache that pre-allocates `[maxConcurrent, kvHeads, maxSeqLen, headDim]` upfront and writes new sequences in-place via slice-assignment. Failed because (a) it only covered attention layers — the hybrid Qwen3.5-35B-A3B model has ~30 Mamba/SSM layers (`MambaCache`/`ArraysCache`) that still concat on merge, and at this layer count they dominate the merge cost; (b) the slab's per-row slice-assigns accumulated as 120N small Metal ops per merge; (c) even the bulk single-slice-assign-per-layer fast path required every layer to be slab-mode, which the hybrid model violates. Reverted in this branch — recoverable from reflog as commit `9fe46ae` if needed.

**Decided direction**: **Unified slot-resident KV cache across all cache families (`BatchKVCacheSimple`, `ArraysCache`/`MambaCache`, `BatchCacheList` sub-caches), with bulk cohort admissions**. This is "the slab attempt redone correctly": (a) covers hybrid models by extending slab mode to `ArraysCache`, (b) does one slice-assign per layer per merge event instead of 120×N slice-assigns per cohort, (c) bulk-merges the entire admitted cohort in a single call from `prefillBatch` instead of the current `for i in 0..<B { mergeCacheIntoBatch(...) }` loop at `BatchScheduler.swift:1333`.

**Paged attention status**: **declined for this workload.** Two Codex consultations evaluated it. Technically feasible via `gatherMM` without custom Metal kernels (~3-5 weeks), but carries an estimated 2-3× per-attention-step overhead vs the fused `MLXFast.scaledDotProductAttention` decode kernel. On a 512 GB M3 Ultra with no memory pressure, paged attention's main advantage (memory efficiency from non-rectangular allocation) is zero, and its latency advantage (O(1) merge cost) is achievable via unified slot-resident without paying the fused-kernel loss. The math is net-neutral to net-negative compared to unified slot-resident. See §4 and §10 for the full reasoning and citations.

---

## 1. Problem Statement

### 1.1 Symptom

`./Scripts/demo-concurrent-throughput.sh --concurrent 150 --ramp-step-users 2 --ramp-step-s 10 --hold 15 --max-tokens 1500` against `mlx-community/Qwen3.5-35B-A3B-4bit` on M3 Ultra 512 GB shows periodic, multi-second throughput dips during the ramp and hold phases. The dips:

- Are visible on the live chart (`Scripts/demo/watch_live.py`) and persist after fixing metrics-side sampling artifacts.
- Are confirmed real by independent measurement: `proxy_tps` (computed from raw SSE bytes received by the driver) shows 16 dips ≥ 2 s long, p50 = 3.3 s, max = 9.3 s during a B=150 hold.
- Coincide with GPU idle time (user observation, validated independently).
- Persist when the ramp cadence is jittered with `--ramp-jitter-pct 40 --ramp-jitter-seed 42`, which initially looked like the cause was independent of the ramp but turned out to be wrong.

### 1.2 Causal mechanism (validated by probes)

Diagnostic probes added in commit `66bbbac` to `Sources/MacLocalAPI/Models/BatchScheduler.swift` (gated behind `AFM_DEBUG=1`):

- `LOOP GAP`: top-of-decode-loop wall-clock gap detector (>500 ms threshold). Reports B, pending queue depth, and a phase split.
- Per-iteration `model=X dispatch=X other=X` attribution (which phase ate the gap).
- `merge=yes/no` flag (whether `drainPendingQueue()` returned new requests this iteration).
- `EVAL FLUSH (512-step)` wall time around the periodic graph-materialization call inside the `if stepCount % 512 == 0` block at line ~789.
- `MEMORY CLEARCACHE` wall time around `Memory.clearCache()` at line ~782.
- `FILTER` wall time around the `filter()` call in `finishSlot()` (~10 ms threshold).

A focused B=150 run with these probes produced:

```
Total slow iterations (>500 ms): 23
  with merge=yes: 22
  with merge=no:  1   (B=1, startup, irrelevant)

Phase contribution across all 23 slow iterations:
  total gap : 69.74 s
  model     : 60.99 s  (87 %)
  dispatch  :  0.03 s  (0  %)
  other     :  8.72 s  (13 %)

Model-wall scaling with B (p50 → max):
  B = [ 30,  60): 0.77 → 1.12 s
  B = [ 60,  90): 1.48 → 3.75 s
  B = [ 90, 120): 4.57 → 5.71 s
  B = [120, 160): 6.80 → 9.12 s
```

**Verdict**: Every measurable stall is in the `model()` forward pass on an iteration that just merged a new request. dispatch is fine (deferred-dispatch pipelining works). `EVAL FLUSH` is essentially free (<3 ms). `FILTER` is essentially free (no entries above 10 ms). `Memory.clearCache()` is essentially free (no entries above 10 ms). The cost is not in cache housekeeping — it is in the model forward pass paying for the lazy-graph materialization of the merge that just happened.

### 1.3 Why the jitter test was misleading at first

Earlier in the investigation, jittering the ramp cadence (`--ramp-jitter-pct 40`) did NOT change the visible dip cadence on the chart. This made it look like the ramp wasn't causal. The actual reason: **only EXPENSIVE merges (high-B) produce visible dips**. Cheap merges at small B happen on the jittered schedule but don't show as dips. The visible dip cadence is set by "how often does B advance enough to make the next merge cost ≥ 1-2 s of model() wall," which is roughly invariant to input timing because B grows monotonically through the ramp regardless of inter-arrival jitter. The phase-split LOOP GAP probe confirmed `merge=yes` on 22/23 slow iterations after this misdirection was resolved.

### 1.4 Things ruled out (with measurements)

| Hypothesis | How ruled out |
|------------|---------------|
| Metrics sampling artifact | `proxy_tps` (from raw SSE bytes) shows independent dips with same cadence and depth. |
| OS memory pressure / VM compressor | 512 GB machine, working set ~50 GB, pressure thresholds nowhere near triggered. |
| Ramp cadence | Jittered ramp with `--ramp-jitter-pct 40` produces the same dip pattern. Phase-split probe shows merges drive the dips. |
| 512-step graph flush | `EVAL FLUSH` probe: all calls < 3 ms, one outlier at 77 ms. Not a stall source. |
| `filter()` in `finishSlot()` | `FILTER` probe (10 ms threshold): zero entries in a full B=150 run. |
| `Memory.clearCache()` in decode loop | `MEMORY CLEARCACHE` probe: zero entries. |
| Seq-dim cache pre-alloc grow | `AFM_BATCH_KV_STEP=4096` (16× the default 256) made no measurable difference. |
| Slot completion churn | Per-sequence STATS lines show steady ~5 tok/s; finishes are continuous, not bursty. |

---

## 2. Failed Approach: Slab KV Cache

### 2.1 What was tried

Branch `feature/batch-fixed-slab-kv`, commit `9fe46ae` (reset away in `6a4179d`, recoverable via `git reflog`).

In `Scripts/patches/BatchKVCache.swift`, `BatchKVCacheSimple` was extended with a "slab mode":

- New init taking a `capacity` parameter that pre-allocates K/V at `[capacity, kvHeads, maxSeqLen, headDim]` once at construction time.
- `addSlot(keys:values:)` writes one new sequence's K/V into row `batchSize` of the slab via slice-assignment, then bumps `batchSize`. No concat along batch dim.
- `addSlotsBulk(from: BatchKVCacheSimple)` writes ALL new rows from a source slab into the destination slab in ONE slice-assignment per layer.
- Host-side mirrors (`hostLeftPadding`, `hostPerSeqOffset`) so `addSlot` does not call `asArray(Int32.self)` on every call (which would force a GPU sync per addSlot — that early version was a regression).
- `allOffsetsEqual` and `zeroPadding` rewritten to scan the host arrays directly so they never trigger `.item()`-style synchronous materializations.
- `filter()` rewritten to compact in-place via row-copy when in slab mode.

Wired into `BatchScheduler.mergeCacheIntoBatch()` and `prefillBatch()`. CLI flag was env-var `AFM_BATCH_KV_STEP` (separate concern).

### 2.2 Why it failed

Three independent reasons, in order of impact:

1. **Hybrid model**: `mlx-community/Qwen3.5-35B-A3B-4bit` is a hybrid Mamba/Transformer architecture. Approximately 60 attention layers use `BatchKVCacheSimple` (covered by slab mode) but ~30 SSM layers use `MambaCache`/`ArraysCache` which were NOT modified. Those layers still concat on merge. At this layer count they dominate the merge cost. The slab fix made attention-layer merges cheap but the chart looked the same because Mamba was the new bottleneck.

2. **Per-row slice-assigns accumulate as command-buffer ops**: Even where slab mode was used, each `addSlot` does `keys[row, ..., :L, :] = newSlotKeys` which generates one Metal slice-assign command per layer per K/V tensor = 120 ops per slot. At N=36 accumulated slot-adds between decode steps (which happened during the ramp), that's 4320 graph nodes accumulated. The next forward pass materialized them all at once, hitting the same Metal command-queue serialization wall as the original concat-based code, just with smaller per-op chunks.

3. **`addSlotsBulk` fast path required all-slab layers**: The bulk path (one slice-assign covering all new rows) reduced ops to 120 per merge regardless of N, but its precondition was that EVERY layer in the model is slab-mode `BatchKVCacheSimple`. Hybrid models (Mamba layers, CacheList layers, RotatingKVCache layers) failed this check and fell through to the per-slot loop. So the fast path effectively never fired on the target model.

### 2.3 Validation of failure (correctness)

Slab-mode passed 32/32 known-answer tests at B={1,2,4,8} via `Scripts/feature-mlx-concurrent-batch/validate_responses.py`. The failure was performance, not correctness. The implementation is correct; it just doesn't deliver the throughput improvement on this model.

### 2.4 What slab DID achieve (don't lose this learning)

POST-MERGE wall reduction on the attention-layer side, measured pre vs post:

| B   | pre-fix POST-MERGE | post-fix POST-MERGE | ratio |
|-----|-------------------:|--------------------:|------:|
|  20 | 0.091 s            | 0.065 s             | 1.4× |
|  45 | 1.000 s            | 0.143 s             | 7×   |
|  67 | 1.800 s            | 0.256 s             | 7×   |
| 113 | 8.200 s            | 2.693 s             | 3×   |

So slab mode is the right pattern for `BatchKVCacheSimple`. It just doesn't solve the problem alone on hybrid models. **If/when this work is revived, it should be combined with slab-mode `MambaCache`** — that's the real fix for hybrid models, and the slab attention-layer code is a partial solution worth keeping.

---

## 3. MLX Primitive Discovery

This is the optimistic part. While planning the next attempt, the question came up: does MLX Swift expose primitives that can implement paged attention WITHOUT writing custom Metal kernels? The answer is yes — and there's a buried gem.

### 3.1 What's wrapped in MLX Swift

| Primitive | Swift API | Source location | Status |
|-----------|-----------|-----------------|:-------|
| `scaledDotProductAttention` | `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:sinks:stream:)` | `.build/checkouts/mlx-swift/Source/MLX/MLXFast.swift:118, 203` | The canonical attention call. Strict 4-D contiguous K/V layout `[B, N_kv, T_kv, D]`. Has a fast Metal kernel for `T_q == 1` (the decode case); other shapes fall through to general matmul-based attention. |
| `gatherMM` | `gatherMM(_:_:lhsIndices:rhsIndices:sortedIndices:stream:)` | `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:1371` | Fused gather + matmul. Doc explicitly notes: "more efficient than explicitly applying a `take` followed by a `matmul`." `lhsIndices`/`rhsIndices` are flat indices into the batch dims. **This is the primary candidate for paged attention without custom kernels.** |
| `blockMaskedMM` | `blockMaskedMM(_:_:blockSize:maskOut:maskLHS:maskRHS:stream:)` | `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:426` | Block-sparse matmul. Alternative paged-attention approach: keep K/V contiguous and mask out blocks not belonging to the current sequence. |
| `MLXFast.metalKernel(...)` | Compile a Metal shader source string into a callable kernel | `.build/checkouts/mlx-swift/Source/Cmlx/include/mlx/c/fast.h:145` (C ABI) | Last-resort escape hatch for custom Metal kernels. We have user permission to use this if needed. |

### 3.2 What's NOT wrapped in MLX Swift but exists in MLX C++ core

**`segmented_mm`** is the buried gem. Variable-length segment-aware matmul, equivalent in spirit to flash-attention's `flash_attn_varlen_func`.

**Locations**:

- C++ entry point: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/ops.cpp:5535` — `array segmented_mm(array a, array b, array segments, ...)`
- C++ header: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/ops.h:1506`
- C ABI header: `.build/checkouts/mlx-swift/Source/Cmlx/include/mlx/c/ops.h:950` — `int mlx_segmented_mm(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_array segments, const mlx_stream s);`
- C ABI implementation: `.build/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c/ops.cpp:3015`
- Metal kernel: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented.metal` — full implementation with shape variants for fp16, bf16, fp32, multiple `BM`/`BN`/`BK`/`WM`/`WN` combinations (lines 30-34)
- Metal kernel template: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented.h`
- Metal dispatch: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp:2207, 2365`
- CPU implementation: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/cpu/masked_mm.cpp:57`
- Python binding (the proof someone already wrapped this elsewhere): `.build/checkouts/mlx-swift/Source/Cmlx/mlx/python/src/ops.cpp:4430` — `mx.segmented_mm(a, b, segments, ...)` exposed in `mlx.core`
- Python tests: `.build/checkouts/mlx-swift/Source/Cmlx/mlx/python/tests/test_blas.py:1261` — `test_segmented_mm` with reference implementation
- **Swift wrapper: does not exist anywhere in `.build/checkouts/mlx-swift/Source/MLX/`. Confirmed by exhaustive grep.**

**Constraint**: 2D-only. From `ops.cpp:5541`: `if (a.ndim() != 2 || b.ndim() != 2) { throw std::invalid_argument("[segmented_mm] Batched matmul not supported"); }`. The `segments` argument has shape `(..., 2)` where the last dim is `[start, end]` row-pair offsets into `a`/`b`. For multi-head attention this means head dim and batch dim must be flattened in.

### 3.3 Other potentially-relevant ops in `mlx_ops.h`

Found via `grep -E "^int mlx_" mlx-c/include/mlx/c/ops.h | grep -iE "gather|scatter|index|sparse"`. These are mostly already in MLX Swift but worth listing:

- `mlx_gather` / `mlx_gather_single` — basic gather (Swift-wrapped as `take`/`take_along_axis`)
- `mlx_gather_mm` — fused gather + matmul (Swift: `gatherMM`)
- `mlx_gather_qmm` — fused gather + quantized matmul (Swift: `gatherQuantizedMM`) — relevant for quantized weights but not directly for K/V cache (K/V is bf16 even when weights are 4-bit)
- `mlx_block_masked_mm` — block-sparse matmul (Swift: `blockMaskedMM`)
- `mlx_masked_scatter` / `mlx_scatter*` family — write side
- `mlx_take` / `mlx_take_along_axis` — Swift-wrapped

The `mlx_fast_*` surface in `Cmlx/include/mlx/c/fast.h` is small and fully mapped to MLX Swift's `MLXFast.swift`:

- `mlx_fast_layer_norm`
- `mlx_fast_rms_norm`
- `mlx_fast_rope` / `mlx_fast_rope_dynamic`
- `mlx_fast_scaled_dot_product_attention`
- `mlx_fast_metal_kernel_*` (custom kernel API — Swift wraps this for user-defined kernels)

There are no hidden paged-attention or flash-varlen primitives in the `mlx_fast` namespace. The interesting unexploited stuff is all in `mlx_ops.h`, not `mlx_fast.h`.

---

## 4. Architectural Implications for Paged Attention in AFM

### 4.1 Strategies evaluated, with decision

**DECIDED: Strategy 0 — Unified slot-resident KV cache with bulk cohort admissions**

Not a new idea, but a deliberate redo of the slab attempt with the three failure modes addressed head-on. The approach:

1. **Slab mode is made uniform across all cache families**:
   - `BatchKVCacheSimple` — already designed in reflog `9fe46ae`, needs recovery + refinement.
   - `ArraysCache` / `MambaCache` — new slab implementation. Simpler than `BatchKVCacheSimple` because there's no seqLen dim and no left padding — just a fixed-size list of tensors with a batch dim.
   - `BatchRotatingKVCache` — deferred to v2 (only affects Gemma 4, which uses `RotatingKVCache` on a subset of layers).
   - `BatchCacheList` — delegates to its sub-caches, which are now all slab-mode.

2. **Bulk cohort admissions**: replace the `for i in 0..<B { mergeCacheIntoBatch(individualCache: perRequestCaches[i], ...) }` loop in `BatchScheduler.prefillBatch` (line ~1333) with a single `bulkMergeCohort(prefillCaches: [KVCache])` call that does ONE slice-assign per layer regardless of cohort size N. Each slab cache type implements `addSlotsBulk(from: SameSlabType)` to copy N rows in one Metal dispatch.

3. **Feature flag**: `--kv-cache slab|standard` at the AFM CLI, plumbed through `MLXModelService` → `BatchScheduler`. Default remains `standard`. Opt-in per run.

**Why this works where the slab attempt failed**:

| Slab failure mode | Fix in unified slot-resident |
|--------------------|------------------------------|
| Only covered attention layers; Mamba still concats | Slab mode extends to `ArraysCache`/`MambaCache` in the same patch |
| Per-slot slice-assigns = 120N Metal ops per cohort | Bulk-cohort `addSlotsBulk` = 120 Metal ops per cohort, regardless of N |
| `addSlotsBulk` fast path required all-slab layers | Precondition met by construction — slab is the protocol across all cache families |

**Expected impact**: recover most of the gap from current ~450 tok/s aggregate toward the ~750 tok/s ceiling at B=150. POST-MERGE wall at B=127 should drop from ~7 s into low single-digit seconds.

**Scope estimate**: 2-3 weeks focused work. Milestones tracked in `docs/unified-slot-resident-plan.md` (to be written).

---

### 4.2 Strategies evaluated and declined

**Strategy A — `gatherMM`-based paged attention (feasible but net-neutral)**

Evaluated in the second Codex consultation (§10.2). Technically implementable: `gatherMM` is a genuinely fused gather+matmul, confirmed at `steel_gemm_gather.h:293-306` and `matmul.cpp:2158-2205`. Shape semantics for multi-head attention work with `rhsIndices` of shape `[B, H]` for per-head per-sequence block selection.

**Why declined**: Codex's follow-up estimated **~2-3× per-attention-step overhead** relative to `MLXFast.scaledDotProductAttention`'s fused decode kernel, because replacing the SDPA call with `gatherMM(QK) + softmax + gatherMM(scoreV)` is a 3-kernel-dispatch chain vs SDPA's single fused kernel with in-kernel fp32 softmax. On a 512 GB M3 Ultra with no memory pressure, paged attention's unique advantage (memory efficiency from non-rectangular allocation) is zero, and its latency advantage (O(1) merge cost) is achievable via unified slot-resident without paying the 2-3× per-step cost. Net math: ~450 tok/s → ~450-600 tok/s (no significant improvement because the stall-elimination win is canceled by the per-step regression). Compare to unified slot-resident's ~450 → ~650-700 tok/s with zero per-step regression.

**Strategy B — `segmented_mm`-based paged attention (wrong shape semantics)**

Evaluated in the second Codex consultation. `segmented_mm` has a hard 2D-only constraint and its output shape is `segments.shape[:-1] + [a.shape(0), b.shape(1)]` with the segment axis orthogonal to the M row axis (citation: `ops.cpp:5571-5577`, `steel_gemm_segmented.h:70-84`). That shape is incompatible with per-sequence attention decoding where each row needs a different K window. Would require separate segmented_mm calls per head or complex shape gymnastics. **Not worth the 20-line Swift wrapper** — not suitable as the primary primitive.

**Strategy C — Custom Metal kernel via `MLXFast.metalKernel(...)`**

Original scope estimate from Codex: 2-3 months. Off the table for now because Strategy 0 (unified slot-resident) is much cheaper and expected to reach the throughput ceiling without writing new Metal code. Stays available as a future stage-2 if unified slot-resident leaves residual attention-layer stalls.

### 4.2 The Mamba problem is independent

Even with full paged attention for the transformer attention layers, the ~30 Mamba layers in Qwen3.5-35B-A3B still need their own merge-cost fix. The Mamba state is per-sequence and small (a single state tensor per layer per sequence), so the merge cost there is fundamentally different from K/V concat — but it still grows with B and currently uses the same concat-along-batch-dim pattern in `MambaCache`/`ArraysCache`.

A complete solution likely needs:
1. Paged attention for the transformer layers (Strategies A/B/C above).
2. Slab-style or paged-style state management for the Mamba layers.

These are independent work items. The transformer paged-attention work should be validated on a pure-attention model first (e.g., Qwen3-30B-Instruct, Llama 3.3 70B) where the Mamba problem doesn't apply, before tackling the hybrid case.

### 4.3 Feature flag is mandatory

Whatever direction is taken, paged attention MUST be a CLI/runtime opt-in, not a default. Reasons:

- Apple Foundation Models backend doesn't go through `BatchScheduler` — paged attention is irrelevant there.
- Models with `RotatingKVCache` (Gemma 4 sliding window), K/V sharing (Gemma 4), `CacheList` (Qwen3.5 hybrid attention variants), multimodal inputs (VLMs) all have assumptions baked into the model code that the alternative cache path may break. The slab failure showed how easily a "drop-in replacement" hits hidden coupling.
- Existing AFM users running production workloads cannot tolerate a regression on day one.

Proposed CLI shape (matches existing AFM ArgumentParser style):

```
--kv-cache standard|paged          # default: standard
--kv-block-size N                  # default depends on model, probably 64-256
```

Note: vLLM has no equivalent on/off flag because vLLM was built around paged attention from day one and never maintained the alternative. AFM is in a different position — we're adding paged attention to an existing system with users.

---

## 5. Diagnostic Infrastructure Reference

All probes are gated on `AFM_DEBUG=1`. Located in `Sources/MacLocalAPI/Models/BatchScheduler.swift` (current commit `66bbbac` includes them).

| Probe label | Where | What it measures |
|-------------|-------|------------------|
| `LOOP GAP` | top of `generationLoop()` while loop, ~line 600 | Wall-clock gap between consecutive iteration tops; threshold 500 ms; reports B, pending queue depth, and a phase split (model/dispatch/other) of the previous iteration |
| `EVAL FLUSH (512-step)` | inside the `if stepCount % 512 == 0` block, ~line 789 | Wall time of the periodic graph-materialization call on cache arrays |
| `MEMORY CLEARCACHE` | inside the `if totalTokensGenerated % 1024 < activeB` block, ~line 782 | Wall time of `Memory.clearCache()`; threshold 10 ms |
| `FILTER` | inside `finishSlot()` around the `filter()` block, ~line 1620 | Wall time of the per-slot batched-cache rebuild on slot completion; threshold 10 ms; reports B_before/after and current seqLen |

Run pattern for diagnostic mode:

```bash
pkill -f "concurrent_load_driver" 2>/dev/null
pkill -f "afm mlx" 2>/dev/null
sleep 2
AFM_DEBUG=1 ./Scripts/demo-concurrent-throughput.sh \
  --concurrent 150 --ramp-step-users 2 --ramp-step-s 10 \
  --ramp-jitter-pct 40 --ramp-jitter-seed 42 \
  --hold 15 --cooldown 20 --max-tokens 1500 \
  --skip-render --skip-html --skip-verify
```

Then:

```bash
LATEST=$(ls -t /tmp/afm-demo-server-*.log | head -1)
grep -E "LOOP GAP|EVAL FLUSH|FILTER:|MEMORY CLEARCACHE" "$LATEST"
python3 Scripts/demo/correlate_dips.py "$LATEST" Scripts/demo/out/trace.jsonl
```

Note that `AFM_DEBUG=1` adds a forced sync of `output.logits` on the iteration after a merge (POST-MERGE probe — also in the codebase though removed during the slab revert; reintroduce if needed). This breaks the normal pipelined overlap, so absolute throughput numbers in DEBUG mode are NOT representative of real-world performance. The probe wall numbers themselves are what matter, not the aggregate tok/s.

### 5.1 Helper: `Scripts/demo/correlate_dips.py`

Aligns server log events (`Batched prefill` lines, `STATS` lines per finish) with trace.jsonl throughput dips. Reports cadence comparison and per-dip event windows. Useful for ruling event types in or out.

### 5.2 Ramp jitter for cadence causality tests

`Scripts/demo/concurrent_load_driver.py` accepts `--ramp-jitter-pct PCT` and `--ramp-jitter-seed N`. Both plumbed through `Scripts/demo-concurrent-throughput.sh`. Use this to break the ramp's fixed period and check whether observed periodic behavior is forced by the ramp or intrinsic to the server.

---

## 6. Branch and Commit Reference

Current state of this work (as of writing):

- **Main work branch**: `feature/batch-fixed-slab-kv` (despite the name, the slab work has been reset away — branch now contains the diagnostic infrastructure and experiment scripts only)
- **Baseline**: `6a4179d` — `feat(demo): step ramp, cooldown, prefill stats, moving average`
- **Diagnostic probes**: `66bbbac` — `chore(batch): FILTER + EVAL FLUSH timing probes (AFM_DEBUG)`
- **Ramp jitter feature**: `ed5e051` — `feat(demo): --ramp-jitter-pct for cadence causality test`
- **Correlation analyzer**: `599e639` — `feat(demo): correlate_dips.py — align server log events with trace dips`
- **Reverted slab experiment** (recoverable from reflog): `9fe46ae` — `feat(batch): slab-mode BatchKVCacheSimple for in-place slot adds`

Recovery command if the slab work needs to be revisited:

```bash
git branch slab-experiment 9fe46ae
git checkout slab-experiment
```

---

## 7. Reading List for an AI Agent Picking This Up

Read these files in this order:

1. **This document** — full context
2. `Scripts/patches/BatchKVCache.swift` — the cache classes; understand `BatchKVCacheSimple`'s `merge`, `extend`, `filter`, and the relationship to `BaseKVCache`. Also `BatchRotatingKVCache` for the sliding-window case.
3. `Sources/MacLocalAPI/Models/BatchScheduler.swift` — focus on `generationLoop()`, `mergeCacheIntoBatch()`, `prefillBatch()`, `finishSlot()`. Note the diagnostic probes throughout.
4. `.build/checkouts/mlx-swift/Source/MLX/MLXFast.swift` lines 100-220 — understand the SDPA call signature constraint.
5. `.build/checkouts/mlx-swift/Source/MLX/Ops.swift` lines 426-490 (`blockMaskedMM`) and 1371-1410 (`gatherMM`).
6. `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/ops.cpp:5535` — the C++ definition of `segmented_mm` and its constraints.
7. `.build/checkouts/mlx-swift/Source/Cmlx/mlx/python/tests/test_blas.py:1261` — Python test of `segmented_mm` showing how it's expected to be called and what the segment shape means in practice.
8. `vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3*.swift` (or whichever model file the benchmark target uses) — understand how the existing model's attention layer calls `MLXFast.scaledDotProductAttention`. This is the call site that paged attention has to replace.
9. `docs/paged-attention-radix-feasibility.md` — earlier feasibility note focused on the prefix-cache angle (a different problem from this one but contextually related).
10. `CLAUDE.md` (project root) — global project conventions, especially the vendor patch system and the Metal buffer lifecycle note.

---

## 8. Open Questions — Updated

Most of the original questions were answered by the Codex consultations (§10). Remaining questions for the unified slot-resident implementation:

### Answered

1. ~~Does `gatherMM` shape semantics work for multi-head?~~ **Yes**, `rhsIndices: [B, H]` gives per-head per-sequence block selection (Codex §10.2, verified at `ops.cpp:5504-5510`).
2. ~~How much does `gatherMM` cost vs SDPA fast path?~~ **~2-3× per step** (Codex §10.2, hypothesis). Enough to make it net-neutral vs unified slot-resident on this hardware.
3. ~~Is SDPA reachable from a custom path with gathered K/V?~~ **No** — SDPA internally copies non-contiguous inputs back to contiguous (Codex §10.1, `scaled_dot_product_attention.cpp:617`). Must either bypass SDPA entirely or keep cache contiguous.

### Open

1. **How much does `MambaCache.extend()` actually cost on Qwen3.5-35B-A3B at B=150?** The slab investigation found Mamba dominates after attention slab is applied, but never quantified. **Action**: add a probe around `BatchCacheList.merge(...)` and `ArraysCache.extend(other:)` and measure in the next debug run. If Mamba merges dominate at ~80% of the remaining stall, the Mamba-first approach is correct.
2. **Pure-attention model baseline**: does the same merge cliff appear at B=150 on a non-hybrid model (e.g., Qwen3-30B-Instruct-4bit if that's in the cache)? If the cliff is smaller on a pure-attention model, it confirms that the Mamba path is a significant contributor on hybrid models, which validates the unified-across-all-families approach. If the cliff is the same size, attention itself is the primary driver even without Mamba.
3. **What's the ideal seqLen capacity for the unified slab?** `max_tokens` bound (default 1500-3000) + some headroom for prefill. At `[maxConcurrent=200, kvHeads=4, maxSeqLen=4096, headDim=128, bf16]` per layer: ~800 MB/layer × 60 layers × 2 (K+V) = ~96 GB resident. Fine on 512 GB. At `[maxSeqLen=8192]` that doubles to ~192 GB — still fine but worth making configurable via `--kv-max-seq-len`.
4. **Does `BatchRotatingKVCache` need slab mode in v1?** Only affects Gemma 4. Deferred to v2 unless the benchmark target model uses it.
5. **Can `BatchCacheList.merge(...)` and `BatchCacheList` sub-cache extension be made non-concat in one commit**, or does it need a dedicated "bulk" variant separate from the single-slot merge path?

---

## 9. References

- vLLM PagedAttention paper: Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).
- vLLM source: `vllm/attention/ops/paged_attn.py`, `vllm/worker/cache_engine.py`, `csrc/attention/paged_attention_*.cu` (web reference, not in this repo).
- vLLM CLI flags: `--block-size`, `--max-num-batched-tokens`, `--max-num-seqs`, `--enable-prefix-caching`, `--swap-space`, `--gpu-memory-utilization`, `--enforce-eager`. None of these are an "enable paged attention" flag — paged is always on in vLLM.
- MLX Swift gatherMM: `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:1371`
- MLX Swift blockMaskedMM: `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:426`
- MLX Swift MLXFast SDPA: `.build/checkouts/mlx-swift/Source/MLX/MLXFast.swift:118, 203`
- MLX C++ segmented_mm (no Swift wrapper): `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/ops.cpp:5535`

---

## 10. Codex Consultations (grounded source analysis)

Two consultations were run via the `codex:codex-rescue` skill against the real vendor source. Both verdicts converge on "unified slot-resident, not paged attention." Summaries preserved here so the reasoning survives a session break.

### 10.1 First consultation — architectural assessment (task `task-mnqrq82s-8xd1sw`, 16 min, model: default)

**Prompt**: vLLM paged attention feasibility in MLX Swift, scope estimate, hybrid model handling, alternatives to the failed slab.

**Key findings**:

1. **vLLM's paged attention in CUDA**: Fixed-size block pool shaped `(2, num_blocks, block_size * num_kv_heads * head_size)`, block sizes 8/16/32, block table `block_tables[seq, logical_block] -> physical_block_id`, kernel reads `physical_block_number = block_table[block_idx]` and computes K/V pointer offsets. Separate `forward_decode()` and `forward_prefix()` paths. Per Kwon et al. 2023 §§1, 3 and `paged_attn.py`, `attention_kernels.cuh`.

2. **MLX Swift has no paged attention primitive at the attention level**. `MLXFast.scaledDotProductAttention` is dense-only and the MLX Metal backend at `scaled_dot_product_attention.cpp:617` explicitly copies non-contiguous inputs to contiguous GPU buffers when stride/layout checks fail. **Consequence**: "gather K/V then call SDPA" dies because SDPA copies back. This rules out the naive paged-attention-by-gather approach.

3. **Hybrid model constraint**: AFM's `MambaCache` is literally `ArraysCache` and `ArraysCache.extend(other:)` at `KVCache.swift:1203` does `MLX.concatenated([c, o])` — same batch-dim concat bug as `BatchKVCacheSimple`, untouched by the slab work. vLLM's Jamba implements Mamba-specific cache interfaces separately from paged attention. SGLang does the same. **Hybrid is a two-cache problem, not a one-unified-paging problem.**

4. **Scope estimate for full paged attention**: 2-3 months (custom Metal kernel path), because the minimum meaningful unit is "block allocator + Metal decode kernel + model attention call site rewrite + Mamba non-concat path."

5. **Ranked alternatives** (highest expected benchmark impact first):
   - Fix Mamba concat path independently (highest leverage, ~450 → ~550-600 tok/s expected)
   - Unified slot-resident across both attention and Mamba with bulk admissions (~450 → ~650-700 tok/s, the only smaller-delta path with a realistic shot at the ~750 ceiling)
   - Pre-admission batching (small win, amortizes but doesn't remove)
   - Background concat overlap (unlikely to work — GPU is idle during stalls, nothing to overlap)
   - `takeAlong`/scatter gather + SDPA (do not do this — SDPA copies back)

**First-consultation verdict**: Not yet on paged attention. Do unified slot-resident first.

### 10.2 Second consultation — follow-up on missed MLX primitives (task `task-mnquk5ly-u95md3`, 2 min, model: `gpt-5.3-codex-spark`)

**Prompt**: First consultation missed three MLX primitives (`gatherMM`, `blockMaskedMM`, `segmented_mm`) that could implement paged attention without custom Metal kernels. Specifically asked about the SDPA-bypass path: replace `MLXFast.scaledDotProductAttention` entirely with `gatherMM(QK) + softmax + gatherMM(scoreV)`, so the "SDPA copies back" objection does not apply.

**Key findings**:

1. **`gatherMM` is genuinely fused**: Confirmed by source. `GatherMM::eval_gpu` routes through Metal kernels `gather_mm` / `gather_mm_rhs` / `gather_mv` (citation: `matmul.cpp:2158-2205`) which take index arrays and compute pointer remapping at launch time. No intermediate gathered operand buffer is materialized. Kernel code at `steel_gemm_gather.h:293-306` resolves `indx_A/indx_B` from the index arrays and performs GEMM with those pointers directly.

2. **Multi-head shape semantics work**: `rhsIndices` of shape `[B, H]` gives per-head per-sequence block selection. `[B]` broadcasts to "same block list for all heads per sequence" via the `broadcast_arrays` path at `ops.cpp:5504-5510`. Both shapes are viable.

3. **`segmented_mm` is the wrong primitive for paged attention**: 2D-only constraint (`ops.cpp:5535-5545`), output shape `segments.shape[:-1] + [a.shape(0), b.shape(1)]`, segment axis orthogonal to M rows (`steel_gemm_segmented.h:70-84`). One dispatch requires each segment to own a complete `[M, N]` matrix, which doesn't match per-sequence attention decoding. Would need extra dispatches or restructuring. **Secondary at best; not worth the 20-line Swift wrapper for this purpose.**

4. **Per-step overhead of `gatherMM`-based attention vs SDPA fast path**: Estimated **2-3×** at our scale (B=150, H=16, T_q=1, D=128, kv_len ~1000). Reason: SDPA fast path is one fused Metal kernel (citation: `scaled_dot_product_attention.cpp:590-597, 643-708`, `MLXFast.swift:77-84`) with in-kernel fp32 softmax. Replacement is a 3-kernel chain (gatherMM + mask/softmax + gatherMM) plus potential layout traffic. Labeled as hypothesis (no benchmark run). Not 1.1×; 5× only if additional copies dominate.

5. **Revised scope for full `gatherMM`-based paged attention** (Mamba separately fixed): **~3-5 weeks**. Week 1 wiring + index/layout plumbing, week 2 correctness/shape parity (masking, GQA/MQA), weeks 3-4 tests + perf smoke. Optional follow-up Metal fusion stage for perf parity with SDPA.

6. **Softmax fusion loss**: Modest numerical delta (A2 can still use `Softmax` with `precise=true`) but moderate latency delta from memory-bound pass-through and the extra kernel launches. Not catastrophic numerically, but not free.

**Second-consultation verdict**: `gatherMM`-based paged attention is **technically feasible in 3-5 weeks** (not 2-3 months) without custom Metal kernels. But the 2-3× per-step overhead means it is **net-neutral to net-negative vs unified slot-resident on this hardware**, because unified slot-resident eliminates the same merge-cliff without the per-step cost. **Recommendation: stay on unified slot-resident; keep gatherMM paged attention as stage-2 optionality only if unified slot-resident leaves residual attention stalls.**

### 10.3 Combined verdict and decision

Both consultations converge: **the correct path for AFM on M3 Ultra is unified slot-resident with bulk cohort admissions, not paged attention**. Paged attention's two canonical wins — memory efficiency and O(1) merge — map differently on this hardware:

- **Memory efficiency**: zero value on a 512 GB unified-memory machine with no pressure.
- **O(1) merge**: equally achievable by unified slot-resident without the 2-3× per-step SDPA-bypass cost.

The decision is recorded. See `docs/unified-slot-resident-plan.md` for the concrete implementation plan (to be written in the next work step).
