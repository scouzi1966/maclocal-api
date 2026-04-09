# Unified Slot-Resident KV Cache — Implementation Plan

Date: 2026-04-08
Status: Plan approved. Implementation not yet started.
Author note: Written for an AI coding agent executing this plan. Read `docs/merge-cliff-investigation.md` first for full context on why this is the chosen direction. This document is the concrete execution plan — the decision rationale lives in the investigation doc.

---

## 0. Prerequisites

Before touching any code, confirm:

- You have read `docs/merge-cliff-investigation.md`, especially §4.1 (decided direction), §10 (Codex consultations), and §8 (open questions).
- You understand why the previous slab attempt failed — the three failure modes listed in §2.2 of the investigation doc.
- You understand that paged attention has been evaluated and declined for this workload (§10.3 in the investigation doc). If you think paged attention is the right answer, re-read §10.2 first.
- The `slab-reference` branch exists at `9fe46ae`. Use `git show 9fe46ae:Scripts/patches/BatchKVCache.swift` to reference the prior slab code for `BatchKVCacheSimple` as a starting point. **Do not cherry-pick it wholesale** — the previous design had per-slot slice-assigns that were part of the failure.

Working branch: create a new branch from current `feature/batch-fixed-slab-kv` HEAD:

```bash
git checkout feature/batch-fixed-slab-kv
git checkout -b feature/unified-slot-resident-kv
```

---

## 1. Target Architecture

### 1.1 The protocol

Add a new protocol `SlabAddressable` to `Scripts/patches/BatchKVCache.swift` (or a new file `Scripts/patches/BatchKVSlab.swift` if you prefer to keep the scope visible). The protocol defines the bulk-cohort contract every slab-mode cache family must implement:

```
protocol SlabAddressable: BaseKVCache {
    // Physical capacity (fixed at init); logical active count in batchSize.
    var capacity: Int { get }
    var batchSize: Int { get }

    // Bulk-cohort admission: copy all rows from `other` into this slab's
    // rows [batchSize ..< batchSize + other.batchSize] in ONE slice-assign
    // per underlying tensor. Must not do per-row work.
    // Precondition: batchSize + other.batchSize <= capacity.
    // Precondition: other has the same capacity model (compatible layout).
    func addSlotsBulk(from other: Self)

    // In-place compaction. Moves trailing active rows into holes left by
    // removed slots. Does NOT reallocate the slab.
    // After this call, batchSize == keepIndices.count and active rows are
    // laid out as rows [0 ..< batchSize).
    func filter(_ keepIndices: [Int])
}
```

Where `BaseKVCache` is the existing mlx-swift-lm abstract base at `vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:125` (patched in `Scripts/patches/KVCache.swift:125`).

**Design constraints**:
- `addSlotsBulk` is the ONLY admission path in slab mode. Single-slot admission is prohibited — it falls back to the standard non-slab path if called. The scheduler must always batch cohort admissions.
- `addSlotsBulk` must do ≤ O(layers × 2) Metal dispatches per call (one K slice-assign + one V slice-assign per layer, or equivalent for non-KV caches). Zero per-row work.
- `filter` compacts in place via a single gather-and-write per layer, NOT per-row copies.
- All host-side metadata (leftPadding arrays, perSeqOffset arrays) must live in Swift memory, not MLXArray, so no implicit GPU sync on read.

### 1.2 Slab types to implement

| Class | Status | Scope |
|-------|:-------|-------|
| `BatchKVCacheSimple` | Partial prior attempt (reflog `9fe46ae`) | Redo with the `SlabAddressable` contract. Keep host-mirror arrays for `leftPadding`/`perSeqOffset`. Drop per-slot `addSlot` from the public surface — only `addSlotsBulk`. |
| `ArraysCache` / `MambaCache` | Not done | New slab mode. Simpler than `BatchKVCacheSimple`: no seqLen dim, no left padding, just a list of ≤ 2 tensors each with a batch dim. Pre-allocate at `[capacity, ...other dims...]`, track active `batchSize`, in-place write on `addSlotsBulk`, in-place compact on `filter`. |
| `BatchCacheList` | Not done | Delegates `addSlotsBulk` and `filter` to each sub-cache, which must be slab-mode. If any sub-cache is NOT slab-mode (e.g., `BatchRotatingKVCache`), the whole `BatchCacheList` falls back to non-slab mode for correctness. |
| `BatchRotatingKVCache` | Deferred to v2 | Only affects Gemma 4. If the benchmark target model uses it, escalate; otherwise defer. |

### 1.3 Scheduler integration

The only change to `BatchScheduler` is replacing the per-slot merge loop in `prefillBatch` with a single bulk-cohort merge call. The current code at `Sources/MacLocalAPI/Models/BatchScheduler.swift:1333`:

```
for i in 0..<B {
    mergeCacheIntoBatch(individualCache: perRequestCaches[i], modelState: ...)
}
```

Becomes (pseudocode):

```
if cacheMode == .batched && slabMode && allLayersAreSlabCompatible(batchCaches, newBatchCaches) {
    bulkMergeCohort(into: batchCaches, from: newBatchCaches)
} else {
    for i in 0..<B {
        mergeCacheIntoBatch(individualCache: perRequestCaches[i], modelState: ...)
    }
}
```

`bulkMergeCohort` iterates layers and calls `addSlotsBulk` on each. If the precondition fails (one layer can't do bulk), it does NOT fall back silently — it asserts, because this means the slab precondition was violated earlier in the ramp and we're already in an inconsistent state.

The existing `mergeCacheIntoBatch` path stays as the `--kv-cache standard` fallback. Slab mode is opt-in via CLI.

### 1.4 CLI flag

New AFM CLI flag on the `mlx` subcommand:

```
--kv-cache standard|slab     # default: standard
```

Location: `Sources/MacLocalAPI/main.swift` or wherever the `mlx` subcommand options are defined (grep for the existing `--concurrent` flag to find it).

Plumbing:
1. Parse at the CLI layer.
2. Pass to `MLXModelService.init(...)` (or similar constructor).
3. Store on `BatchScheduler` as a `let slabMode: Bool`.
4. Use in `prefillBatch` to gate the bulk path.
5. Use in cache construction sites — when `slabMode == true`, construct slab-capable cache variants for each layer.

Also add a lower-priority flag later:

```
--kv-max-seq-len N           # default: model-dependent, ~4096 for most
```

Default probably does not need exposing in v1 — pick a conservative default (4096) and make it configurable only if benchmarks show we're running out of headroom.

---

## 2. Milestones

Each milestone is independently testable and rollback-safe. Do them in order. Do not skip ahead.

### Milestone M0 — Instrumentation baseline (½ day)

Before any implementation work, add two diagnostic probes to quantify the current state:

1. **Mamba merge probe**: wall-time probe around `ArraysCache.extend(other:)` at `Scripts/patches/KVCache.swift:1203`. Gated on `AFM_DEBUG=1`. Threshold 10 ms. Logs `MAMBA EXTEND: B_before=N, wall=Xs`.

2. **BatchCacheList merge probe**: wall-time probe around `BatchCacheList.merge(...)` at `Scripts/patches/BatchKVCache.swift:1028`. Gated on `AFM_DEBUG=1`. Logs `BCL MERGE: subCount=N, B_before=M, wall=Xs`.

Run the benchmark once with `AFM_DEBUG=1` at `--concurrent 150` and record the current per-merge wall-time breakdown. Save the log as `/tmp/afm-baseline-m0-$(date +%Y%m%d_%H%M%S).log`.

**Validation**: the log shows `MAMBA EXTEND` and `BCL MERGE` entries at high B with wall times that sum to a meaningful fraction of the per-merge stall. Document the breakdown in `docs/merge-cliff-investigation.md` §8 Answered.

**Rollback**: revert the probe commit.

### Milestone M1 — `ArraysCache` / `MambaCache` slab mode (2-3 days)

Implement slab mode for the simpler cache family first. This validates the `SlabAddressable` protocol design on a simple type before tackling `BatchKVCacheSimple`.

Changes:
1. Add `capacity: Int` field to `ArraysCache` (default 0 = non-slab, >0 = slab mode).
2. Add slab-mode init: `init(size: Int, capacity: Int, protoShapes: [MLXShape])` that pre-allocates `cache[i]` at `[capacity, ...shape[1...]]` using the proto shapes as a template.
3. Add `addSlotsBulk(from other: ArraysCache)` that does ONE slice-assign per tensor: `cache[i]![batchSize..<batchSize+other.batchSize, ...] = other.cache[i]![0..<other.batchSize, ...]`.
4. Add in-place `filter(_ keepIndices: [Int])` that compacts via a single gather per tensor.
5. Adopt `SlabAddressable` protocol.

Wire into `BatchCacheList.merge(...)` — when sub-cache is `ArraysCache` AND slab mode is enabled, use the new slab-mode init and `addSlotsBulk` path instead of the concat-based merge.

**Validation**:
- Unit test: create two `ArraysCache` instances with slab mode, populate them, bulk-merge, verify the merged state via `state` getter matches the expected concatenation.
- Correctness test: run `python3 Scripts/feature-mlx-concurrent-batch/validate_responses.py 1 2 4 8` against a hybrid model (Qwen3.5-35B-A3B-4bit) with `--kv-cache slab`. Expect 32/32 pass.
- Perf test: run `AFM_DEBUG=1 ./Scripts/demo-concurrent-throughput.sh --concurrent 150 --kv-cache slab --max-tokens 1500 --skip-render --skip-html --skip-verify`. Grep for `MAMBA EXTEND` — entries should be absent (no more Mamba extend path) OR show wall < 10 ms (slab path is measurably cheap). LOOP GAP entries should NOT show `merge=yes` attribution to the Mamba path.

**Rollback**: revert the M1 commit; the slab-mode `ArraysCache` is purely additive and gated on capacity > 0.

**Gate**: if M1 alone doesn't materially change the benchmark (aggregate < 500 tok/s), STOP and re-check the instrumentation — the Mamba-dominates hypothesis may have been wrong, and M2 will also not help.

### Milestone M2 — `BatchKVCacheSimple` slab mode (3-4 days)

Redo the prior slab attempt, this time with the `SlabAddressable` protocol and no per-slot `addSlot` on the public surface.

Start from the reflog commit:

```bash
git show 9fe46ae:Scripts/patches/BatchKVCache.swift > /tmp/slab-9fe46ae.swift
# Study this as reference. Do NOT cherry-pick it.
```

Key changes vs the reflog version:
1. Remove `addSlot(keys:values:)` — bulk-only.
2. Keep `addSlotsBulk(from:)` — this was the fast path that required all-slab layers, now it is the only path and the precondition is guaranteed by `BatchCacheList` gating.
3. Keep host-mirror arrays (`hostLeftPadding`, `hostPerSeqOffset`).
4. Keep `allOffsetsEqual` / `zeroPadding` reading from host arrays.
5. Keep in-place `filter(_ keepIndices:)` with host-side compaction.
6. Adopt `SlabAddressable` protocol.

Wire into `BatchCacheList.merge(...)` and direct-construction paths the same way as M1.

**Validation**:
- Correctness: 32/32 known-answer tests on a PURE-ATTENTION model first (Qwen3-30B-Instruct-4bit if available, otherwise the smallest attention-only model in the test cache), with `--kv-cache slab`.
- Correctness: 32/32 known-answer tests on the hybrid Qwen3.5-35B-A3B-4bit, with `--kv-cache slab`. This tests the M1 + M2 combination.
- Perf test: `./Scripts/demo-concurrent-throughput.sh --concurrent 150 --kv-cache slab --max-tokens 1500`. Expected: POST-MERGE wall at B=127 drops from ~7 s to low single-digit seconds. Aggregate from ~450 tok/s to ~650-700 tok/s. LOOP GAP entries should mostly disappear.
- Comparison run: same command with `--kv-cache standard` (no other changes). Measure both aggregates and report the delta.

**Rollback**: revert the M2 commit.

### Milestone M3 — Bulk cohort admission in `prefillBatch` (1 day)

The first two milestones fix the cache types. This milestone fixes the scheduler loop that currently calls `mergeCacheIntoBatch` per-slot.

Change `Sources/MacLocalAPI/Models/BatchScheduler.swift:1333` from:

```swift
for i in 0..<B {
    mergeCacheIntoBatch(individualCache: perRequestCaches[i], modelState: i == 0 ? result.state : nil)
}
```

to:

```swift
if slabMode && canBulkMerge(batchCaches, prefillCaches) {
    bulkMergeCohort(into: &batchCaches, from: prefillCaches, state: result.state)
} else {
    for i in 0..<B {
        mergeCacheIntoBatch(individualCache: perRequestCaches[i], modelState: i == 0 ? result.state : nil)
    }
}
```

Where `bulkMergeCohort(into:from:state:)` is a new private method that iterates layers and calls `(dest as! SlabAddressable).addSlotsBulk(from: src as! SlabAddressable)` for each.

`canBulkMerge` checks that every layer of both caches conforms to `SlabAddressable` and has compatible capacity and shape.

**Validation**:
- Same as M2 perf test. Expected: an additional reduction in POST-MERGE wall, because a cohort of e.g. 36 admitted slots is now 1 Metal op per layer instead of 36 × 1 op per layer per slot.
- Debug log should show a new line `BULK MERGE COHORT: cohort_size=N, wall=Xs` at high B, with wall ≤ tens of ms.

**Rollback**: revert M3, the previous per-slot loop works fine with M1+M2 (just slightly slower on large cohorts).

### Milestone M4 — CLI flag + integration polish (1 day)

1. Add `--kv-cache standard|slab` flag to the AFM CLI.
2. Plumb through to `BatchScheduler.init(...)`.
3. Default to `standard`.
4. When `slab`, construct slab-capable cache variants and enable bulk merge.
5. Log the chosen mode at startup.

**Validation**:
- `afm mlx -m <model> --kv-cache slab --concurrent 150` logs the chosen mode.
- `afm mlx -m <model> --kv-cache standard --concurrent 150` logs the default and uses the old path.
- Both work correctly on the hybrid model and produce coherent output on a quick smoke test.

### Milestone M5 — Benchmark and document (½ day)

Run the full concurrent demo on both modes and produce a comparison:

```bash
# Baseline
pkill -f afm; sleep 2
AFM_DEBUG=1 ./Scripts/demo-concurrent-throughput.sh --concurrent 150 --ramp-step-users 2 --ramp-step-s 5 --hold 15 --cooldown 20 --max-tokens 1500 --kv-cache standard --skip-render --skip-html --skip-verify
cp Scripts/demo/out/trace.jsonl /tmp/standard-trace.jsonl
cp Scripts/demo/out/requests.jsonl /tmp/standard-requests.jsonl

# Slab
pkill -f afm; sleep 2
AFM_DEBUG=1 ./Scripts/demo-concurrent-throughput.sh --concurrent 150 --ramp-step-users 2 --ramp-step-s 5 --hold 15 --cooldown 20 --max-tokens 1500 --kv-cache slab --skip-render --skip-html --skip-verify
cp Scripts/demo/out/trace.jsonl /tmp/slab-trace.jsonl
cp Scripts/demo/out/requests.jsonl /tmp/slab-requests.jsonl
```

Compare:
- Aggregate throughput over run span (from requests.jsonl)
- Number of dips ≥ 2 s in each trace
- POST-MERGE wall distribution by B
- Per-sequence tg/s wall and pure

Write results into a new doc `docs/merge-cliff-results.md` with the before/after numbers and a chart (static matplotlib PNG, not MP4).

---

## 3. Failure modes and gates

If any of these triggers, STOP and escalate:

1. **M1 doesn't improve aggregate by at least 50 tok/s on the hybrid model.** Means the Mamba-dominates hypothesis was wrong. Re-instrument and re-diagnose before continuing.
2. **M2 breaks correctness** on a pure-attention model. Means the slab-mode `BatchKVCacheSimple` has a bug distinct from the reflog version. Compare against `git show 9fe46ae:Scripts/patches/BatchKVCache.swift` and look for the divergence.
3. **M2 + M1 combined doesn't reach 650 tok/s aggregate at B=150.** Either the per-step decode cost has another hidden bottleneck, or the hypothesis about the remaining gap is wrong. Add finer instrumentation and investigate.
4. **LOOP GAP entries still appear with `merge=yes` after M3 is merged.** Means `bulkMergeCohort` isn't firing — check `canBulkMerge` and the flag plumbing.

---

## 4. File inventory (what gets touched)

| File | Changes |
|------|---------|
| `Scripts/patches/BatchKVCache.swift` | Add `SlabAddressable` protocol. Extend `BatchKVCacheSimple` with slab mode. Extend `BatchCacheList` to delegate bulk merge. |
| `Scripts/patches/KVCache.swift` | Extend `ArraysCache` with slab mode. Add `MAMBA EXTEND` probe (M0). |
| `Sources/MacLocalAPI/Models/BatchScheduler.swift` | Add `slabMode` field. Add `bulkMergeCohort` helper. Wire into `prefillBatch` at line ~1333. |
| `Sources/MacLocalAPI/main.swift` (or wherever `mlx` subcommand lives) | Add `--kv-cache` flag. |
| `Sources/MacLocalAPI/Models/MLXModelService.swift` | Plumb `slabMode` through to `BatchScheduler`. |
| `docs/merge-cliff-investigation.md` | Update §8 with M0 findings. |
| `docs/unified-slot-resident-plan.md` | This file — update milestone status as work proceeds. |
| `docs/merge-cliff-results.md` (new) | M5 benchmark results and before/after comparison. |

---

## 5. Non-goals

These are explicitly out of scope for this work:

- Paged attention (declined, see investigation doc §10.3)
- `BatchRotatingKVCache` slab mode (deferred to v2 unless the benchmark target uses Gemma 4)
- `QuantizedKVCache` slab mode (not the target workload)
- `ChunkedKVCache` slab mode (chunked attention is a different feature)
- New MLX primitive wrappers (`mlx_segmented_mm`, etc. — declined, see §10.2)
- Custom Metal kernels (not needed for this approach)
- Rewriting any `MLXFast.scaledDotProductAttention` call sites in model code — unified slot-resident keeps the cache contiguous so SDPA is untouched
- Memory-efficiency optimizations (not a constraint on 512 GB)
- Prefix-cache integration changes (separate concern, lives in `Sources/MacLocalAPI/Models/RadixTreeCache.swift` and is outside this scope)

---

## 6. Validation matrix

Before declaring the work done:

| Test | Command | Pass criterion |
|------|---------|----------------|
| Known-answer, pure attention | `python3 Scripts/feature-mlx-concurrent-batch/validate_responses.py 1 2 4 8` against a pure-attention model with `--kv-cache slab` | 32/32 |
| Known-answer, hybrid | same command against Qwen3.5-35B-A3B-4bit with `--kv-cache slab` | 32/32 |
| Benchmark, slab | concurrent demo at B=150 with `--kv-cache slab` | Aggregate ≥ 650 tok/s, POST-MERGE p50 at B=127 ≤ 2.0 s |
| Benchmark, standard | same demo with `--kv-cache standard` | Unchanged from baseline (regression check) |
| Startup log | either mode | Shows `[MLXModelService] kvCache=slab/standard` |
| Build check | `swift build -c release` | Clean, no warnings from new code |

---

## 7. Rollback plan

Every milestone is independently revertible. The full rollback is:

```bash
git checkout feature/batch-fixed-slab-kv
git branch -D feature/unified-slot-resident-kv
```

The existing `feature/batch-fixed-slab-kv` baseline at `6a4179d` stays intact and is not affected by any of this work.

---

## 8. What the agent doing this work should produce

Expected commits, in order:

1. `chore(batch): M0 Mamba + BatchCacheList merge probes`
2. `feat(kv): M1 slab-mode ArraysCache/MambaCache for in-place bulk admission`
3. `feat(kv): M2 slab-mode BatchKVCacheSimple with SlabAddressable protocol`
4. `feat(batch): M3 bulkMergeCohort in prefillBatch`
5. `feat(cli): M4 --kv-cache flag plumbed to BatchScheduler`
6. `docs: M5 unified slot-resident benchmark results`

Final PR description should cite:
- The two Codex consultations (`task-mnqrq82s-8xd1sw` and `task-mnquk5ly-u95md3`)
- The before/after benchmark numbers from M5
- This plan doc as the source of the architecture
- Non-goals explicitly called out so reviewers don't ask "why not paged attention"
