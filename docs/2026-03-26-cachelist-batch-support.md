# CacheList Batch Support in BatchScheduler

## Context

Models using `CacheList` (composite per-layer caches) crash or fall back to sequential prefill with `--concurrent`. Three models are affected: GLM-5 (glm_moe_dsa), FalconH1, BaichuanM1. Each uses `CacheList` to wrap 2 sub-caches per layer (e.g. MLA attention + DSA indexer, or MambaCache + KVCacheSimple).

The `prefillBatch()` validation at line 915 rejects CacheList because it's neither `KVCacheSimple` nor `ArraysCache`. Additionally, CacheList is missing several KVCache protocol implementations needed by the batch path.

## Affected Files

- `Sources/MacLocalAPI/Models/BatchScheduler.swift` — prefillBatch validation, cache creation, extraction
- `vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` (via `Scripts/patches/KVCache.swift`) — CacheList class
- `vendor/mlx-swift-lm/Libraries/MLXLMCommon/BatchKVCache.swift` (via `Scripts/patches/BatchKVCache.swift`) — may need BatchCacheList

## Affected Models

| Model | CacheList pattern | Sub-caches per layer |
|---|---|---|
| GLM5MoeDsa | `CacheList(KVCacheSimple(), KVCacheSimple())` | MLA attn [0] + DSA indexer [1] |
| FalconH1 | `CacheList(MambaCache(), KVCacheSimple())` | SSM state [0] + KV attn [1] |
| BaichuanM1 | `CacheList(MambaCache(), KVCacheSimple/RotatingKVCache)` | Conv state [0] + KV attn [1] |

## Plan

### Step 1: Fix CacheList missing protocol implementations

**File:** `Scripts/patches/KVCache.swift` (CacheList class, line ~1189)

Add:

- **`offset`** — return max offset across sub-caches: `caches.map { $0.offset }.max() ?? 0`
- **`truncateToOffset()`** — delegate to each sub-cache: `caches.forEach { $0.truncateToOffset() }`
- **`metaState`** — flatten sub-cache meta states: `caches.flatMap { $0.metaState }`

These are needed for the prefix cache save path in `finishSlot()` (lines 1173-1197).

### Step 2: Update prefillBatch validation

**File:** `Sources/MacLocalAPI/Models/BatchScheduler.swift`, line 915

Change:
```swift
let canBatch = templateCache.allSatisfy { $0 is KVCacheSimple || $0 is ArraysCache }
```
To:
```swift
let canBatch = templateCache.allSatisfy { $0 is KVCacheSimple || $0 is ArraysCache || $0 is CacheList }
```

### Step 3: Handle CacheList in prefillBatch cache creation

**File:** `Sources/MacLocalAPI/Models/BatchScheduler.swift`, lines 957-962

For each CacheList layer, create a `BatchCacheList` that wraps batched versions of each sub-cache:
```
CacheList(KVCacheSimple, KVCacheSimple)  →  BatchCacheList([BatchKVCacheSimple, BatchKVCacheSimple])
CacheList(MambaCache, KVCacheSimple)     →  BatchCacheList([MambaCache(leftPadding:), BatchKVCacheSimple])
```

### Step 4: Create BatchCacheList class

**File:** `Scripts/patches/BatchKVCache.swift` (new class)

A batched wrapper for CacheList that:
- Holds an array of batched sub-caches
- Forwards `update()` via fatalError (same as CacheList — model uses subscript)
- Provides subscript access to individual batched sub-caches
- Implements `innerState()` by flattening sub-caches
- Implements `state` get/set by delegating to sub-caches
- Provides `extract(index:) -> CacheList` — extracts per-sequence sub-caches and wraps in CacheList
- Provides `static merge([CacheList]) -> BatchCacheList` — merges per-sequence CacheLists into batched form

### Step 5: Handle CacheList extraction in prefillBatch

**File:** `Sources/MacLocalAPI/Models/BatchScheduler.swift`, lines 992-1010

Add a third branch for CacheList layers:
```swift
} else if let bclCache = cache as? BatchCacheList {
    for i in 0..<B {
        perRequestCaches[i].append(bclCache.extract(i))
    }
}
```

### Step 6: Handle CacheList in mergeCacheIntoBatch

**File:** `Sources/MacLocalAPI/Models/BatchScheduler.swift`

When merging individual CacheList caches into the batch, create BatchCacheList by merging each sub-cache position across sequences.

## Verification

1. **GLM-5 with --concurrent 6**: Should not crash on `[take]` error
2. **FalconH1 with --concurrent**: Should work if model is available in cache
3. **Qwen3.5-35B (no CacheList)**: Regression test — should still work as before
4. **Nemotron (MambaCache + KVCacheSimple, no CacheList)**: Should still work with SSM fix
5. Run ToolCall-15 benchmark with GLM-5 loaded
