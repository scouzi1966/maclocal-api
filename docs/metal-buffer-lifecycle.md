# Metal Buffer Lifecycle in BatchScheduler

## Problem

MLX uses lazy graph evaluation — operations build a compute graph that is only materialized when explicitly requested via `MLX.eval()`. In the BatchScheduler decode loop, each step's `cache.update()` performs a slice assignment:

```swift
self.keys![.ellipsis, _idx ..< _idx + 1, 0...] = newKeys
```

Under MLX's copy-on-write semantics, this creates a **new MLXArray** backed by a new Metal buffer. The old array becomes garbage only if no other reference holds it — but the lazy compute graph retains references to all intermediate arrays until the graph is materialized.

### Scale of the problem

| Factor | Count |
|--------|-------|
| Layers per model | 40-60 (Qwen 3.5: 40, Gemma 4 31B: 60) |
| Arrays per layer per step | 2 (keys + values) |
| Metal buffers per step | 80-120 |
| macOS Metal allocation limit | 499,000 buffers |
| Steps until crash | ~4,000-6,200 |

At ~28 tok/s per request, a single long generation (4096 tokens) takes ~146s and creates ~490K intermediate buffers — right at the OS limit. With prefix cache saving additional arrays, the server crashes after 1-2 long requests.

### Symptoms

```
MLX/ErrorHandler.swift:343: Fatal error: [metal::malloc] Resource limit (499000) exceeded.
```

The crash is **not an out-of-memory error** — it occurs even on machines with 512 GB unified memory. It is a count limit on the number of Metal buffer objects the process can hold simultaneously.

## Solution

`BatchScheduler.generationLoop()` periodically materializes all cache arrays:

```swift
// Every 512 decode steps, collapse the lazy graph
if stepCount % 512 == 0 {
    MLX.eval(batchCaches.flatMap { $0.innerState() })
}
```

This forces MLX to:
1. Execute all pending GPU operations in the compute graph
2. Materialize the cache arrays into their final values
3. Release references to all intermediate arrays
4. Allow the Metal buffer allocator to reclaim freed buffers

### Why 512 steps?

| Interval | Buffer accumulation | Overhead | Notes |
|----------|-------------------|----------|-------|
| Every step | 0 | ~2ms/step (7% slowdown) | Defeats lazy graph benefits |
| Every 64 | ~7,680 | Negligible | Conservative but safe |
| **Every 512** | **~61,440** | **~0.1% overhead** | **Good balance — 8x headroom to 499K** |
| Every 1024 | ~122,880 | Minimal | Tight at B=15 with 60 layers |
| Never | Unbounded | None | Crashes after ~4,000 steps |

The 512-step interval keeps peak buffer count well under the 499K limit while adding negligible overhead (~2ms every 18 seconds at 28 tok/s).

### Correctness

The materialization call is semantically invisible to the model:
- Cache arrays hold the **same values** before and after materialization
- It only affects **when** the GPU executes accumulated operations
- The decode loop's `asyncEval(tokenArrays)` already materializes sampled tokens each step — the cache materialization complements this by also materializing the KV state

### Interaction with other cache operations

| Operation | Creates new arrays? | Covered by periodic materialization? |
|-----------|-------------------|--------------------------|
| `cache.update()` (decode) | Yes — slice assignment | Yes |
| `cache.update()` (prefill) | Yes — concatenation for growth | Partially — prefill does its own materialization |
| Prefix cache save | Yes — `contiguous()` snapshots | No — these are intentional long-lived copies |
| `Memory.clearCache()` | Frees MLX memory pool | Complementary — runs every 1024 tokens |
| `filter()` / `extend()` | Yes — index selection, concat | Yes |

### History

- **Discovered**: 2026-04-05, Gemma 4 31B (60 layers) with prefix caching crashed after 15 serial ToolCall-15 scenarios (~7,600 total decode steps)
- **Root cause**: `RotatingKVCache` and `KVCacheSimple` slice assignments accumulating 120 Metal buffers/step without materialization
- **Fix**: Periodic materialization every 512 steps in the decode loop
- **Verified**: Server survived 6 consecutive ToolCall-15 runs (serial + concurrent + batch) without restart — ~20,000+ decode steps
