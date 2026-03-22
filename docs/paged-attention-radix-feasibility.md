# Paged Attention Feasibility for Radix Prefix Cache

Date: 2026-03-22

## Summary

Implementing true paged attention on top of AFM's current radix prefix cache is feasible in principle, but not as a small incremental extension to the existing implementation.

The current radix cache is a prefix-state snapshot system. It stores:
- full token prefixes
- per-layer serialized cache arrays
- per-layer cache metadata

That is useful for restoring contiguous prompt state, but it is not a paged KV memory system.

True paged attention would require:
- fixed-size KV pages or blocks
- per-sequence page tables
- attention/cache readers that can consume non-contiguous KV storage
- shared page ownership across requests and cached prefixes
- page-level eviction and refcounting

AFM does not currently have those pieces.

## Current implementation

The current radix cache implementation in
[Sources/MacLocalAPI/Models/RadixTreeCache.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Sources/MacLocalAPI/Models/RadixTreeCache.swift)
stores `KVCacheEntry` values containing:

- `tokens: [Int]`
- `layerStates: [[MLXArray]]`
- `layerMetaStates: [[String]]`

This means each cached prefix stores serialized layer state snapshots, not references into a shared KV block pool.

The current cache API in
[Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Scripts/patches/KVCache.swift)
is also snapshot-oriented. The `KVCache` protocol exposes:

- `update(keys:values:)`
- `state`
- `metaState`
- `offset`
- `trim(_:)`
- `truncateToOffset()`

This API assumes each cache instance represents a contiguous logical prefix and can be restored from full arrays.

## Why paged attention is not a small patch

### 1. The current cache interface is contiguous-state oriented

There is no abstraction for:
- page handles
- block tables
- shared segments
- refcounted cache storage
- gathering KV from non-contiguous pages

So paged attention cannot be introduced just by changing `RadixTreeCache`.

### 2. Model execution paths are not page-table aware

The design note in
[Sources/MacLocalAPI/Models/RequestScheduler.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Sources/MacLocalAPI/Models/RequestScheduler.swift)
already states that true continuous batching requires:

- per-sequence KV cache management
- paged attention
- batch-aware model/runtime changes

The same limitation applies even for single-sequence paged attention. The attention path would need to consume page-table-backed KV state instead of simple contiguous tensors.

### 3. Current radix cache entries own snapshots, not shared physical blocks

Today:
- radix lookup finds the longest prefix
- AFM restores saved arrays back into per-request `KVCache` instances
- decode continues from restored contiguous state

There is no shared GPU-resident page pool underneath. That means current restore/save behavior is fundamentally different from a paged-attention design.

## What would count as "true" paged attention

A real paged-attention design for AFM would look like this:

- a fixed-size `KVPage` / `KVBlock` abstraction
- a `KVPagePool` allocator
- `PagedKVCache` instances that own logical page tables, not monolithic state arrays
- attention paths that read KV via page tables
- radix nodes that reference shared page chains for token prefixes
- LRU/refcount eviction at the page level

That would allow:
- sharing prefixes without materializing full copied states
- more efficient trim/extend behavior
- a foundation for future continuous batching

## What is feasible in the short term

### Option 1: true page-backed radix cache

This is the correct long-term architecture:

- radix tree indexes token prefixes
- each cached prefix points to shared KV page chains
- restore reuses page references instead of copying full arrays
- eviction operates on pages

This is feasible, but it is a large refactor.

### Option 2: chunked snapshot cache without true paged attention

AFM could experiment with storing cache state in fixed-size chunks while still materializing contiguous state before attention.

This may reduce some copy costs, but it is not true paged attention and would not provide the full benefits of vLLM/SGLang-style paged KV systems.

If the goal is specifically paged attention, this should be treated only as an intermediate optimization experiment, not the final design.

## Feasibility verdict

- Short term: not feasible as a small extension to the current radix cache
- Medium term: feasible with a major cache/runtime refactor
- Main risk: vendor MLX cache/model interfaces are still oriented around contiguous state snapshots
- Best first milestone: add a page-oriented cache abstraction below the current `KVCache` model before touching radix storage

## Recommended implementation order

### Phase 1: page-oriented cache abstraction

Introduce new concepts below `KVCache`, for example:

- `KVPage`
- `KVPagePool`
- `KVPageTable`
- `PagedKVCache`

This phase should solve:
- page allocation
- page append
- page trim
- page snapshot/reference semantics
- refcounting hooks

Do this first without radix integration.

### Phase 2: single-sequence paged cache correctness

Before mixing in radix or batching, prove that one sequence can:

- prefill into pages
- decode across appended pages
- trim safely
- continue generation correctly

This isolates paged attention correctness from cache sharing complexity.

### Phase 3: radix integration

Once page-backed caches work, update radix entries so they store:

- references to shared page chains
- token prefix ownership metadata

instead of raw `MLXArray` snapshots.

At that point, radix becomes a logical prefix index over shared physical KV storage.

### Phase 4: eviction and sharing policy

Add:
- page-level LRU
- refcounting
- safe shared-prefix eviction
- accounting and debug logging

### Phase 5: continuous batching

Only after the page-backed cache foundation exists should AFM attempt:
- per-sequence page tables in a batch
- mixed prefill/decode scheduling
- true continuous batching

## Recommendation

Do not try to add paged attention directly to the existing `RadixTreeCache` implementation.

The better plan is:
1. design and prototype a page-backed `KVCache` layer
2. prove single-sequence correctness
3. migrate radix entries to page references
4. then revisit batching

This keeps the work decomposed into:
- cache layout
- attention/runtime compatibility
- prefix sharing
- batching

instead of mixing all four at once.
