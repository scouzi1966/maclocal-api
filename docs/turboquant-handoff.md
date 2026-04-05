# TurboQuant Implementation Handoff

**Date:** 2026-04-04
**Branch:** `feature/codex-turboquant-fastpath`
**Status:** In progress

## Purpose

This document is the handoff note for future engineering work on TurboQuant in AFM. It is written so another coding agent can resume the work without reconstructing the prior planning, branch structure, or current support boundaries from git history alone.

It covers:
- the original implementation plan
- what has already been implemented
- what remains to be done
- which code paths are still uncovered
- the recommended next branches and review order

## Source References

The implementation direction so far is based primarily on `mlx-vlm`, with `vllm-turboquant` used as a secondary systems reference.

Primary reference implementation:
- `mlx_vlm/turboquant.py`
- `mlx_vlm/generate.py`
- `mlx_vlm/models/base.py`
- `mlx_vlm/tests/test_turboquant.py`

Secondary reference:
- `mitkox/vllm-turboquant`

Important design decision already made:
- follow the **`mlx-vlm` runtime-cache approach**
- do **not** adopt the `vllm` offline metadata-artifact flow as the primary design unless a later need appears

## Original Plan

The original TurboQuant rollout was split into these conceptual phases:

1. Add vendored MLX-side TurboQuant cache/config scaffolding
2. Wire TurboQuant selection through AFM runtime and serial generation paths
3. Add explicit TurboQuant attention dispatch instead of hiding behind dense cache behavior
4. Replace dense serialized cache state with packed TurboQuant state
5. Add inline Metal-backed decode fast paths
6. Add batch/concurrent scheduler support
7. Harden prefix/radix cache behavior across mixed cache formats
8. Expand API/runtime coverage and performance validation

This was later grouped into three reviewable stacked PRs:

- PR 1: plumbing
- PR 2: core serial TurboQuant behavior
- PR 3: initial Metal fast path

## Current Branch / PR Map

Fine-grained implementation branches:

- `feature/codex-turboquant-vendor`
- `feature/codex-turboquant-runtime`
- `feature/codex-turboquant-attention`
- `feature/codex-turboquant-codecs`
- `feature/codex-turboquant-metal`

Grouped review branches:

- `feature/codex-turboquant-plumbing`
  - PR [#89](https://github.com/scouzi1966/maclocal-api/pull/89)
- `feature/codex-turboquant-core`
  - PR [#90](https://github.com/scouzi1966/maclocal-api/pull/90)
- `feature/codex-turboquant-fastpath`
  - PR [#91](https://github.com/scouzi1966/maclocal-api/pull/91)

## What Has Been Done

### 1. Vendor and config scaffolding

Implemented:
- TurboQuant cache/config enums and structs
- TurboQuant metadata structs
- prompt-cache identity support
- TurboQuant-aware cache factory plumbing

Main files:
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Scripts/patches/Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)
- [Scripts/patches/LanguageModel.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/LanguageModel.swift)

Design note:
- [feature-codex-turboquant-vendor.md](/Volumes/edata/codex/dev/git/apr3/maclocal-api/docs/feature-codex-turboquant-vendor.md)
- [feature-codex-turboquant-runtime.md](/Volumes/edata/codex/dev/git/apr3/maclocal-api/docs/feature-codex-turboquant-runtime.md)

### 2. Runtime selection in AFM

Implemented:
- `--kv-bits`
- `--kv-quant-scheme`
- fractional `kvBits` auto-enable TurboQuant
- integer `kvBits` can force TurboQuant when scheme is `turboquant`
- serial `generate(...)` and `generateStreaming(...)` propagate the settings

Main files:
- [Sources/MacLocalAPI/main.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/main.swift)
- [Sources/MacLocalAPI/Models/MLXModelService.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/Models/MLXModelService.swift)
- [Scripts/patches/Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)

### 3. Explicit attention dispatch

Implemented:
- TurboQuant-specific attention protocol methods
- decode vs prefill dispatch split
- shared attention helper checks for TurboQuant first

Main files:
- [Scripts/patches/AttentionUtils.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/AttentionUtils.swift)
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

Design note:
- [feature-codex-turboquant-attention.md](/Volumes/edata/codex/dev/git/apr3/maclocal-api/docs/feature-codex-turboquant-attention.md)

### 4. Packed codec state

Implemented:
- packed MSE-style state
- key/value asymmetric bit split for fractional settings
- prompt-cache serialization for packed state
- dense shadow buffers remain runtime-only

Main file:
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

Design note:
- [feature-codex-turboquant-codecs.md](/Volumes/edata/codex/dev/git/apr3/maclocal-api/docs/feature-codex-turboquant-codecs.md)

### 5. Initial Metal decode fast path

Implemented:
- inline Metal kernel manager/scaffolding in Swift
- serial decode score-side fast path
- grouped-query decode reshaping
- restore-time shadow rehydration so prompt-cache reload preserves history

Important current limitation:
- the current fast path keeps a **safe value-side route**
- it does **not** yet claim a fully packed value-side Metal weighted-sum implementation

Main file:
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

Design note:
- [feature-codex-turboquant-metal.md](/Volumes/edata/codex/dev/git/apr3/maclocal-api/docs/feature-codex-turboquant-metal.md)

## Current Working Behavior

### Supported today

- serial non-streaming generation
- serial streaming generation
- simple/single-request MLX path
- `KVCacheSimple` replacement with `TurboQuantKVCache`
- recursive TurboQuant conversion inside `CacheList` when subentries are standard attention caches
- prompt-cache save/load for TurboQuant caches

### Partially supported today

- mixed-cache architectures where only some layers are standard attention caches
  - attention subcaches can use TurboQuant
  - non-attention or sliding-window subcaches stay native

### Not supported today

- `RotatingKVCache`
- `ArraysCache`
- `MambaCache`
- `BatchKVCacheSimple`
- `BatchRotatingKVCache`
- `BatchCacheList`
- radix/prefix cache format hardening for mixed TurboQuant and non-TurboQuant reuse
- fully fused packed value-side Metal decode
- packed prefill fast path
- per-request API toggle for TurboQuant

## Code Paths Audited

### Covered

- `afm mlx` CLI flag parsing
- serial `generate(...)`
- serial `generateStreaming(...)`
- attention helper dispatch
- prompt-cache save/load at the cache layer

### Explicitly not finished

- [BatchScheduler.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/Models/BatchScheduler.swift)
- [BatchKVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/BatchKVCache.swift)
- [RadixTreeCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/Models/RadixTreeCache.swift)
- `/v1/batch/completions`
- `/v1/batches`
- concurrent request mode (`--concurrent`)
- tool-batching-specific behavior under concurrent mode
- radix cache persistence and invalidation rules for TurboQuant caches

### Important nuance

Because serial controller paths already route through the same model service, TurboQuant is indirectly present for:
- plain completions
- chat completions
- streaming and non-streaming serial API requests

But there is still no dedicated end-to-end API regression matrix proving full parity across:
- tool calling
- structured output
- logprobs
- stop sequences
- batch file processing

## Known Gaps

### 1. Batch / concurrent support

This is the biggest missing feature area.

What is missing:
- TurboQuant-aware batch cache equivalents
- per-slot TurboQuant cache extraction and merge
- scheduler compatibility with TurboQuant cache classes
- safe behavior with left padding and mixed cache types

Why it matters:
- without this, TurboQuant is effectively serial-only in AFM

Recommended next branch:
- `feature/codex-turboquant-batch`

### 2. Prefix/radix cache hardening

Current prompt-cache save/load works at the cache-object level, but AFM still lacks a clear format/version boundary for TurboQuant inside prefix/radix cache flows.

What is missing:
- cache-format/version tagging for TurboQuant entries
- invalidation on incompatible config changes
- mixed-cache restore rules when a reused prefix was created under a different quantization mode

Recommended next branch:
- `feature/codex-turboquant-prefix`

### 3. Fully packed value-side execution

The current Metal slice accelerates score-side work but keeps a safe route on the value side.

What is missing:
- packed value-side weighted-sum kernel path
- end-to-end packed decode path without dense fallback
- stronger numeric validation against `mlx-vlm` behavior

Recommended next branch:
- `feature/codex-turboquant-fused-decode`

### 4. Prefill fast path

Current prefill still falls back instead of using a real packed TurboQuant execution path.

What is missing:
- TurboQuant prefill kernel path
- chunked prefill compatibility
- validation across larger prompt blocks

Recommended next branch:
- `feature/codex-turboquant-prefill`

### 5. Sliding-window and hybrid cache coverage

Right now `RotatingKVCache` and `MambaCache`-style paths are intentionally skipped.

What is missing:
- explicit policy for sliding-window TurboQuant support vs permanent exclusion
- support or firm rejection rules for hybrid architectures in batch mode

Recommended next step:
- decide whether these are in-scope for AFM or should remain unsupported with explicit documentation

### 6. API and observability coverage

What is missing:
- effective TurboQuant mode surfaced in `/props` or `/v1/models`
- structured logging showing active cache mode and fallback decisions
- end-to-end API tests for streaming, tools, JSON schema, and logprobs with TurboQuant enabled

Recommended next branch:
- `feature/codex-turboquant-api-validation`

## Suggested Next Implementation Order

1. **Batch/concurrent support**
   - highest user-visible gap
   - needed before TurboQuant is useful in `--concurrent` serving mode

2. **Prefix/radix cache hardening**
   - needed so TurboQuant does not create subtle cache reuse bugs later

3. **Fully packed decode/value path**
   - performance work after the execution model is broader

4. **Prefill fast path**
   - important, but easier to judge once batch and cache semantics are settled

5. **API/benchmark/observability polish**
   - after the functional surface is stable

## Recommended Branch Plan From Here

- `feature/codex-turboquant-batch`
  - BatchScheduler
  - BatchKVCache
  - concurrent-mode generation
  - `/v1/batch/completions` validation

- `feature/codex-turboquant-prefix`
  - prefix/radix cache format/versioning
  - invalidation rules
  - prompt-cache and radix compatibility checks

- `feature/codex-turboquant-fused`
  - value-side packed Metal path
  - fused decode improvements
  - prefill kernel experimentation if still appropriate here

- `feature/codex-turboquant-validation`
  - API matrix
  - performance comparison
  - observability and diagnostics

## Testing Guidance For The Next Agent

Minimum regression set already used on the current stack:

```bash
MACAFM_MLX_METALLIB="$PWD/default.metallib" swift test --filter TurboQuantCacheTests --parallel --num-workers 1
MACAFM_MLX_METALLIB="$PWD/default.metallib" swift test --filter 'KVCacheTruncateTests|BatchedPrefillTests' --parallel --num-workers 1
```

What should be added next:

- targeted `BatchScheduler` tests with TurboQuant caches
- prefix-cache save/restore tests mixing TurboQuant and dense cache modes
- controller-level streaming and non-streaming tests with TurboQuant enabled
- explicit API tests for:
  - tools
  - structured output
  - stop sequences
  - logprobs
  - batch file path

## Practical Resume Notes

If resuming from this repo state:

1. Start from `feature/codex-turboquant-fastpath`
2. Read the five branch docs plus this handoff note
3. Treat PRs `#89`, `#90`, and `#91` as the current stacked review structure
4. Do not edit vendor submodules directly; continue using:
   - [Scripts/patches](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches)
   - [Scripts/apply-mlx-patches.sh](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/apply-mlx-patches.sh)
5. Re-apply patches before concluding a test run if only patch-source files changed

## Current Bottom Line

AFM now has:
- TurboQuant selection
- serial cache creation
- explicit TurboQuant attention dispatch
- packed cache serialization
- a first Metal-backed decode slice

AFM does **not** yet have:
- full-engine TurboQuant coverage
- concurrent/batch TurboQuant support
- radix-cache-safe TurboQuant reuse
- full packed-value execution

That is the real boundary of the current implementation.
