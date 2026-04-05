# TurboQuant Metal Decode Path

**Branch:** `feature/codex-turboquant-metal`
**Status:** Implemented

## Overview

This branch introduces the first inline Metal-backed execution path for TurboQuant in AFM. The focus is intentionally narrow: single-token serial decode over the packed MSE-style cache representation, with correctness-preserving fallback where the fully packed value path is not yet ready.

## Scope

- Add inline Metal shader dispatch for TurboQuant decode scoring
- Add TurboQuant serial decode fast-path scaffolding
- Fix decode-after-prompt-cache-restore history reconstruction
- Keep unsupported cases on safe fallback paths

## Main Design

### 1. Inline Metal kernels live in the vendored Swift patch layer

Following the `mlx-vlm` pattern, the branch uses inline Metal kernel source strings compiled through MLXFast rather than standalone `.metal` files.

Implemented in:
- [KVCache.swift](../Scripts/patches/KVCache.swift)

The current kernelized pieces are centered on MSE score computation for grouped decode queries.

### 2. Fast path is intentionally decode-only and narrow

The current fast path targets the safe subset:
- serial decode
- one-token query length
- grouped-query attention layouts where query heads can be folded over KV heads
- no unsupported mask complications

Unsupported or not-yet-finished cases fall back to the existing dense attention route.

### 3. Prompt-cache restore now preserves decode history

This branch fixes an important correctness gap: when a TurboQuant cache is restored from packed prompt-cache state, subsequent decode must rebuild prior history before appending new keys/values. Without that, the restored cache would behave like it only contained the newest token block.

### 4. Value accumulation stays conservative

The branch keeps a practical compromise:
- score-side work goes through the TurboQuant Metal path
- value-side accumulation stays on the safe route instead of claiming a fully packed-value Metal implementation prematurely

This avoids shipping a numerically unstable packed-value fast path while still exercising a real kernel-backed decode path.

## Files

- [Scripts/patches/KVCache.swift](../Scripts/patches/KVCache.swift)
- [Tests/MacLocalAPITests/TurboQuantCacheTests.swift](../Tests/MacLocalAPITests/TurboQuantCacheTests.swift)

## Current Behavior

- TurboQuant single-token decode uses the Metal-backed score path when the cache/query layout fits the supported shape
- multi-token prefill still uses fallback behavior
- restored packed caches can continue decoding without losing prior context
- numeric validation is bounded against the dequantized reference rather than claiming bit-exact equivalence

## What This Branch Does Not Do

- No fully packed value-side Metal weighted-sum path yet
- No fully fused end-to-end TurboQuant SDPA path
- No batch/concurrent TurboQuant cache execution
- No rotating/sliding-window TurboQuant support

## Why This Slice Exists

This branch is the first real execution-path branch in the stack. It proves that the packed TurboQuant representation can drive actual kernel-backed decode logic without forcing the rest of AFM onto an unfinished fused path.

## Validation

Validation on this branch covers:
- TurboQuant cache unit tests
- decode after packed cache restore
- KV cache truncate regressions
- batched prefill regressions unrelated to TurboQuant

## Next Branch

The next meaningful slice is batch/concurrent integration:
- batch cache compatibility
- scheduler support
- radix/prefix-cache format hardening
- broader API-path parity
