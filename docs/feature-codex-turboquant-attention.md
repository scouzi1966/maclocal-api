# TurboQuant Attention Dispatch

**Branch:** `feature/codex-turboquant-attention`
**Status:** Implemented

## Overview

This branch promotes TurboQuant from a cache-type placeholder into a first-class attention execution path. Instead of relying on generic dense attention code to treat TurboQuant like a normal cache, the attention stack can now detect TurboQuant caches and delegate decode/prefill handling to TurboQuant-specific methods.

## Scope

- Add TurboQuant-specific attention protocol methods
- Route attention through TurboQuant caches in the shared attention helper
- Keep the implementation safe by allowing fallback behavior internally

## Main Design

### 1. Attention dispatch is explicit

The core design change is that TurboQuant is no longer "just another cache" from the perspective of attention execution. The shared attention helper checks for a TurboQuant-capable cache and calls TurboQuant-specific methods:
- `decodeAttention(...)`
- `prefillAttention(...)`

Implemented in:
- [AttentionUtils.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/AttentionUtils.swift)

This mirrors `mlx-vlm`'s model-base dispatch pattern and avoids hiding TurboQuant behind generic dense attention calls.

### 2. Decode and prefill are distinct paths

The branch makes single-token decode and multi-token prefill separate TurboQuant entry points. That matters because decode and prefill have very different optimization opportunities and should not be forced through the same implementation contract.

Implemented in:
- [KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

### 3. Fallback remains legal

This branch does not claim a fully accelerated TurboQuant path yet. Instead, it makes TurboQuant dispatch explicit while allowing the underlying TurboQuant cache implementation to fall back to dense behavior until later branches land the real packed-state and Metal logic.

That keeps correctness and architecture separate:
- this branch solves dispatch
- later branches solve representation and performance

## Files

- [Scripts/patches/AttentionUtils.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/AttentionUtils.swift)
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Tests/MacLocalAPITests/TurboQuantCacheTests.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Tests/MacLocalAPITests/TurboQuantCacheTests.swift)

## What This Branch Does Not Do

- No packed TurboQuant state yet
- No inline Metal kernels yet
- No batch/concurrent TurboQuant batching support
- No prefix-cache hardening for mixed TurboQuant and non-TurboQuant formats

## Why This Slice Exists

Without an explicit attention dispatch hook, later optimization work would be forced into generic cache code or model-specific hacks. This branch creates the right architectural seam for later TurboQuant implementations.

## Validation

Validation checks that:
- single-token decode calls the TurboQuant decode path
- multi-token prefill calls the TurboQuant prefill path
- non-TurboQuant caches continue using their existing dispatch

## Next Branch

`feature/codex-turboquant-codecs` replaces the dense placeholder representation with packed TurboQuant state suitable for serialization and later fast-path execution.
