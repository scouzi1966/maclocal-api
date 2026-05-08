# TurboQuant Vendor Scaffolding

**Branch:** `feature/codex-turboquant-vendor`
**Status:** Implemented

## Overview

This branch establishes the vendored MLX-side scaffolding needed for TurboQuant without exposing it broadly through AFM yet. The goal is to make TurboQuant a first-class cache format in the patched `mlx-swift-lm` layer so later branches can wire runtime selection, attention dispatch, and real Metal fast paths on top of a stable cache API.

For AFM, "vendor" means changes live in `Scripts/patches/` and are applied into `vendor/mlx-swift-lm` through `Scripts/apply-mlx-patches.sh`.

## Scope

- Add TurboQuant cache/config types to the vendored cache layer
- Add metadata and serialization support for TurboQuant caches
- Add a shared cache factory path that can instantiate TurboQuant caches
- Add focused unit coverage for cache identity and metadata behavior

## Main Design

### 1. TurboQuant becomes a cache format, not a one-off flag

The branch introduces explicit TurboQuant types in the patched cache layer so later code can reason about TurboQuant as a cache implementation, not just as a special interpretation of `kvBits`.

Core additions are in:
- [KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)

This includes:
- `KVCacheFormat`
- `KVQuantizationScheme`
- `TurboQuantVariant`
- `TurboQuantConfiguration`
- metadata artifact structs for future compatibility with offline metadata if needed

### 2. TurboQuant cache identity survives prompt-cache round trips

The branch makes prompt-cache save/load aware of TurboQuant cache identity and metadata. This is important because later branches depend on prompt-cache restores reconstructing the right cache class rather than silently reloading as a dense `KVCacheSimple`.

### 3. Cache creation is centralized

The branch extends the vendored cache factory path so models using the shared attention-cache helper can allocate TurboQuant-ready caches through one code path rather than per-model special cases.

Primary wiring:
- [LanguageModel.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/LanguageModel.swift)
- [KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

## Files

- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Scripts/patches/Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)
- [Scripts/patches/LanguageModel.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/LanguageModel.swift)
- [Scripts/apply-mlx-patches.sh](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/apply-mlx-patches.sh)
- [Tests/MacLocalAPITests/TurboQuantCacheTests.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Tests/MacLocalAPITests/TurboQuantCacheTests.swift)

## What This Branch Does Not Do

- No AFM CLI or request-surface enablement
- No actual TurboQuant attention path
- No Metal kernels
- No batch/concurrent integration
- No model-family-specific gating beyond cache construction

## Why This Slice Exists

This branch de-risks the rest of the stack by creating the cache/config surface first. Later branches can then focus separately on:
- selection and runtime wiring
- attention dispatch
- packed-state codecs
- Metal execution paths

## Validation

Primary validation is unit-test based:
- TurboQuant metadata round-trip
- prompt-cache identity preservation
- cache factory behavior

## Next Branch

`feature/codex-turboquant-runtime` builds on this branch by wiring TurboQuant selection into AFM runtime configuration and generation parameters.
