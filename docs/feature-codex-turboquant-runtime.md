# TurboQuant Runtime Selection

**Branch:** `feature/codex-turboquant-runtime`
**Status:** Implemented

## Overview

This branch exposes TurboQuant as a real runtime option in AFM. The key outcome is that TurboQuant can now be selected from the normal AFM startup flags and propagated through the serial generation path without needing model-specific hand wiring.

## Scope

- Accept TurboQuant-oriented KV flags from the CLI
- Auto-select TurboQuant for fractional KV bit-widths
- Allow explicit TurboQuant selection for integer bit-widths
- Propagate TurboQuant selection into serial generation parameters
- Apply `mlx-vlm`-style cache replacement rules

## Main Design

### 1. Selection is global server/model configuration

TurboQuant is enabled from AFM startup flags, not per-request. The relevant user-facing controls are:
- `--kv-bits`
- `--kv-quant-scheme`

Implemented in:
- [main.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/main.swift)

Rules:
- fractional `--kv-bits` auto-enables TurboQuant
- integer `--kv-bits` uses legacy uniform quantization unless `--kv-quant-scheme turboquant` is set
- missing `--kv-bits` keeps the dense path

### 2. GenerateParameters becomes the decision point

The branch makes `GenerateParameters` responsible for deciding whether a request path uses:
- dense KV cache
- uniform quantized KV cache
- TurboQuant KV cache

Implemented in:
- [Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)

This keeps cache selection logic in one place and lets the model service remain a caller rather than duplicating selection rules.

### 3. Cache replacement follows `mlx-vlm` compatibility rules

When TurboQuant is active, regular attention caches can be replaced or created as `TurboQuantKVCache`, while unsupported cache classes are left alone.

Implemented in:
- [KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

Current policy:
- `KVCacheSimple` can become `TurboQuantKVCache`
- `CacheList` is traversed recursively
- `RotatingKVCache` is skipped
- `ArraysCache` and `MambaCache`-style paths are skipped
- legacy uniform `QuantizedKVCache` remains its own path

### 4. Serial generation paths are wired

The model service now includes `kvBits` and `kvQuantScheme` in the generated MLX parameters used by both:
- serial non-streaming generation
- serial streaming generation

Implemented in:
- [MLXModelService.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/Models/MLXModelService.swift)

## Files

- [Sources/MacLocalAPI/main.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/main.swift)
- [Sources/MacLocalAPI/Models/MLXModelService.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Sources/MacLocalAPI/Models/MLXModelService.swift)
- [Scripts/patches/Evaluate.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/Evaluate.swift)
- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Tests/MacLocalAPITests/TurboQuantCacheTests.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Tests/MacLocalAPITests/TurboQuantCacheTests.swift)

## What This Branch Does Not Do

- No real TurboQuant attention math yet
- No packed-state cache representation yet
- No batch/concurrent TurboQuant path
- No request-level TurboQuant override

## Why This Slice Exists

This branch ensures TurboQuant can be selected, propagated, and instantiated cleanly before the attention implementation becomes more complicated. It separates "can AFM choose TurboQuant?" from "can AFM execute TurboQuant efficiently?"

## Validation

Validation focuses on:
- CLI parsing
- parameter selection behavior
- cache replacement rules
- prompt-cache metadata continuity

## Next Branch

`feature/codex-turboquant-attention` turns TurboQuant into an explicit attention dispatch path instead of just a different cache container.
