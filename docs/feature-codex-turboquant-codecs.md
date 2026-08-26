# TurboQuant Packed Codec State

**Branch:** `feature/codex-turboquant-codecs`
**Status:** Implemented

## Overview

This branch replaces the earlier dense-only TurboQuant placeholder with a packed TurboQuant cache representation. The important design change is that serialized TurboQuant state is no longer just a disguised dense KV tensor; it now stores norms and packed low-bit index state in a form that matches the intended TurboQuant runtime model more closely.

## Scope

- Introduce packed MSE-style TurboQuant state
- Split fractional bit-widths asymmetrically across keys and values
- Preserve prompt-cache serialization and restore behavior for packed state
- Keep dense shadow state only as a runtime helper

## Main Design

### 1. Serialized state becomes packed source-of-truth

`TurboQuantKVCache` now stores:
- key norms
- key packed indices
- value norms
- value packed indices

instead of persisting dense key/value tensors as its main state.

Implemented in:
- [KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)

### 2. Fractional bits are split key/value asymmetrically

Following the `mlx-vlm` direction, fractional TurboQuant bit-widths are interpreted as:
- keys use `floor(bits)`
- values use `ceil(bits)`

This keeps both sides on simple integer low-bit codecs while letting values receive the higher effective precision where it tends to matter more.

### 3. Dense shadow buffers remain runtime-only

The branch intentionally keeps dense shadow keys/values in memory as a compatibility layer for the still-incomplete attention path. The packed state is the serialized truth; the dense state is just a runtime convenience until fused or semi-fused execution paths are fully landed.

### 4. Prompt-cache restore is now meaningful for TurboQuant

Because TurboQuant state is serialized in packed form, prompt-cache save/load now exercises a real TurboQuant round-trip rather than merely preserving a class tag around dense arrays.

## Files

- [Scripts/patches/KVCache.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Scripts/patches/KVCache.swift)
- [Tests/MacLocalAPITests/TurboQuantCacheTests.swift](/Volumes/edata/codex/dev/git/apr3/maclocal-api/Tests/MacLocalAPITests/TurboQuantCacheTests.swift)

## What This Branch Does Not Do

- No real Metal decode/prefill execution yet
- No value-side packed fast path
- No batch-aware TurboQuant cache classes
- No new public AFM request surface

## Why This Slice Exists

Attention fast paths and prompt-cache correctness both depend on a stable packed representation. This branch creates that representation before trying to optimize execution over it.

## Validation

Validation covers:
- prompt-cache save/load with TurboQuant identity preserved
- packed-state shape expectations
- dequantized round-trip back to dense tensors

## Next Branch

`feature/codex-turboquant-metal` adds the first real inline Metal-backed decode path on top of this packed-state representation.
