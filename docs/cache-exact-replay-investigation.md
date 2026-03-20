# Cache Exact Replay Investigation

Date: 2026-03-20

## Summary

AFM prompt prefix caching is working for normal partial-prefix reuse, including concurrent batching, but exact or near-exact replay is not currently safe on several models.

Confirmed:
- Partial-prefix replay works in normal safe mode.
- Exact full replay with `suffix_tokens=0` is invalid in the current vendor generate path.
- Near-exact replay with a tiny suffix can diverge even when the cache hit is structurally valid.
- This is not limited to one model family.

Main operational conclusion:
- Keep exact full replay bypassed in safe mode.
- Keep using partial-prefix caching as the production path.

## Models tested

The near-exact replay issue was reproduced on:
- `mlx-community/Qwen3.5-35B-A3B-4bit`
- `mlx-community/Qwen3.5-35B-A3B-4bit --no-think`
- `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`
- `mlx-community/gpt-oss-20b-MXFP4-Q8`

## Test prompt

The main deterministic repro prompt was:

```text
Reply with exactly WRENCH-LATTICE-9021 and nothing else.
```

Generation settings:
- `temperature=0`
- `seed=42`
- `stream=false`
- `max_tokens=32`

## Unsafe replay env

Safe mode keeps exact replay bypassed for recurrent-cache models.

Diagnostic mode:

```bash
AFM_PREFIX_CACHE_ALLOW_UNSAFE_EXACT_REPLAY=<suffix_len>
```

Behavior:
- unset: exact replay bypass remains active
- `=1`: leave a `suffix=1`
- `=8`: leave a `suffix=8`
- `=16`: leave a `suffix=16`
- `true` / `yes`: treated as `1`

Related debug env:

```bash
AFM_PREFIX_CACHE_TRACE_BOUNDARY=1
```

This logs the token IDs and decoded fragments around the replay split.

## What failed

### Zero suffix

When exact full replay was allowed as `cached_tokens == prompt_tokens`, AFM restored the cache and passed an empty prompt suffix into the vendor generate path.

Observed runtime failure:

```text
Fatal error: [reshape] Cannot infer the shape of an empty array
```

This showed that `suffix_tokens=0` is not currently a valid continuation contract for the `mlx-swift-lm` generate path.

### Near-exact replay

Leaving a tiny suffix avoids the empty-input crash, but it does not guarantee correctness.

Observed behavior on repeated identical requests:
- empty outputs
- truncated outputs
- control-token-like outputs
- wrong assistant text

Examples:
- Qwen no-think with `suffix=1`: empty output, `<end_code>`, or malformed assistant prefix
- Nemotron Nano with `suffix=1`: repeated empty output
- GPT-OSS 20B with `suffix=1`: broken control-token-like text

## Partial replay validation

A dedicated partial-replay validator compared:
- cached partial replay from a warmed prefix
- cold replay of the same full prompt in safe mode

Tested suffix windows:
- `1, 2, 4, 8, 16, 32`

In those partial-prefix tests, cached output matched the cold baseline.

Interpretation:
- restored state is good enough for normal partial reuse
- restored state is not equivalent enough for exact or near-exact continuation

## Boundary-sensitive behavior

The suffix-length sweep on Qwen no-think produced a more specific result:

- `suffix=32`: passed
- `suffix=12`: passed
- `suffix=8`: passed
- `suffix=16`: failed

This shows the issue is not simply "smaller suffix is worse". The exact replay boundary matters.

### Failing boundary: `suffix=16`

Logged split:

```text
prefix_tail_decoded=" WRENCH-LATTICE-9"
suffix_head_decoded="021 and nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```

Observed output:

```text
WRENCH-LATTICE-9
```

The replay boundary cut through the target answer itself.

### Passing boundary: `suffix=12`

Logged split:

```text
prefix_tail_decoded="ATTICE-9021 and"
suffix_head_decoded=" nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```

Observed output:

```text
WRENCH-LATTICE-9021
```

Here the full answer-bearing phrase was already on the cached side, and the replayed suffix only covered trailing instruction/template material.

## Practical interpretation

This appears to be a boundary-sensitive state-equivalence problem.

What is likely true:
- The restored cache state is good enough for shared-prefix continuation with a meaningful replay window.
- The restored state is not exact enough to safely resume when the replay boundary cuts through the prompt's final completion-shaping token span.

This is consistent with:
- model state being partially but not fully equivalent after restore
- exact continuation needing more state than AFM currently stores
- or vendor continuation semantics requiring a stronger handoff than cache state alone

## Current safe policy

Recommended production behavior:
- Keep `--enable-prefix-caching`.
- Keep exact replay bypassed in safe mode.
- Use partial-prefix reuse only.

Unsafe exact replay should remain diagnostic-only.

## Logging added during this investigation

Regular logs now include:
- `PrefixCache Prefill`
- `PrefixCache Save complete`
- `CacheProfile phase=restore`
- `CacheProfile phase=save`
- `STATS` with:
  - `cache: HIT cached/total suffix=n`
  - `cache: MISS suffix=n`

Boundary tracing can additionally print:
- `prefix_tail_ids`
- `suffix_head_ids`
- decoded prefix/suffix fragments

## Useful commands

Safe mode:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm \
  mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --no-think \
  --enable-prefix-caching \
  --port 9999
```

Unsafe replay with forced suffix:

```bash
AFM_PREFIX_CACHE_ALLOW_UNSAFE_EXACT_REPLAY=16 \
AFM_PREFIX_CACHE_TRACE_BOUNDARY=1 \
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm \
  mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --no-think \
  --enable-prefix-caching \
  --port 9999
```

Exact replay validator:

```bash
python3 Scripts/feature-mlx-concurrent-batch/validate_exact_replay.py \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt 'Reply with exactly WRENCH-LATTICE-9021 and nothing else.'
```

Partial replay validator:

```bash
python3 Scripts/feature-mlx-concurrent-batch/validate_partial_replay.py
```

Concurrent prefix-cache harness:

```bash
python3 Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py 8
```

## Status

As of this investigation:
- concurrent prefix caching is working
- partial replay is working
- exact full replay is intentionally bypassed
- near-exact replay remains unsafe and should not be enabled in production
