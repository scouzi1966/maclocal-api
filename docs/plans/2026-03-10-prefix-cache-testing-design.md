# Prefix Cache Testing Design

**Date:** 2026-03-10
**Status:** Approved
**Issue:** Prefix cache works for Qwen3-Coder-Next-4bit (2355+ token hits) but fails for Qwen3.5-35B-A3B-4bit (3-206 token hits). No crash, but zero performance benefit for the 35B model.

## Problem

The radix tree prefix cache (`RadixTreeCache`) stores KV cache state keyed by token sequences. On subsequent requests with shared prefixes, it skips prefill for matched tokens. This works for Qwen3-Coder-Next but fails for Qwen3.5-35B-A3B — tokens diverge after 3 positions between consecutive requests with identical system prompts and tools. Current logging only reports hit counts, not WHY misses occur.

## Goals

1. Enhanced logging that self-diagnoses cache misses (divergence point, expected vs actual tokens)
2. Deterministic A/B test harness comparing cache ON vs OFF for realistic workloads
3. Assertion-based regression tests integrated into the test suite
4. Unit test coverage for RadixTreeCache data structure

## Deliverables

### 1. Enhanced `-vv` Prefix Cache Logging

**Files:** `MLXModelService.swift`, `RadixTreeCache.swift`

Every request logs:
- Code flow: streaming/non-streaming, useCache, radixCache state
- Input: total tokens, first 20 token IDs
- Lookup: tokens matched, divergence point (position + expected vs actual token ID)
- Restore: effectivePrefix, layer count, layer offsets after restore
- Insert: token count, layers stored, trim details
- Skip/invalidation: which branch and why

Key addition to `RadixTreeCache.findPrefix()`:
```
[PrefixCache] Radix traversal: matched 3 tokens, diverged at pos 3: input=2 vs cached=13455
```

### 2. Test Script: `Scripts/test-prefix-cache.sh`

Standalone script, also invokable from `test-assertions.sh`.

**For each model:**
1. Start afm with `-vv --enable-prefix-caching` → log to file
2. Run workload sequence (cache ON), capture timings from response JSON
3. Kill server
4. Start afm with `-vv` (no cache) → separate log
5. Run same workload sequence (cache OFF), capture timings
6. Kill server
7. Parse both logs, compare metrics, generate report

**Workloads (hardcoded request sequences):**

- **W1: OpenCode session** — Real OpenCode system prompt (~10K chars) + 11 tool schemas. 5 multi-turn requests: initial prompt, then growing conversation with user→assistant→tool→user turns appended. Simulates a coding session where the system prefix is constant.

- **W2: Identical repeats** — Same exact request body sent 10 times. Baseline sanity: should see near-100% cache hit rate after first request. Any miss indicates a bug.

- **W3: Growing conversation** — System prompt + tools, then 10 requests each appending one user+assistant turn. Tests that cache correctly extends and the growing suffix doesn't break prefix matching.

**Metrics per request (from response `timings` + afm log):**

| Field | Source |
|-------|--------|
| `prompt_n` | response JSON `timings.prompt_n` |
| `prompt_ms` | response JSON `timings.prompt_ms` |
| `predicted_n` | response JSON `timings.predicted_n` |
| `predicted_ms` | response JSON `timings.predicted_ms` |
| `cached_tokens` | afm log `Radix hit: N/M` |
| `first_20_tokens` | afm log `First 20 tokens: [...]` |
| `diverge_pos` | afm log `diverged at pos N` |
| `diverge_expected` | afm log `vs cached=N` |
| `diverge_actual` | afm log `input=N` |

**Output:** `test-reports/prefix-cache-TIMESTAMP/` containing:
- `results.jsonl` — per-request metrics for both cache ON and OFF
- `report.html` — summary tables and per-model comparison
- `afm-{model}-cache-on.log` / `afm-{model}-cache-off.log` — raw server logs

**Report summary table:**
```
| Model | Workload | Requests | Avg prompt_n (cache) | Avg prompt_n (no cache) | Tokens Saved | Time Saved | Hit Rate |
```

### 3. Assertions in `test-assertions.sh` (Section 14)

Gated by `--include-prefix-cache` flag (requires server start/stop per model).

Assertions:
- Request 1 always a miss: `cached_tokens == 0`
- W2 (identical repeats) requests 2-10: `cached_tokens > 0`
- W2 requests 2-10: `prompt_n(cache) < prompt_n(no_cache)`
- Qwen3-Coder-Next W1: `cached_tokens > 2000` for requests 2+
- Qwen3.5-35B-A3B W1: documented as known-broken (`cached_tokens < 200`), test marked XFAIL — becomes a regression test when fixed

### 4. Unit Tests for RadixTreeCache

Swift-based tests (or assertion script) testing the radix tree data structure with realistic-sized token arrays:

- Insert 10K token array, find exact match → hit 10K
- Insert 10K array, find 12K array sharing 10K prefix → hit 10K
- Insert 10K array, find array diverging at position 500 → hit 0 (no cached node before divergence)
- Edge split: two 10K arrays sharing 8K prefix → both findable
- LRU eviction with maxEntries=3, insert 4 → oldest evicted
- Growing conversation: insert [A], insert [A,B], insert [A,B,C] → find [A,B,C,D] hits on longest
- Invalidate all → all miss
- Layer state integrity: stored MLXArrays match retrieved ones
- Volume: 50 sequential inserts with growing arrays, verify no memory leak or tree corruption

### 5. Skill: `.claude/skills/test-prefix-cache/SKILL.md`

Documents:
- How to run: `./Scripts/test-prefix-cache.sh` or `./Scripts/test-assertions.sh --include-prefix-cache`
- What to look for in results
- How to interpret divergence logs
- Known issues (Qwen3.5-35B-A3B token divergence)
- How to add new workloads

## Architecture Notes

- Tests use `timings` from the response JSON (already returned by afm)
- Server logs parsed with grep/awk for cache-specific lines
- Each model tested independently (separate server start/stop)
- Cache OFF baseline uses the same binary, just without `--enable-prefix-caching`
- No code changes to the cache logic itself — only logging enhancements
- Model cache dir: `MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache`

## Non-Goals

- Fixing the Qwen3.5-35B-A3B token divergence (separate task, after testing reveals root cause)
- Changing the radix tree to single-slot (keep as-is)
- Multi-user/concurrent testing (afm is single-user serial)
