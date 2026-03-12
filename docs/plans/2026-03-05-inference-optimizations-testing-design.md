# Inference Optimizations Testing Design

**Date:** 2026-03-05
**Branch:** `feature/inference-optimizations`
**Status:** Design

## Scope

Tests for 3 optimization areas implemented on the feature branch:

1. **Radix Tree Prefix Cache** — always-on, replaces single-slot PromptCacheBox
2. **Structured Output (prompt injection fallback)** — json_schema via prompt injection; XGrammar C++ interop deferred
3. **KV Cache Eviction** — `--kv-eviction streaming` CLI flag (StreamingLLM)

XGrammar constrained decoding tests are **deferred** until the Python bridge is replaced with C/C++ interop. The Python bridge code remains on the branch but is not tested or shipped.

## Test Location

All tests extend the existing `Scripts/test-assertions.sh` suite, except KV eviction which stays in its own `Scripts/test-kv-eviction.sh` (requires server flags not available in the shared test-assertions server).

## Test Cases

### Section 8b: Prefix Cache (standard tier)

| # | Name | Assertion | Method |
|---|------|-----------|--------|
| 1 | Cache hit on shared prefix | `cached_tokens > 0` on 2nd request | Two requests with identical system prompt, different user message |
| 2 | Full cache hit on repeated request | `cached_tokens > 0` on 2nd request | Exact same request sent twice |
| 3 | Multi-slot caching | `cached_tokens > 0` on 4th request | Three different system prompts, then repeat 1st — radix tree should still hold it |
| 4 | Cache miss on new prefix | `cached_tokens == 0` | First request with a unique prefix |
| 5 | Streaming cache hit | `cached_tokens > 0` in streaming usage chunk | Two streaming requests with shared prefix |
| 6 | TTFT improvement on cache hit (full tier) | Cached TTFT < cold TTFT | Compare `prompt_time` of cold vs cached request |

### Section 8c: Structured Output (standard tier)

| # | Name | Assertion | Method |
|---|------|-----------|--------|
| 1 | json_schema produces valid JSON | Response parses as JSON matching schema | Send request with `response_format: json_schema`, validate `name` (string) and `age` (integer) fields |

### Section 8d: KV Eviction — standalone `test-kv-eviction.sh`

| # | Name | Tier | Assertion | Method |
|---|------|------|-----------|--------|
| 1 | Server starts with --kv-eviction streaming | smoke | Health endpoint responds | Start server with flag, curl /health |
| 2 | Basic generation works | smoke | Non-empty response or valid thinking | Send chat completion |
| 3 | Streaming generation works | smoke | SSE delivers content and [DONE] | Send streaming request |
| 4 | Invalid --kv-eviction value rejected | smoke | Non-zero exit code | Start with `--kv-eviction invalid` |
| 5 | Long prompt eviction (DEFERRED) | standard | 200 instead of 400 | Requires `--max-model-len` merge |
| 6 | Eviction preserves coherence (DEFERRED) | standard | Response is not garbage | Requires `--max-model-len` merge |

## Tier Assignment

| Test Group | smoke | standard | full |
|------------|-------|----------|------|
| Prefix cache hit (shared prefix) | | x | x |
| Prefix cache hit (exact repeat) | | x | x |
| Multi-slot caching | | x | x |
| Cache miss | | x | x |
| Streaming cache hit | | x | x |
| TTFT improvement | | | x |
| Structured output (prompt injection) | | x | x |
| KV eviction flag acceptance | x | x | x |
| KV eviction generation | x | x | x |

## `/test-macafm` Skill Update

Update the failure interpretation guide to map new test groups to code:

| Test Group | Code Location |
|------------|---------------|
| Cache | `RadixTreeCache.swift` (`findPrefix`, `insert`, `evictLRU`), `MLXModelService.swift` (radix cache wiring in `generate`/`generateStreaming`) |
| Structured | `response_format` prompt injection in `buildUserInput()`, `MLXModelService.swift` |
| Eviction | `KVCacheSimple.evictStreamingLLM` in `Scripts/patches/KVCache.swift`, `applyStreamingLLMEviction` in `MLXModelService.swift`, `--kv-eviction` flag in `main.swift` |

## What NOT to Test

- **XGrammar constrained decoding** — deferred until C++ interop replaces Python bridge
- **Continuous batching** — `RequestScheduler` is scaffolding, not wired in
- **Cross-branch features** — KV eviction enforcement tests deferred until `feature/max-model-len` merge
