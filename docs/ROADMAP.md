# AFM Roadmap

## Chunked Prefill (Token Budget per Decode Step)

**Status:** Planned
**Priority:** Medium-High
**Area:** Concurrent Batching / Latency
**Depends on:** Phase 2 dense batched decoding (`--concurrent`)

### Current Behavior

When `--concurrent` is enabled, `BatchScheduler.prefillPending()` runs each new request's prefill to completion before entering the decode loop. If 3 requests arrive at once with 2000-token prompts each, the 3rd request waits for requests 1 and 2 to fully prefill (~200ms each) before starting — stalling all active decode sequences during each prefill.

### Desired Behavior

Introduce a **token budget** per scheduler step (like vLLM's `--max-num-batched-tokens` / llama.cpp's `--batch-size`). Each step allocates a fixed token budget (e.g., 512) across:
1. **Decode tokens** from active sequences (1 token each, priority)
2. **Prefill chunks** from pending requests (remaining budget)

A 2000-token prompt would be prefilled across ~4 steps instead of blocking in one shot. Active sequences keep decoding every step with minimal latency impact.

### Why It Matters

- **Latency smoothing:** Active sequences don't freeze during new request prefill
- **TTFT fairness:** Multiple arriving requests get interleaved prefill rather than FIFO blocking
- **Scales with prompt length:** Without this, long system prompts (OpenCode sends ~4000 tokens) cause multi-hundred-ms stalls for all active generations
- **Industry standard:** vLLM (chunked prefill), llama.cpp (`--batch-size`), and SGLang all implement this

### Implementation Notes

- Add `--max-batch-tokens N` CLI flag (default 512)
- `prefillPending()` becomes `prefillChunk()` — processes up to N-activeSlots tokens of the next pending request
- Track partial prefill state: which request, how many tokens processed, intermediate KV cache
- Partial prefill cache can't be merged into batch until complete — keep separate
- Decode tokens get priority: budget = max(N - activeSlots, 0) available for prefill
- When prefill completes, merge into batch via existing `mergeCacheIntoBatch()`

### CLI Flag

```
--max-batch-tokens N    Token budget per decode step (default: 512, 0 = no chunking)
```

## Streaming Tool Call Arguments (Incremental `delta.tool_calls`)

**Status:** Planned
**Priority:** Medium
**Area:** Tool Calling / OpenAI Compatibility

### Current Behavior

afm buffers the entire tool call body until `</tool_call>` is received, runs post-processing (cross-parameter dedup, type coercion, key remapping), then emits the clean JSON arguments as a single `delta.tool_calls` chunk.

### Desired Behavior

Stream `function.arguments` incrementally token-by-token, matching OpenAI's behavior:

```
data: {"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"lo"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cati"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"on\":"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \"Paris"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}}
data: {"delta":{},"finish_reason":"tool_calls"}
```

### Why It Matters

- Clients that show progressive tool call construction (arguments "typed out") see them appear all at once instead of incrementally
- Better OpenAI API spec compliance
- Most clients (OpenCode, Vercel AI SDK) concatenate fragments and work fine either way, but strict spec compliance improves ecosystem compatibility

### Implementation Notes

- **Simple case:** Stream each `</parameter>` as a JSON key-value fragment as it closes (no buffering needed)
- **Complex case:** When cross-parameter dedup or key remapping is needed, must fall back to buffered mode
- Hybrid approach: stream params incrementally by default, buffer only when the model emits duplicate/conflicting parameters (detected heuristically)
- The XML→JSON translation is the core challenge — can't stream JSON fragments until the parameter name and value are known
- Grammar-constrained mode (`--enable-grammar-constraints`) could simplify this since the output is already well-formed

## Speculative Decoding

**Status:** Parked (analysis complete, not yet scheduled)
**Priority:** Medium
**Area:** Inference Performance / Latency
**Analysis:** `docs/roadmap/speculative-decoding-analysis.md`

### Summary

Speculative decoding generates K draft tokens cheaply, then verifies them against the target model in a single forward pass. Reduces wall-clock latency by accepting multiple tokens per step. All major frameworks implement this (llama.cpp, vLLM, mlx-lm). Feasible in AFM but constrained by the app's general-purpose, multi-tenant nature.

### Phased Approach

#### Phase 1: N-gram Prompt Lookup (Low effort)
- Zero memory overhead, no draft model, model-agnostic
- CPU-only integer array scan — fits in the `asyncEval()` pipeline overlap
- High value for OpenCode/OpenClaw (code editing has 70-90% token overlap with prompt)
- Falls back gracefully to standard decode when no match (zero overhead)
- `--speculation ngram` CLI flag, disabled by default
- Auto-disable when `maxConcurrent >= 2`
- llama.cpp has 5 n-gram variants (`ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, `ngram_cache`); start with `ngram_simple` (linear scan, no data structures)

#### Phase 2: External Draft Model (Medium effort)
- `--draft-model <id> --num-draft-tokens 3`
- Separate `ModelContainer` for draft model, vocabulary match validation at load
- Draft K tokens → target verifies K+1 in one forward pass → reject sample from `max(0, q-p)`
- Auto-disable at concurrency > 1
- Best targets: Mistral (dedicated 0.5B draft models exist), Qwen3.5 dense (0.8B/2B share 248K vocab)
- For hybrid architectures (Qwen3.5 GatedDeltaNet, NemotronH Mamba): requires `ArraysCache` state snapshot/rollback on rejection

#### Phase 3: MTP for Qwen3.5 (High effort)
- Use model's built-in MTP head (`mtp_num_hidden_layers: 1`) instead of separate draft model
- Stop stripping `mtp.*` weights during model loading (currently filtered in Qwen3_5MoE.swift, Qwen3Next.swift, DeepseekV3.swift, MiniMaxM2.swift)
- Implement MTP head in Swift (single transformer layer: fuse pre-norm hidden state + embed(predicted_token) → draft logits)
- Dense models only — MoE acceptance rates too low (~9-11% vs ~80% dense, per mlx-lm benchmarks)

### Key Constraints

- **Apple Silicon is bandwidth-bound:** Theoretical speedup ceiling lower than GPU servers. mlx-lm measured 1.52x (4-bit dense Qwen3.5) to 1.09x (MoE) on M4 Pro.
- **Multi-tenant tension:** Speculation is a single-request optimization. At batch≥2, per-request speculation competes for GPU time with other requests. Must auto-disable under load (vLLM confirms overhead exceeds benefit at high QPS).
- **Feature interaction:** Every streaming state machine (think extraction, tool call detection, stop sequences) builds state on tokens as they arrive. Speculative tokens must not stream or advance state machines until verified. Universal fix: buffer speculation output → verify → feed accepted tokens through state machines → burst-stream. Adds per-round latency but reduces total generation time.
- **Hybrid architecture rollback:** Qwen3.5 GatedDeltaNet and NemotronH Mamba layers store recurrent state in `ArraysCache`, not KV pairs. Rollback requires full array copy (not just offset pointer move). Cost proportional to `num_linear_layers × state_size`.
- **Sampling parameter interaction:** Full OpenAI sampler chain (temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty) must run identically on draft and target. Seed-based determinism becomes harder with variable-length accept/reject cycles.
- **MLX JIT overhead:** mlx-lm's `speculative_generate_step` has ~50% overhead vs standard `generate_step` even with 0 draft tokens (Issue #250) due to dynamic input shape resolution. May affect MLX Swift similarly.

### Not Planned

- **EAGLE/Medusa:** Require training custom draft heads per model (1-2 days on 8× GPU). No pre-trained heads for Qwen3.5 or Nemotron. Out of scope.
- **LayerSkip:** Requires specially trained checkpoints (only exist for Llama). `LanguageModel` protocol doesn't support early exit.
- **Always-on speculation:** Must be opt-in due to overhead at high concurrency and feature interaction complexity.
