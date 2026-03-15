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
