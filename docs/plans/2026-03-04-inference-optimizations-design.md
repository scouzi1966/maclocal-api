# AFM Inference Optimizations Design

**Date:** 2026-03-04
**Branch:** `feature/inference-optimizations`
**Status:** Design

## Current State

AFM achieves 130 tok/s on Qwen3.5-35B-A3B-4bit (M3 Ultra), 29% above Python mlx-lm baseline. Profiling shows **24.5% bandwidth utilization** (198 GB/s of 819 GB/s) — the system is dispatch-limited, not compute-limited. Existing optimizations: fused QKV attention, silu_mul, GDN 4-projection, MLXFast.rmsNorm, single-slot prefix caching, quantized KV cache (`--kv-bits`).

## Optimization Areas

### 1. Radix Tree Multi-Slot Prefix Cache

**Problem:** Current `PromptCacheBox` is single-slot — only one prompt prefix is cached. Multi-turn conversations and multiple users sharing the same system prompt get no benefit.

**Solution:** Replace `PromptCacheBox` with a radix tree (compressed trie) managing KV cache blocks, following SGLang's RadixAttention design.

**How it works:**
- Token sequences are stored as variable-length edges in a radix tree
- Each node points to paged KV cache blocks in GPU memory
- New requests match against the tree to find the longest cached prefix
- Generated tokens are also inserted into the tree (subsequent requests referencing prior output benefit)
- LRU eviction removes least-recently-used leaves when memory is full
- Cache-aware scheduling prioritizes requests with high cache hit rates

**Impact:** vllm-mlx measured 5.8x TTFT speedup with prefix caching. For multi-turn chat and shared system prompts, this is the single biggest TTFT improvement.

**Effort:** Medium. CPU-side data structure managing GPU-side KV pages. No Metal kernel work. Requires refactoring `PromptCacheBox` into a tree structure and adding block-level KV cache management.

**Implementation approach:**
- Define `KVCacheBlock` struct (fixed-size, e.g., 64 tokens of KV data)
- Build `RadixTree<KVCacheBlock>` with insert/match/evict operations
- Block pool allocator with LRU eviction policy
- Wire into `MLXModelService.generateStreaming()` — on cache hit, skip prefill for matched prefix tokens
- Expose stats via debug logging (`[PrefixCache] hit=384 miss=128`)

### 2. Continuous Batching

**Problem:** `SerialAccessContainer` enforces a mutex around entire generation. Requests are serialized — one user blocks all others.

**Solution:** Replace the serial lock with a continuous batching scheduler that processes multiple sequences per forward pass.

**How it works:**
- A scheduler loop runs every iteration, maintaining a batch of active sequences
- Each iteration: run one forward pass for all sequences in the batch
- Completed sequences are evicted immediately; new requests are inserted without waiting
- Prefill and decode can be mixed in the same batch step (chunked prefill)
- Attention kernels handle per-request masking within the flattened batch

**Impact:** vllm-mlx achieved 4.3x aggregate throughput at 16 concurrent requests. This is the biggest throughput multiplier for serving scenarios.

**Effort:** High. Major architectural change:
- Replace `SerialAccessContainer` with `BatchScheduler`
- Implement per-sequence KV cache management (paged attention)
- Modify the token generation loop to process batched forward passes
- Handle per-sequence stop conditions, tool call detection, and streaming
- Chunked prefill to prevent long prompts from blocking decode

**Dependencies:** Benefits enormously from Radix Tree prefix cache (shared prefix blocks across batch).

**Implementation phases:**
1. **Phase 1:** Paged KV cache manager (block allocation, per-sequence page tables)
2. **Phase 2:** Batch scheduler (request queue, iteration-level scheduling)
3. **Phase 3:** Batched forward pass (attention with per-request masks)
4. **Phase 4:** Chunked prefill (interleave prefill chunks with decode steps)

### ~~3. Speculative Decoding (Medusa-Style)~~ — DEFERRED

Removed from scope. Medusa heads require per-model training/sourcing, creating an ongoing maintenance burden as new models are added. Revisit if a model-agnostic speculative decoding approach emerges.

### 3. Compressed FSM for Structured Output

**Problem:** Current `response_format: json_schema` uses prompt injection — not guaranteed to produce valid JSON.

**Solution:** Compile JSON schemas into finite state machines (FSMs), mask logits at each step to only allow valid tokens. Compressed FSMs skip deterministic sequences (multiple tokens in one step).

**How it works:**
1. JSON schema → FSM at request time (cached per schema)
2. Each FSM state maps to a set of valid token IDs
3. Before sampling, mask logits to zero out invalid tokens
4. When the FSM has a deterministic transition (only one valid next token), emit it without sampling
5. Compressed FSM: chain deterministic transitions into multi-token jumps

**Impact:** Guaranteed valid JSON output. SGLang reports 3x faster JSON generation via compressed FSMs.

**Effort:** Low-Medium. CPU-side logit masking, no Metal work.
- JSON schema → FSM compiler (or integrate `outlines-core` / `llguidance`)
- Logit mask application in the sampling chain (before temperature/top_p)
- Schema cache (LRU, keyed by schema hash)
- Support `json_object` mode (any valid JSON) and `json_schema` mode (specific schema)

### 4. KV Cache Eviction (H2O / StreamingLLM)

**Problem:** When conversations exceed `--max-model-len`, requests are rejected with `context_length_exceeded`. No graceful degradation.

**Solution:** Instead of hard rejection, evict low-importance KV entries to maintain a fixed-size cache while preserving the most important context.

**Two approaches:**
- **H2O (Heavy-Hitter Oracle):** Track cumulative attention scores per token. Keep top-K "heavy hitter" tokens + most recent N tokens. Evict the rest.
- **StreamingLLM:** Keep "attention sink" tokens (BOS, punctuation that consistently receive high attention) + sliding window of recent tokens. Simpler, fixed budget.

**Impact:** Enables theoretically infinite context with constant memory. Quality degrades gracefully rather than hard-failing.

**Effort:** Medium.
- Attention score accumulator (per-token running sum across decode steps)
- Eviction policy (H2O: top-K + recent window; StreamingLLM: sinks + window)
- KV cache compaction (remove evicted entries, reindex)
- CLI flag: `--kv-eviction h2o|streaming|none` (default: none, preserving current behavior)

**Note:** PyramidKV (layer-aware compression with different ratios per layer) is more sophisticated but substantially harder to implement.

## Priority Order

1. **Radix Tree Prefix Cache** — highest bang-for-buck, medium effort, big TTFT improvement
2. **Compressed FSM** — low effort, solves a real correctness problem (invalid JSON)
3. **KV Cache Eviction** — medium effort, enables long conversations gracefully
4. **Continuous Batching** — highest effort, highest impact for multi-user, do last

## What NOT to Pursue

- **FP8 KV cache:** No Apple Silicon hardware support
- **Apple Neural Engine (ANE):** 3x slower than GPU for matmul, 2-4x CoreML overhead
- **Tensor parallelism:** Single-device only; irrelevant
- **AWQ/GPTQ:** CUDA-dependent; MLX-native quantization is faster on Mac
- **Additional Metal kernel fusions:** Diminishing returns — already at 24.5% bandwidth util, problem is dispatch overhead not kernel efficiency. `mx.compile` is the better path for reducing dispatches.

## References

- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE-3 Speculative Decoding](https://github.com/SafeAILab/EAGLE)
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [vllm-mlx: Apple Silicon Serving](https://github.com/waybarrios/vllm-mlx)
- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- [KIVI: KV Cache Quantization](https://arxiv.org/abs/2402.02750)
- [KVSplit: Asymmetric KV on Apple Silicon](https://github.com/dipampaul17/KVSplit)
