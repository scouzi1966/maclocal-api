# Speculative Decoding in AFM: Feasibility Analysis

## Executive Summary

Speculative decoding is **architecturally feasible but faces significant constraints** in AFM due to three factors: (1) Apple Silicon's memory-bandwidth-bound nature limits the theoretical speedup ceiling, (2) the hybrid attention architectures (GatedDeltaNet) in target model families like Qwen3.5 introduce state rollback complexity absent in pure-transformer models, and (3) the app's general-purpose nature (serving arbitrary models with arbitrary sampling parameters) reduces the acceptance rates that make speculation worthwhile. Of the approaches surveyed, **external draft model speculation** and **n-gram/prompt-lookup speculation** are the most practical paths. MTP (native multi-token prediction) is a high-value future option but requires implementing architecture support that is currently explicitly stripped during model loading.

---

## 1. AFM's Current Generation Architecture

AFM has two generation paths:

| Path | Entry Point | Concurrency | KV Cache |
|------|-------------|-------------|----------|
| **Serial** | `container.perform {}` | Mutex — one request at a time | `KVCacheSimple` per layer |
| **Batched** | `BatchScheduler` actor | 2-8 concurrent requests | `BatchKVCacheSimple` (left-padded) |

**Token generation loop** (both paths):
1. Forward pass: `model([B, seq_len])` → `logits [B, 1, vocab]`
2. Logit processor chain: repetition penalty → presence penalty → top_k → min_p
3. Sampling: temperature scaling → softmax → categorical/argmax
4. `asyncEval()` for GPU pipeline overlap
5. Detokenize + yield to stream

**Critical details for speculation:**
- The `LanguageModel` protocol accepts `(tokens, cache, state)` and returns `LMOutput(logits, state)`
- Forward pass shape is fixed: `[B, N]` input, `[B, 1, V]` output during decode
- KV caches are per-layer arrays. For Qwen3.5 hybrid models, linear attention layers use `ArraysCache` (recurrent state: conv1d + GatedDeltaNet state), while full attention layers use `KVCacheSimple`
- The `asyncEval()` call provides CPU-GPU overlap but **not** multi-model parallelism

---

## 2. How Reference Implementations Work

### 2.1 llama.cpp

Pluggable `common_speculative` interface with multiple backends:

| Mode | Description |
|------|-------------|
| `draft` | Neural draft model (smaller LLM) |
| `eagle3` | EAGLE-3 feature-level autoregression |
| `ngram_simple` | Basic n-gram lookup from prompt context |
| `ngram_map_k` | Key-only mapping for faster n-gram matching |
| `ngram_mod` | Hash-based n-gram with occupancy monitoring |
| `ngram_cache` | Pre-computed static/dynamic lookup tables |

**Draft-verify loop:**
1. Draft model generates K candidate tokens
2. Target model evaluates all K+1 in a single batch forward pass
3. Sequential acceptance with rejection sampling from `max(0, q-p)` distribution
4. KV cache trim on rejection via `llama_memory_seq_rm()`
5. Minimum guarantee: at least one token accepted per iteration

**Tree-based variants** randomize traversal order (mt19937) to preserve output distribution. On mismatch, system samples from residual distribution `max(0, q - p)` normalized, following the SpecInfer algorithm.

CLI flags: `--draft-max 16 --draft-min 5 --draft-p-min 0.9`

### 2.2 vLLM

Integrates with continuous batching via separate draft/target runners. Supports external draft model, prompt lookup, Medusa, EAGLE, and MLPSpeculator.

Key characteristics:
- Batch-level speculation: all requests in a continuous batch are speculated simultaneously
- At **high QPS, overhead exceeds benefit** — speculation helps most at low concurrency
- Draft models can use different tensor parallel sizes than target models
- Dynamic adjustment planned: auto-shorten speculation under heavy load

### 2.3 SGLang

Deep EAGLE integration (1/2/3) as primary speculation method:
- Draft and target models share embeddings and LM head to reduce memory
- Tree verification with custom attention masks
- CUDA graph optimization for draft/extend phases
- Up to 6.5x with EAGLE-3, 1.38x throughput at batch=64

### 2.4 Python mlx-lm

**Production-ready** `speculative_generate_step()` with external draft model. Also implementing MTP (PR #990) for Qwen3.5's native multi-token prediction head.

Performance data on Apple Silicon:

| Model | Quant | M4 Pro Speedup | Acceptance Rate |
|-------|-------|----------------|-----------------|
| Qwen3.5-27B dense | 4-bit | **1.52x** | 80.6% |
| Qwen3.5-27B dense | 8-bit | **1.32x** | ~32% |
| Qwen3.5-35B-A3B MoE | 8-bit | **1.11x** | ~11% |
| Qwen3.5-122B-A10B MoE | 5-bit | **1.09x** | ~9% |

**Critical finding**: mlx-lm's `speculative_generate_step` has ~50% overhead vs `generate_step` even with 0 draft tokens due to MLX JIT being unable to resolve dynamic input shapes (Issue #250). MTP is disabled for batch serving (`is_batchable=False`).

---

## 3. Model Family Analysis

### 3.1 Qwen3.5 (Primary Target)

**Architecture**: Hybrid GatedDeltaNet (linear attention, 3 of every 4 layers) + full gated attention. This is **not** a standard transformer — the linear attention layers maintain recurrent state (conv1d state + DeltaNet state matrix `[B, Hv, Dv, Dk]`), not KV pairs.

**MTP availability**: `mtp_num_hidden_layers: 1` present in Qwen3.5-2B dense config. The MTP weights exist in published model files. However, **AFM currently strips them**:
```swift
// Scripts/patches/Qwen3_5MoE.swift:928
sanitizedWeights = sanitizedWeights.filter { !$0.key.contains("mtp.") }
```

**Draft model candidates**:
- Qwen3.5-0.8B, 2B, 4B (dense) — same vocabulary (248,320 tokens), same architecture family
- MoE acceptance rates are very low (~9-11%) vs dense models (~80%)

**Speculation complication**: Rollback for GatedDeltaNet requires snapshotting the recurrent state (`ArraysCache` with conv1d state + hidden state matrix). Unlike KV cache trim (just move an offset pointer), recurrent state rollback requires a full array copy. Python mlx-lm handles this with explicit state snapshots before speculation.

### 3.2 Nemotron

NemotronH is a hybrid architecture with SSM (Mamba) layers — similar rollback complexity as Qwen3.5. No dedicated small Nemotron draft models exist publicly. Nemotron-3-Nano-4B could draft for Nemotron-Terminal-8B but the size ratio is poor (50%).

### 3.3 Mistral

**Best draft model situation**: Mistral publishes purpose-built draft models:
- `Mistral-Small-3.1-DRAFT-0.5B` (0.6B)
- `Mistral-Large-Instruct-DRAFT` (~0.5B)

Available in MLX format. Vocabulary matches verified. Pure transformer architecture — standard KV cache trim on rollback. **Lowest-friction target for external draft model speculation.**

### 3.4 GLM / LFM

GLM4MoeLite and GLM5MoeDsa are implemented. No dedicated draft models. GLM5 uses DSA (Dynamic Sparse Attention) which adds complexity.

LFM (Liquid Foundation Models) use the `lfm2` tool call format. No public draft models with matching vocabulary.

---

## 4. Limitations Due to AFM's General-Purpose Nature

This is the most important section. AFM is a **general-purpose OpenAI-compatible server**, not a specialized inference engine for a specific model. This fundamentally constrains speculative decoding.

### 4.1 No Guaranteed Draft Model Pairing

The user runs `afm mlx -m <any-model>`. AFM cannot assume a draft model exists or is cached. Options:
- **User-specified**: `--draft-model <model-id>` (cleanest, mirrors mlx-lm)
- **Auto-discovery**: Attempt to find a small model with matching vocabulary — fragile, slow, storage-intensive
- **N-gram only**: No model required, but limited applicability

### 4.2 Sampling Parameter Interaction

AFM supports the full OpenAI sampling parameter set: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `seed`. The verification algorithm's mathematical guarantee (output distribution identical to non-speculative) requires running **the same processor chain** on both draft and target logits:

- `repetition_penalty` and `presence_penalty` operate on a **window of prior tokens** — the draft model needs access to the same token history
- `top_k` and `min_p` modify the distribution shape — rejection sampling must account for these
- `seed`-based deterministic generation becomes harder: speculation introduces variable-length accept/reject cycles that must produce identical output regardless of draft accuracy
- **Greedy decoding (temperature=0)** is the simplest case: accept iff `argmax(target) == draft_token`

### 4.3 Feature Interaction Complexity

Every AFM feature intersects with speculative decoding:

| Feature | Interaction | Complexity |
|---------|-------------|------------|
| **Tool call detection** | Token-level `<tool_call>` tag matching happens in streaming loop. Speculative tokens must not be yielded until verified, but tag detection buffers state across tokens. | Medium |
| **Think extraction** | `<think>` tag buffering (7-8 chars) must operate on verified tokens only. Draft tokens that cross tag boundaries and get rejected would corrupt state. | Medium |
| **Stop sequences** | Buffer-based approach spans chunk boundaries. Verified tokens may include stop sequences that weren't visible in individual draft tokens. | Medium |
| **Logprobs** | Must return logprobs from the **target** model's distribution, not the draft's. Requires storing target logits for all K candidate positions even if draft was used for sampling. | High |
| **Prompt caching** | RadixTree prefix cache stores verified KV state. Speculative tokens must not pollute the cache. | Low |
| **Batch scheduling** | Speculation is fundamentally single-request. With `maxConcurrent >= 2`, speculation per-request competes for GPU time with other requests. At high concurrency, overhead exceeds benefit (confirmed by vLLM). | High |
| **Streaming SSE** | Verified tokens can stream immediately. But streaming a token then rejecting it is catastrophic — once sent via SSE, it cannot be retracted. | Critical |

### 4.4 Hybrid Architecture State Rollback

For pure-transformer models (Mistral, Llama, older Qwen), speculation rollback is cheap: truncate KV cache offset. For Qwen3.5 and NemotronH (the primary target families), rollback requires:

1. **KV cache truncation** for full-attention layers (cheap, pointer move)
2. **Recurrent state restoration** for GatedDeltaNet/Mamba layers (expensive, full array copy)
3. **Conv1d state restoration** (small but adds to the copy cost)

Python mlx-lm handles this by snapshotting `(conv_state, recurrent_state)` before each speculation round and restoring on rejection. The snapshot cost is proportional to `num_linear_layers * state_size`.

For Qwen3.5-35B-A3B: 30 linear attention layers x state arrays = significant snapshot/restore overhead per speculation round.

### 4.5 Memory Bandwidth Saturation

Apple Silicon is memory-bandwidth-bound for LLM inference. Speculative decoding's value proposition is reducing the number of sequential memory reads of the target model's weights. But:

- **Draft model adds bandwidth pressure**: Both models' weights must be read from unified memory
- **Net benefit formula**: `speedup = (accepted_tokens) / (1 + draft_cost/target_cost)`
- For a 0.5B draft + 35B target: draft cost is ~1.4% of target cost per token, so the overhead is minimal IF acceptance rate is high
- For 4-bit quantized models: bandwidth savings from fewer target forward passes are smaller in absolute terms (weights are already compressed), reducing the ceiling
- **fp16 MTP is slower than baseline** on Apple Silicon due to bandwidth saturation (mlx-lm finding)

### 4.6 Single-Sequence vs Multi-Tenant

AFM serves multiple concurrent requests. Speculation is a single-sequence optimization:

- At **batch=1** (single user): speculation is maximally beneficial
- At **batch=2-4**: each speculative round for request A delays request B's token. Net throughput may decrease
- At **batch=8**: speculation should be disabled — the BatchScheduler's pipelined decode is more efficient

This means speculation should be **dynamically enabled/disabled based on current load**, adding complexity.

---

## 5. The Verification Algorithm

The core rejection sampling algorithm used across all frameworks:

```
For each draft token position i with draft probability p[i] and target probability q[i]:
    r = random()
    if r < min(1, q[i][token] / p[i][token]):
        ACCEPT token
    else:
        REJECT: sample from normalized max(0, q[i] - p[i])
        Stop processing remaining draft tokens
```

**Mathematical guarantee:** This procedure produces output distributions identical to sampling from the target model alone (proven in Leviathan et al., 2023 and Chen et al., 2023).

**Greedy simplification (used in llama.cpp simple mode):** When both models use argmax sampling (temperature=0), a draft token is accepted if and only if `argmax(q[i]) == draft_token[i]`. No probability ratio computation needed.

**KV cache on rejection:** All frameworks trim/rollback the KV cache to the last accepted position. The target model's KV cache for positions up to the last accepted token is preserved; everything after is discarded.

---

## 6. Viable Implementation Approaches (Ranked)

### Tier 1: Practical and High-Value

#### A. N-gram / Prompt Lookup Speculation
**Design**: No draft model. Speculate from the prompt itself.

Match the last N tokens against earlier occurrences in the prompt/completion and predict the continuation. Extremely effective for code editing (repetitive patterns), summarization, and translation.

**Implementation sketch**:
1. Maintain a sliding window of generated + prompt tokens
2. For each decode step, search for the longest n-gram match in the window
3. If found, propose the continuation as draft tokens
4. Verify with target model

##### Real-World N-gram Speculation Examples

**Example 1: Code Editing via OpenCode — Renaming a variable**

The user asks: *"Rename `fetchUserData` to `loadUserProfile` throughout this file."*

The prompt contains the original source code. The model's output will repeat large portions of it with targeted substitutions. When the model has generated up to the cursor `|`:

```
Prompt (original code):
  func fetchUserData(id: Int) -> User {
      let result = database.fetchUserData(id: id)
      return result.map { User(from: $0) }
  }

  func refreshCache() {
      let data = fetchUserData(id: currentUser.id)
      cache.store(data)
  }

Generated so far:
  func loadUserProfile(id: Int) -> User {
      let result = database.loadUserProfile(id: id)
      return result.map { User(from: $0) }|
```

The n-gram matcher sees the last 4 tokens: `User`, `(`, `from`, `:`. It scans the prompt and finds an exact match at the same position in the original code. The continuation in the prompt is ` $0) }\n  }\n\n  func refreshCache`. It proposes those tokens as draft candidates (up to K=8).

The target model verifies: tokens ` $0`, `)`, `}`, `\n`, `}`, `\n\n`, `func`, `refreshCache` are all accepted in a single forward pass. **8 tokens verified at once instead of 8 sequential decode steps.** This pattern repeats for every unchanged section of the file.

**Acceptance rate for code editing**: ~70-90% of tokens are unchanged context that n-gram matching predicts perfectly.

**Example 2: OpenClaw Chat — Repeating structured output**

The user asks: *"List all HTTP status codes in the 4xx range with descriptions."*

```
Generated so far:
  - 400 Bad Request: The server cannot process the request due to client error
  - 401 Unauthorized: Authentication is required and has failed
  - 402 Payment Required: Reserved for future use
  - 403 Forbidden: The server understood the request but refuses to authorize it
  - 404 Not Found:|
```

The n-gram matcher sees the last 3 tokens: `Not`, `Found`, `:`. It looks for this pattern in prior output and finds that every previous line followed the pattern `<code> <name>: The server` or `<code> <name>: <description>`. It can't match the specific description (it's unique), but the structural tokens (`: The`, `server`) appear frequently. With a 2-gram match on `: The`, it proposes `The` + `server` as drafts.

Result: the model accepts `The` (correct start) but rejects `server` (the actual continuation is `The requested resource`). **1 token saved** — modest but free.

**Example 3: Summarization — High prompt-output overlap**

User sends a long article and asks: *"Summarize this article."*

```
Prompt excerpt:
  ...The European Central Bank announced on Thursday that it would
  raise interest rates by 25 basis points, bringing the benchmark
  rate to 4.50%. ECB President Christine Lagarde said the decision
  was driven by persistent inflationary pressures...

Generated so far:
  The European Central Bank raised interest rates by 25 basis points
  to 4.50%. ECB President Christine Lagarde cited persistent|
```

The 3-gram `ECB President Christine` matches the prompt. The continuation in the prompt is `Lagarde said the decision was driven by persistent inflationary pressures`. The model is generating a summary, so it may compress — but the next few tokens (`inflationary pressures`) are highly likely.

Draft: `inflationary`, `pressures`. Both verified and accepted. **2 tokens in 1 step.**

vLLM measured **2.8x speedup** on CNN/DailyMail summarization using this exact approach.

**Example 4: Tool call responses — Repeated JSON structure**

AFM serves tool-calling requests where models emit structured XML/JSON. After the model has made one tool call:

```
Generated so far:
  <tool_call><function=read_file><parameter=path>/src/main.swift</parameter></function></tool_call>

Now generating second tool call:
  <tool_call><function=read_file><parameter=path>|
```

The 4-gram `<tool_call><function=read_file><parameter=path>` matches the previous tool call. The structural tokens `>`, `</parameter>`, `</function>`, `</tool_call>` are predictable — only the parameter value differs. The n-gram matcher proposes the closing structure after the value is generated.

**Example 5: Where n-gram fails — Creative/novel generation**

User asks: *"Write a poem about the ocean."*

```
Generated so far:
  Beneath the waves where shadows play,
  the coral reefs hold dreams at bay.|
```

No n-gram from the prompt or prior output matches `dreams at bay`. Every token is novel. The n-gram matcher finds nothing, proposes 0 draft tokens, and decode falls back to standard autoregressive — **zero overhead, zero benefit.**

##### N-gram Match Algorithm

```
function find_draft_tokens(context_tokens, n=4, max_draft=8):
    suffix = context_tokens[-n:]          # last n generated tokens
    candidates = []

    # scan prompt + all generated tokens for matching n-gram
    for i in range(len(context_tokens) - n):
        if context_tokens[i:i+n] == suffix:
            # found match — propose continuation as draft
            continuation = context_tokens[i+n : i+n+max_draft]
            if len(continuation) > len(candidates):
                candidates = continuation

    return candidates  # may be empty (no match found)
```

The search is O(context_length) per decode step but operates on integer token IDs (not strings), making it fast enough to be negligible compared to a model forward pass (~10-50ms on Apple Silicon).

**Pros**: Zero memory overhead, no draft model needed, works with any model, trivially compatible with batch scheduling.

**Cons**: Only helps when output has overlap with prompt (code completion, summarization). Useless for creative generation.

**Expected speedup**: 1.5-2.8x when applicable (vLLM saw 2.8x on CNN/DailyMail), 1.0x (no benefit) for creative tasks.

#### B. External Draft Model Speculation
**Design**: Mirror mlx-lm's `speculative_generate_step()` approach.

```
CLI: afm mlx -m <target> --draft-model <draft> --num-draft-tokens 3
```

**Implementation sketch**:
1. Load draft model as second `ModelContainer`
2. New `SpeculativeGenerator` that holds both containers
3. Loop: draft K tokens → target model verifies K+1 positions in one forward pass → accept/reject
4. KV cache management: separate caches, trim target cache on rejection
5. For hybrid architectures: snapshot recurrent state before draft round

**Pros**: Proven approach (llama.cpp, mlx-lm, vLLM all do this), model-agnostic, user controls draft choice.

**Cons**: 2x model load overhead, user must manage draft model availability, disabled when `maxConcurrent >= 2`.

**Expected speedup**: 1.3-1.5x on Apple Silicon (4-bit quantized), based on mlx-lm benchmarks. Higher for pure-transformer models (Mistral: ~1.5-2x), lower for hybrid models (Qwen3.5 MoE: ~1.1x).

**Best draft model pairings**:

| Target | Draft | Vocab Match | Architecture | Expected Acceptance |
|--------|-------|-------------|--------------|-------------------|
| Mistral-Large | Mistral-Large-DRAFT-0.5B | Yes (dedicated) | Transformer | High (~80%+) |
| Mistral-Small-3.1 | Mistral-Small-3.1-DRAFT-0.5B | Yes (dedicated) | Transformer | High (~80%+) |
| Qwen3.5-27B dense | Qwen3.5-0.8B | Yes (248K vocab) | Hybrid GDN | ~80% (dense) |
| Qwen3.5-35B-A3B MoE | Qwen3.5-0.8B | Yes (248K vocab) | Hybrid GDN | ~9-11% (MoE) |
| Qwen3-Coder-30B-A3B | Qwen3-0.6B | Likely | Transformer | Medium |

### Tier 2: High Value but Requires Significant Work

#### C. MTP (Multi-Token Prediction) for Qwen3.5
**Design**: Use the model's built-in MTP head instead of a separate draft model.

Qwen3.5 dense models ship with `mtp_num_hidden_layers: 1`. The MTP head fuses the pre-norm hidden state with the embedding of the predicted next token through a single transformer layer to predict token t+2.

**Implementation requirements**:
1. **Stop stripping MTP weights** — currently filtered in `sanitizedWeights.filter { !$0.key.contains("mtp.") }`
2. **Implement MTP head in Swift** — single transformer layer that takes `(hidden_state, embed(predicted_token))` and produces logits
3. **State snapshot/rollback** for GatedDeltaNet layers
4. **Generation loop modification** — after each target forward pass, run MTP head to get one draft token, then verify both in next forward pass

**Pros**: No additional model memory (just one extra transformer layer), highest acceptance rate for dense Qwen3.5 models (~80%), native to the model architecture.

**Cons**: Only works for Qwen3.5 (and potentially DeepSeek V3, MiniMax M2 which also have MTP). Requires architecture-specific code. MoE variants see minimal benefit (~9-11%). GatedDeltaNet rollback adds per-step overhead. mlx-lm's JIT overhead issue (~50%) may apply to MLX Swift as well.

**Expected speedup**: ~1.5x for 4-bit dense Qwen3.5, ~1.1x for MoE variants.

#### D. EAGLE-Style Draft Head
**Design**: Train a lightweight draft head (single transformer layer) on the target model's second-to-last hidden state.

**Implementation requirements**:
1. Extract hidden states during generation (already accessible in MLX Swift's forward pass)
2. Implement EAGLE head: `draft_logits = lm_head(eagle_layer(hidden[-2] + embed(current_token)))`
3. Tree-based verification with custom attention masks
4. Separate KV cache for the EAGLE head

**Pros**: 2.7-3.5x speedup potential (EAGLE-1), 3-4.3x (EAGLE-2). Much higher than external draft models. Single model memory footprint.

**Cons**: Requires training EAGLE heads per model (1-2 days on 8x GPU). No pre-trained EAGLE heads for Qwen3.5 or Nemotron yet. Tree attention masks need custom implementation. Highest engineering complexity.

### Tier 3: Research / Long-Term

#### E. Layer-Skipping Self-Speculation
Use the first N layers of the target model as a draft, then verify with all layers. No additional memory.

**Blocker**: Requires either (a) models trained with the LayerSkip recipe (only exists for Llama), or (b) accepting degraded acceptance rates. The `LanguageModel` protocol doesn't support early exit.

#### F. Medusa Heads
Multiple prediction heads attached to the base model.

**Blocker**: Requires training Medusa heads per model. No pre-trained heads for target families. MLX Swift doesn't support tree attention masks natively.

---

## 7. Summary of Speedup Ranges Across Methods

| Method | Typical Speedup | Memory Overhead | Training Required | AFM Feasibility |
|--------|----------------|-----------------|-------------------|-----------------|
| N-gram / prompt lookup | 1.5-2.8x (when applicable) | Negligible | No | **High** |
| External draft model | 1.3-2.0x | +5-15% | No | **High** |
| MTP (native, Qwen3.5) | 1.1-1.5x | +1 layer | No (built-in) | **Medium** |
| EAGLE-1/2/3 | 2.7-6.5x | +2-5% | Yes | Low |
| Medusa | 2.2-3.6x | +1-3% | Yes | Low |
| LayerSkip | 1.5-2.1x | None | Yes (special) | Very Low |
| Layer skip (no retrain) | 1.3-2.0x | None | No | Low |
| Lookahead decoding | 2.7-6.3x | Negligible | No | Low |

---

## 8. Recommended Implementation Path

### Phase 1: N-gram Prompt Lookup (Low effort, immediate value)
- Zero memory overhead, model-agnostic
- Particularly valuable for OpenCode/OpenClaw integrations (code editing has high prompt-output overlap)
- Disable when `maxConcurrent >= 2`
- Add `--speculation ngram` CLI flag, disable by default

### Phase 2: External Draft Model (Medium effort, broad value)
- `--draft-model <id> --num-draft-tokens 3`
- Vocabulary matching validation at load time
- Separate `ModelContainer` for draft
- Auto-disable at concurrency > 1
- Works with all pure-transformer models (Mistral has dedicated draft models ready)
- For hybrid models: implement state snapshot/rollback for `ArraysCache`

### Phase 3: MTP for Qwen3.5 (High effort, targeted value)
- Stop filtering `mtp.*` weights
- Implement MTP head in Swift
- GatedDeltaNet state snapshot/rollback
- Dense-only (MoE acceptance too low to justify)

### Not Recommended for AFM
- EAGLE/Medusa: No pre-trained heads for target families, training infrastructure out of scope
- Layer-skipping: No LayerSkip-trained models for target families
- Always-on speculation: Must be opt-in due to overhead at high concurrency

---

## 9. Key Takeaway

The fundamental tension is that AFM is a **multi-model, multi-tenant server** while speculative decoding is a **single-model, single-request optimization**. Every reference implementation (llama.cpp, vLLM, SGLang, mlx-lm) either operates in single-request mode or explicitly disables speculation under load. AFM should follow the same pattern: speculation as an opt-in feature that auto-disables when the BatchScheduler has multiple active slots.

The most pragmatic near-term win is **n-gram prompt lookup** — zero cost when it doesn't apply, meaningful speedup for the code-assistant use case (OpenCode integration), and no model management complexity. External draft models are the next step, with Mistral's dedicated 0.5B draft models being the lowest-friction target.
