# AFM QA Report: Qwen3.5-35B-A3B-4bit Test Suite

**AFM Version:** v0.9.5-0cfba17 | **Model:** mlx-community/Qwen3.5-35B-A3B-4bit | **Date:** 2026-02-25

---

## 1. Broken Models (Failures by Error Type)

### Connection Errors with `--enable-prefix-caching` (3 FAILs)
All 3 failures occur on `agent-cached-turn*` variants, exclusively on the **second prompt** per turn pair. The first prompt (simple factual Q) always succeeds.

| Index | Label | Prompt | Error |
|-------|-------|--------|-------|
| 46 | agent-cached-turn1 | "Read the file Sources/MacLocalAPI/main.swift..." | Connection error |
| 48 | agent-cached-turn2 | "Now add a --timeout flag..." | Connection error |
| 50 | agent-cached-turn3 | "Write a unit test..." | Connection error |

**Pattern:** Prefix caching works for the initial request but crashes/disconnects on the second, different prompt sharing the same system prompt prefix. The non-cached equivalents (agent-no-cache-turn1/2/3) succeed on identical prompts. **This is likely an AFM bug in `PromptCacheBox` — possibly a race condition or cache invalidation failure when the suffix changes.**

---

## 2. Anomalies & Red Flags

### A. Thinking-Budget Exhaustion (Critical — 19 instances)

The model's thinking (reasoning) tokens count against `max_tokens`, causing **empty `content`** output when reasoning is verbose. This is the single biggest quality issue.

**Simple questions that exhaust 4096 tokens on thinking alone:**

| Index | Label | Prompt | Reasoning Tokens | Content |
|-------|-------|--------|-----------------|---------|
| 27 | scientist | "What is the capital of Japan?" | 4096 | `""` |
| 29 | eli5 | "What is the capital of Japan?" | 4096 | `""` |
| 123 | code-python | "What is the capital of Japan?" | 4096 | `""` |
| 125 | code-swift | "What is the capital of Japan?" | 4096 | `""` |
| 95 | developer-role | "developer: You are a helpful coding assistant..." | 4096 | `""` |

These are **trivial factual questions** where the model overthinks due to persona/prompt conflict. The reasoning shows: `"Thinking Process: 1. **Analyze the Request:** ... Role: Physics Professor ... The question asks for a geographical/political fact (c..."` — the model gets stuck deliberating how to reconcile the system persona with a non-matching question.

**Creative tasks that exhaust budget:**

| Index | Label | Prompt | Tokens | Content |
|-------|-------|--------|--------|---------|
| 16 | seed-42-run1 | "Write a limerick about a cat." | 4096 | `""` |
| 18 | seed-42-run2 | "Write a limerick about a cat." | 4096 | `""` |
| 88 | streaming-seeded | "Write a 4-line poem about the ocean." | 4096 | `""` |
| 90 | non-streaming-seeded | "Write a 4-line poem about the ocean." | 4096 | `""` |
| 120 | special-chars | "Repeat these characters exactly..." | 4028 | `""` |

For the limerick, reasoning shows: `"Drafting - Attempt 1: There was a young cat named Tim, (A)..."` — the model drafts multiple attempts within thinking, never finishing.

**Context-heavy prompts exhausting budget:**

| Index | Label | Prompt | Content |
|-------|-------|--------|---------|
| 62 | prefill-default | "top 3 performance bottlenecks" | `""` |
| 64 | prefill-large-4096 | same | `""` |
| 66 | prefill-small-256 | same | `""` |
| 92 | max-completion-tokens | "max_completion_tokens: 100" | `""` |
| 109 | tool-call-multi | "weather in London and time in Tokyo?" | `""` |

### B. Short max_tokens Completely Broken for Thinking Models

| Index | Label | max_tokens | Reasoning Tokens | Content |
|-------|-------|-----------|-----------------|---------|
| 51 | short-output | 50 | 50 | `""` |
| 52 | short-output | 50 | 50 | `""` |

With `max_tokens=50`, **all tokens go to thinking**, producing zero user-visible content. The `content_preview` field shows: `"Thinking Process: 1. **Analyze the Request:**..."` leaking into content_preview, suggesting the test harness may not be correctly separating think/content in this edge case.

### C. Logprobs Not Returned

Lines 55-56 use `--max-logprobs 5` but `logprobs_count: 0` for both. Either logprobs are silently dropped or the test harness isn't capturing them.

### D. Guided JSON / Response Format Not Enforced

Per CLAUDE.md, response_format uses "prompt injection (not guaranteed valid JSON)." Confirmed failures:

- **Line 36** (guided-json-simple, "Generate a person record"): Returns verbose markdown with `record_id`, `personal_info`, `contact_info`... instead of `{"name": ..., "age": ...}` matching the schema
- **Line 38** (guided-json-nested, "Describe Tokyo"): Returns markdown table, not JSON with `city`/`population`/`landmarks`
- **Line 77** (response-format-json, capital question): Returns `"The capital of Japan is Tokyo."` (plain text, not JSON)
- **Line 80** (response-format-schema, person profile): Returns full markdown character profile, not `{"name":..., "age":..., "hobbies":[...]}` matching schema

### E. Stop Sequences Appear Double-Encoded

Stop fields show double-JSON-encoding: `"stop": ["[\"3.\"]"]` instead of `"stop": ["3."]`. This is a **test harness bug**, not an AFM bug. All stop-sequence tests (71-76) produce complete output because the stop strings are never actually matched.

---

## 3. Variant Comparison (A/B Highlights)

### Seed Reproducibility: PASS
- `seed-42-run1` vs `seed-42-run2` for capital question: both produce identical 197-token outputs
- Both limerick attempts: identical 4096-token exhaustion

### Streaming vs Non-Streaming Seeded: PASS
- `streaming-seeded` vs `non-streaming-seeded` for capital: both 206 tokens, identical content
- Both poem attempts: identical exhaustion

### Prefix Caching vs No Caching:
- **No caching**: All 6 prompts succeed (lines 39-44)
- **With caching**: First prompt succeeds, second prompt FAILs with connection error (lines 45-50)

### Penalty Variants (bread essay):
- `no-penalty` (line 20): `"The Unbroken Chain"` — 41.84 tok/s
- `with-penalty` (line 22): `"The Staff of Life"` — 39.18 tok/s (6% slower)
- `repetition-penalty` (line 24): `"The Universal Leaven"` — 39.38 tok/s (6% slower)

Penalties reduce speed slightly. All three essays are high quality with distinct titles, confirming sampling variation works.

### Prefill Step Sizes:
All three (default, 4096, 256) produce identical outputs for the capital question (454 tokens each). The bottleneck question exhausts budget across all variants. Prefill step size has negligible impact on output.

---

## 4. Quality Assessment

All models scoring below 3 on coherence or relevance:

| Index | Label | Issue | Score |
|-------|-------|-------|-------|
| 126 | code-swift | Swift async function: 4096 tokens of reasoning, zero code output | 1 |
| 80 | response-format-schema | Person profile: ignores JSON schema constraint entirely | 2 |
| 16, 18 | seed-42-run1/2 | Limerick: spends entire budget drafting in thinking, zero output | 2 |
| 27, 29 | scientist, eli5 | Capital of Japan: trivial Q produces zero content | 2 |
| 51, 52 | short-output | Any prompt: max_tokens=50 incompatible with thinking model | 2 |
| 88, 90 | seeded poems | 4-line poem: budget exhausted on drafting iterations | 2 |

---

## 5. Performance Summary

| Label | Prompt | tok/s | Tokens | Flag |
|-------|--------|-------|--------|------|
| greedy | sky blue | 42.47 | 1650 | |
| combined-samplers | sky blue | 41.71 | 1689 | |
| tool-call-multi | weather+time | 42.37 | 4131 | |
| long-form | MoE blog | 41.96 | 4159 | |
| seed-42-run2 | limerick | 42.00 | 4126 | |
| long-prompt | repeat text | 42.39 | 2020 | |
| long-output | cookie recipe | 42.21 | 2038 | |
| multilingual | translations | 41.94 | 1097 | |
| no-streaming | moon poem | 42.31 | 1988 | |
| with-penalty | bread essay | 39.18 | 4130 | ~7% slower |
| repetition-penalty | bread essay | 39.38 | 4130 | ~6% slower |
| min-p | capital | 38.82 | 278 | slightly slow |
| short-output | capital | 35.44 | 83 | LOW |
| agent-no-cache-turn1 | capital | 25.19 | 284 | **OUTLIER** |
| agent-no-cache-turn2 | capital | 27.88 | 284 | **OUTLIER** |
| agent-no-cache-turn3 | capital | 27.96 | 284 | **OUTLIER** |

**Typical range:** 39-42 tok/s for standard prompts.

**Outliers:**
- Agent prompts (220+ prompt tokens with long system prompt): 25-37 tok/s — the large system prompt significantly impacts per-token speed, likely due to longer prefill time being amortized over fewer generation tokens
- `short-output` (max_tokens=50): 35-37 tok/s — small batch overhead
- Penalty variants: ~6-7% slower (39 vs 42 tok/s), expected due to extra logit processing

---

## 6. Recommendations (Prioritized)

### P0 — Likely AFM Bugs

1. **Prefix caching crash on prompt change** (lines 46, 48, 50): `--enable-prefix-caching` causes connection errors when the second request has a different prompt suffix. Investigate `PromptCacheBox` for cache invalidation/corruption when the suffix changes after a cache hit.

2. **Thinking tokens consume max_tokens budget**: 19 tests produce empty content because `reasoning_content` exhausts the full `max_tokens`. AFM should either:
   - Implement a `thinking_budget` / `max_reasoning_tokens` parameter separate from `max_tokens`
   - Reserve a minimum portion of `max_tokens` for content (e.g., at least 25%)
   - Document this limitation prominently

3. **Logprobs silently not returned** (lines 55-56): `--max-logprobs 5` produces `logprobs_count: 0`. Either the feature is broken or results aren't being serialized.

### P1 — AFM Feature Gaps

4. **Guided JSON not constraining output** (lines 36, 38): `--guided-json` has no effect on generation. If this is intended to use constrained decoding, it's not working. If it's prompt-injection only, consider documenting the limitation.

5. **Response format enforcement weak** (lines 77, 79, 80): `response_format: json_object` doesn't guarantee JSON output unless the prompt also asks for JSON. Consider adding stronger system prompt injection or structured output support.

6. **Stop sequences test encoding** (lines 71-76): The test harness double-encodes stop sequences as `["[\"3.\"]"]`. Fix the test script to pass raw arrays. Cannot assess stop sequence functionality until this is fixed.

### P2 — Model Quality Issues (Not AFM Bugs)

7. **Overthinking on persona-mismatched prompts**: Qwen3.5 spends 4096 tokens deliberating when a system persona (physicist, ELI5, Python expert) receives a simple factual question. This is a model behavior issue — consider adding a note that thinking models need higher `max_tokens` or a thinking budget.

8. **Creative writing budget exhaustion**: Simple creative tasks (limerick, 4-line poem) trigger exhaustive drafting in thinking. The model iterates through `"Attempt 1... Attempt 2..."` without converging. This is model behavior, not an AFM bug.

### Working Well

- All sampler variants (greedy, default, high-temp, top-p, top-k, min-p, combined) produce correct, high-quality output
- Seed reproducibility is exact across runs and streaming modes
- System prompts (pirate, scientist for on-topic questions) work well
- JSON output when prompted correctly (line 32, 78) is valid
- Penalty variants produce distinct, quality output with expected slight slowdown
- Non-streaming mode produces identical results to streaming
- Long-form content (essays, blog posts, recipes) is excellent quality
- Multilingual output is accurate
- Code generation (Python sieve, reverse string) is clean and correct
- Agent-mode responses are contextually appropriate
- KV quantization and small KV cache produce correct output with no quality degradation

<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":5},{"i":2,"s":5},{"i":3,"s":5},{"i":4,"s":5},{"i":5,"s":5},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":5},{"i":12,"s":5},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":2},{"i":17,"s":5},{"i":18,"s":2},{"i":19,"s":5},{"i":20,"s":4},{"i":21,"s":5},{"i":22,"s":4},{"i":23,"s":5},{"i":24,"s":4},{"i":25,"s":5},{"i":26,"s":5},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":2},{"i":30,"s":5},{"i":31,"s":5},{"i":32,"s":5},{"i":33,"s":5},{"i":34,"s":5},{"i":35,"s":5},{"i":36,"s":3},{"i":37,"s":5},{"i":38,"s":3},{"i":39,"s":5},{"i":40,"s":4},{"i":41,"s":5},{"i":42,"s":4},{"i":43,"s":5},{"i":44,"s":4},{"i":45,"s":5},{"i":46,"s":1},{"i":47,"s":5},{"i":48,"s":1},{"i":49,"s":5},{"i":50,"s":1},{"i":51,"s":2},{"i":52,"s":2},{"i":53,"s":5},{"i":54,"s":5},{"i":55,"s":4},{"i":56,"s":4},{"i":57,"s":5},{"i":58,"s":5},{"i":59,"s":5},{"i":60,"s":5},{"i":61,"s":5},{"i":62,"s":2},{"i":63,"s":5},{"i":64,"s":2},{"i":65,"s":5},{"i":66,"s":2},{"i":67,"s":5},{"i":68,"s":5},{"i":69,"s":4},{"i":70,"s":4},{"i":71,"s":5},{"i":72,"s":5},{"i":73,"s":5},{"i":74,"s":5},{"i":75,"s":5},{"i":76,"s":5},{"i":77,"s":3},{"i":78,"s":5},{"i":79,"s":3},{"i":80,"s":2},{"i":81,"s":5},{"i":82,"s":5},{"i":83,"s":5},{"i":84,"s":5},{"i":85,"s":4},{"i":86,"s":4},{"i":87,"s":5},{"i":88,"s":2},{"i":89,"s":5},{"i":90,"s":2},{"i":91,"s":5},{"i":92,"s":2},{"i":93,"s":5},{"i":94,"s":5},{"i":95,"s":2},{"i":96,"s":5},{"i":97,"s":5},{"i":98,"s":5},{"i":99,"s":5},{"i":100,"s":5},{"i":101,"s":5},{"i":102,"s":4},{"i":103,"s":4},{"i":104,"s":5},{"i":105,"s":4},{"i":106,"s":4},{"i":107,"s":5},{"i":108,"s":4},{"i":109,"s":2},{"i":110,"s":5},{"i":111,"s":4},{"i":112,"s":3},{"i":113,"s":5},{"i":114,"s":4},{"i":115,"s":5},{"i":116,"s":5},{"i":117,"s":5},{"i":118,"s":5},{"i":119,"s":5},{"i":120,"s":2},{"i":121,"s":5},{"i":122,"s":5},{"i":123,"s":2},{"i":124,"s":5},{"i":125,"s":2},{"i":126,"s":1},{"i":127,"s":5},{"i":128,"s":5},{"i":129,"s":5},{"i":130,"s":4},{"i":131,"s":5},{"i":132,"s":5}] -->
