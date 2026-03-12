# Stop Sequence Test Report — Qwen3.5-35B-A3B-4bit

**AFM Version:** v0.9.5-0cfba17 | **Date:** 2026-02-25 | **Model:** mlx-community/Qwen3.5-35B-A3B-4bit

## 1. Broken Models

### CLI `--stop` flag not implemented (4 failures)

| Variant | Error |
|---------|-------|
| stop-cli-only | `Unknown option '--stop'. Did you mean '--top-p'?` |
| stop-cli-multi | Same |
| stop-cli-api-merge | Same |
| stop-cli-api-dedup | Same |

**Verdict: AFM bug.** The `--stop` CLI flag does not exist. Stop sequences only work via the API `stop` field. This is a missing feature, not a model issue.

### `response_format: json_object` + stop sequences (2 failures)

| Variant | Error |
|---------|-------|
| stop-json-object-key | `Jinja.TemplateException error 1` |
| stop-json-object-no-match | Same |

**Verdict: AFM bug.** The Jinja template rendering crashes when `response_format: json_object` is used. This is independent of stop sequences — the stop field just happens to be present. The template error likely occurs because the `json_object` response format injects a system prompt that the Qwen3.5 chat template can't handle.

## 2. Anomalies & Red Flags

### Critical: Stop sequences match inside `<think>` reasoning, not just visible content

Nearly every API stop-sequence test stopped generation inside the model's `<think>` block, producing **empty `content`** with only `reasoning_content`. The stop sequences are being applied to the raw token stream (including think tags), not just the user-visible output.

Examples:
- **stop-api-newline** (`stop: ["\n"]`): Stopped after just "Thinking Process:" — 6 tokens, empty content
- **stop-api-double-newline** (`stop: ["\n\n"]`): Same — 6 tokens, empty content
- **stop-api-period** (`stop: ["."]`): Stopped at "Thinking Process:\n\n1" — 6 tokens
- **stop-immediate** (`stop: ["The","I","A"]`): Stopped at "Thinking Process:\n\n1.  **" — 8 tokens
- **stop-special-chars** (`stop: ["**"]`): Stopped at "Thinking Process:\n\n1." — 7 tokens
- **stop-unicode** (`stop: ["•"]`): 38 tokens, still only reasoning

**This is likely an AFM bug.** Stop sequences should arguably only apply to the visible `content` stream, not to reasoning tokens inside `<think>` tags. For thinking models, the stop sequence implementation needs to be aware of the think/content boundary.

### Thinking-Budget Exhaustion Pattern

Several tests show the model spending its entire (effective) budget on reasoning with no visible content. This is **not** budget exhaustion in the max_tokens sense — it's stop sequences cutting off generation while still in the reasoning phase.

| Variant | Tokens | Content | Stopped By |
|---------|--------|---------|------------|
| stop-api-single | 90 | empty | "3." in reasoning |
| stop-api-multi | 98 | empty | "4." in reasoning |
| stop-streaming | 161 | empty | "3." in reasoning |
| stop-non-streaming | 161 | empty | "3." in reasoning |
| stop-long-phrase | 44 | empty | "In conclusion" in reasoning |
| stop-multi-word | 111 | empty | "Step 3" in reasoning |
| stop-code-fence | 122 | empty | "```" in reasoning |
| stop-html-tag | 107 | empty | "</li>" in reasoning |

### Tests that worked correctly

| Variant | Tokens | Content | Notes |
|---------|--------|---------|-------|
| stop-no-match (`XYZZY`) | 134 | "4" | Correct — stop string never matched, model completed |
| stop-guided-json-no-match (`XYZZY`) | 1150 | Full response | Correct — guided JSON + non-matching stop |
| stop-guided-json-brace (`}`) | 819 | Full response | Correct — stop at `}` but model finished before hitting it in content |

## 3. Variant Comparison

### Streaming vs Non-Streaming (stop-streaming vs stop-non-streaming)
Same prompt, same stop `["3."]` — **identical behavior**: both produced 161 tokens, empty content, reasoning stopped at same point. Stop sequence handling is consistent across modes.

### Seed Reproducibility (stop-seed-run1 vs stop-seed-run2)
Same prompt, same seed 42, same stop `["3."]` — **identical**: 80 tokens each, same reasoning content. Deterministic generation confirmed.

### Guided JSON variants
- **stop-guided-json-value** (`stop: ["Tokyo"]`): 38 tokens, empty content — stopped in reasoning
- **stop-guided-json-comma** (`stop: [","]`): 58 tokens, empty content — stopped in reasoning
- **stop-guided-json-no-match** (`stop: ["XYZZY"]`): 1150 tokens, **full response with content** — correctly unaffected
- **stop-guided-json-brace** (`stop: ["}"]`): 819 tokens, **full response with content** — interestingly worked

The `}` and `XYZZY` cases produced full content because the model's reasoning didn't contain those strings, allowing it to exit `<think>` and produce visible output.

### Temperature (stop-high-temp, temp=1.0)
144 tokens, empty content, stopped in reasoning. Higher temperature didn't change the fundamental stop-in-think issue.

## 4. Quality Assessment

| Variant | Coherence | Relevance | Notes |
|---------|-----------|-----------|-------|
| stop-no-match | 5 | 5 | Clean, correct answer "4" |
| stop-guided-json-no-match | 4 | 4 | Good response, extra plain-text version unnecessary |
| stop-guided-json-brace | 5 | 5 | Clean description with hex code |
| All others with content=empty | N/A | N/A | Cannot rate — stopped in reasoning |

Models scoring below 3: None had bad quality *when they produced output*. The issue is stop sequences preventing output entirely.

## 5. Performance Summary

| Variant | tok/s | Tokens | Notes |
|---------|-------|--------|-------|
| stop-code-fence | 44.15 | 122 | |
| stop-guided-json-no-match | 41.29 | 1150 | Best sustained throughput |
| stop-guided-json-brace | 41.28 | 819 | |
| stop-no-match | 39.80 | 134 | |
| stop-api-word | 38.38 | 90 | |
| stop-system-numbered | 35.37 | 138 | |
| stop-streaming | 33.59 | 161 | |
| stop-non-streaming | 32.20 | 161 | |
| stop-multi-word | 31.92 | 111 | |
| stop-top-p | 31.40 | 111 | |
| stop-long-phrase | 31.28 | 44 | |
| stop-guided-json-comma | 30.67 | 58 | |
| stop-html-tag | 30.22 | 107 | |
| stop-unicode | 30.02 | 38 | |
| stop-api-multi | 28.43 | 98 | |
| stop-high-temp | 28.32 | 144 | |
| stop-api-single | 28.26 | 90 | |
| stop-seed-run2 | 28.31 | 80 | |
| stop-low-max-tokens | 26.61 | 40 | |
| stop-system-pirate | 24.69 | 374 | |
| stop-seed-run1 | 24.13 | 80 | |
| stop-four-max | 22.39 | 102 | |
| stop-immediate | 18.70 | 8 | Low token count = startup overhead |
| stop-api-double-newline | 16.95 | 6 | |
| stop-api-period | 15.95 | 6 | |
| stop-special-chars | 15.74 | 7 | |
| stop-api-newline | 12.93 | 6 | |

Performance is consistent at ~28-44 tok/s for sustained generation. Low tok/s values (12-19) correlate with very short generations (<10 tokens) where startup overhead dominates. No suspicious outliers.

## 6. Recommendations

### Likely AFM Bugs (Priority Order)

1. **Stop sequences match inside `<think>` tags** — This is the highest-priority issue. For thinking models, stop sequences should only apply to content *after* the closing `</think>` tag. This causes 80%+ of tests to produce empty responses. The fix should be in the streaming loop where stop sequence matching occurs.

2. **`--stop` CLI flag missing** — 4 tests fail because the CLI doesn't accept `--stop`. Either add it as a CLI option or document that stop sequences are API-only.

3. **`response_format: json_object` crashes with Jinja error** — Template rendering fails for this model when json_object format is requested. Likely needs a model-specific template fix or fallback.

### Model Quality Issues
- None. When the model produces output (stop-no-match, guided-json-no-match, guided-json-brace), quality is good.

### Working Well
- API stop sequences correctly stop generation when the stop string appears (the matching itself works)
- Streaming and non-streaming produce identical results
- Seed-based reproducibility works
- Guided JSON + stop sequences coexist (when no template error)
- Non-matching stop sequences correctly allow full generation

<!-- AI_SCORES [{"i":0,"s":2},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":1},{"i":7,"s":1},{"i":8,"s":1},{"i":9,"s":1},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":5},{"i":15,"s":4},{"i":16,"s":1},{"i":17,"s":1},{"i":18,"s":2},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":5},{"i":22,"s":2},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2}] -->
