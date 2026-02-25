# Stop Sequences Test Report — Qwen3.5-35B-A3B-4bit

**AFM v0.9.5 | 2026-02-25 | 30 test cases**

---

## 1. Broken Models

| # | Label | Error | Type |
|---|-------|-------|------|
| 16 | `stop-json-object-key` | `Jinja.TemplateException error 1.` | AFM bug |
| 17 | `stop-json-object-no-match` | `Jinja.TemplateException error 1.` | AFM bug |

**Root cause**: Both failures use `response_format: "json_object"`. The Jinja template error occurs when `response_format` is set — likely the chat template rendering path for JSON mode is broken for Qwen3.5. This is **not** a stop-sequence issue; it's a `response_format` bug. Other tests with `--guided-json` (lines 12-15) work fine because guided-json uses a different code path.

**Action**: Investigate `response_format: "json_object"` handling in `MLXChatCompletionsController.swift` or wherever the Jinja template is rendered with response format hints. The `--guided-json` flag works, so the issue is specific to the OpenAI-style `response_format` field.

---

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Critical Pattern)

The **dominant issue** in this test run: Qwen3.5-35B-A3B is a thinking model that wraps reasoning in `<think>` tags. In **22 of 28 successful tests**, the model spent its entire output on `reasoning_content` and produced **empty `content`**. The stop sequences are matching inside the thinking/reasoning text, terminating generation before the model ever emits its actual response.

**This is the core bug being tested**: Stop sequences are being applied to the `<think>...</think>` reasoning content, not just to the visible response content.

Examples of stop sequences matching inside reasoning:
- `stop: ["3."]` matches `"3.  **Generate Content:**"` in thinking (lines 0, 1, 8, 9, 10, 11, etc.)
- `stop: ["\n"]` matches first newline in reasoning (line 3) — only 6 tokens generated
- `stop: ["."]` matches first period in `"Thinking Process:\n\n1"` → only 6 tokens (line 6)
- `stop: ["**"]` matches markdown bold in reasoning (line 22) — only 7 tokens
- `stop: ["</li>"]` matches inside reasoning HTML discussion (line 23)
- `stop: ["•"]` matches bullet char in reasoning (line 24)
- `stop: ["The", "I", "A"]` matches common words in reasoning (line 21) — only 15 tokens

### Specific Anomalies

| Line | Label | Issue | Snippet |
|------|-------|-------|---------|
| 3 | `stop-api-newline` | Stop `\n` killed reasoning after 6 tokens | `"Thinking Process:"` |
| 4 | `stop-api-double-newline` | Stop `\n\n` killed reasoning after 6 tokens | `"Thinking Process:"` |
| 6 | `stop-api-period` | Stop `.` killed reasoning after 6 tokens, 46s gen time | `"Thinking Process:\n\n1"` |
| 21 | `stop-immediate` | Stop `["The","I","A"]` killed after 15 tokens | `"Thinking Process:\n\n1.  **Identify the core question:**"` |
| 22 | `stop-special-chars` | Stop `**` killed after 7 tokens | `"Thinking Process:\n\n1."` |
| 20 | `stop-code-fence` | Stop `` ``` `` killed reasoning, no code emitted | Reasoning about factorial but never wrote code |

### Suspicious Timing

| Line | Label | Tokens | Time (s) | tok/s | Note |
|------|-------|--------|----------|-------|------|
| 6 | `stop-api-period` | 6 | 46.22 | 0.13 | Extremely slow — 46s for 6 tokens |
| 4 | `stop-api-double-newline` | 6 | 16.01 | 0.37 | 16s for 6 tokens |
| 3 | `stop-api-newline` | 6 | 5.93 | 1.01 | Slow for 6 tokens |

The period-stop case (line 6) is especially suspicious — 46 seconds to generate 6 tokens. This suggests the stop sequence matching or buffer flushing may be blocking/spinning before finally triggering.

---

## 3. Variant Comparison

### CLI `--stop` vs API `stop` field

| Mechanism | Label | Stop | Stopped in reasoning? | Actual content? |
|-----------|-------|------|-----------------------|-----------------|
| API `stop` | `stop-api-single` | `["3."]` | Yes | Empty |
| CLI `--stop` | `stop-cli-only` | `"3."` | **No** | Full 10-item list |
| Both merged | `stop-cli-api-merge` | CLI `"5."` + API `["3."]` | Yes | Empty |

**Key finding**: `stop-cli-only` (line 7) is the **only test with a numbered-list stop sequence that produced actual content**. It generated the full 10-fruit list at 40.98 tok/s. This test used `--stop "3."` via CLI args and the model did NOT have `thinking/no_think` template applied (note: `prompt_tokens: 39` vs `14` for API-stop variant — the CLI variant had a different system prompt injected that disabled thinking).

The `stop-cli-multi` test (line 8) also produced full content with CLI `--stop`. Both CLI-stop tests show `prompt_tokens` of 35-39 (vs 12-25 for API tests), suggesting the CLI path may be injecting a system prompt that disables thinking mode.

### Streaming vs Non-Streaming

| Label | Mode | Tokens | tok/s | Output |
|-------|------|--------|-------|--------|
| `stop-streaming` | streaming | 161 | 8.13 | Reasoning only, empty content |
| `stop-non-streaming` | non-streaming | 161 | 8.21 | Reasoning only, empty content |

Identical behavior — the bug affects both paths equally.

### Seed Reproducibility

| Label | Seed | Tokens | tok/s | Output |
|-------|------|--------|-------|--------|
| `stop-seed-run1` | 42 | 80 | 9.19 | Identical reasoning |
| `stop-seed-run2` | 42 | 80 | 9.19 | Identical reasoning |

Deterministic — seed works correctly.

---

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 7 | `stop-cli-only` | 5 | 5 | Perfect 10-item list |
| 8 | `stop-cli-multi` | 5 | 5 | Correct code block + END |
| 14 | `stop-guided-json-no-match` | 5 | 5 | Full JSON record with explanation |
| 15 | `stop-guided-json-brace` | 5 | 5 | Good description + hex code |
| 21 | `stop-no-match` | 5 | 5 | Correct answer "4" |
| 0-6, 9-13, 18-20, 22-29 | (most others) | 3 | 1 | Reasoning is coherent but no actual response |

Models scoring below 3: All tests where stop sequences match inside reasoning produce **no visible response** — relevance is 1 since the user gets nothing useful.

---

## 5. Performance Summary

| Line | Label | Tokens | tok/s | Note |
|------|-------|--------|-------|------|
| 20 | `stop-code-fence` | 122 | 46.72 | Normal (reasoning) |
| 15 | `stop-guided-json-brace` | 819 | 42.61 | Normal |
| 14 | `stop-guided-json-no-match` | 1150 | 42.58 | Normal |
| 8 | `stop-cli-multi` | 326 | 41.53 | Normal |
| 7 | `stop-cli-only` | 309 | 40.98 | Normal |
| 21 | `stop-no-match` | 134 | 39.96 | Normal |
| 29 | `stop-low-max-tokens` | 100 | 38.58 | Normal |
| 25 | `stop-system-pirate` | 374 | 19.57 | Moderate (thinking heavy) |
| 27 | `stop-high-temp` | 121 | 14.48 | Moderate |
| 23 | `stop-html-tag` | 107 | 13.63 | Moderate |
| 9 | `stop-cli-api-merge` | 82 | 10.46 | Low — stopped in reasoning |
| 28 | `stop-seed-run1/2` | 80 | 9.19 | Low — stopped in reasoning |
| 30 | `stop-top-p` | 111 | 8.83 | Low — stopped in reasoning |
| 29 | `stop-four-max` | 102 | 8.89 | Low — stopped in reasoning |
| 11 | `stop-streaming` | 161 | 8.13 | Low |
| 12 | `stop-non-streaming` | 161 | 8.21 | Low |
| 10 | `stop-cli-api-dedup` | 87 | 8.05 | Low |
| 19 | `stop-multi-word` | 111 | 6.87 | Low |
| 1 | `stop-api-multi` | 98 | 6.40 | Low |
| 0 | `stop-api-single` | 90 | 6.22 | Low |
| 26 | `stop-system-numbered` | 138 | 6.23 | Low |
| 21 | `stop-immediate` | 15 | 5.66 | Very low |
| 13 | `stop-guided-json-comma` | 58 | 3.84 | Low |
| 24 | `stop-unicode` | 38 | 3.57 | Low |
| 18 | `stop-long-phrase` | 44 | 2.64 | Low |
| 12 | `stop-guided-json-value` | 38 | 2.07 | Low |
| 5 | `stop-api-word` | 90 | 1.39 | Very low |
| 3 | `stop-api-newline` | 6 | 1.01 | Suspiciously slow |
| 22 | `stop-special-chars` | 7 | 0.50 | Suspiciously slow |
| 4 | `stop-api-double-newline` | 6 | 0.37 | Suspiciously slow |
| 6 | `stop-api-period` | 6 | 0.13 | **Anomalous** — 46s for 6 tokens |

The low tok/s values on short outputs likely reflect stop-sequence buffer timeout/flushing overhead rather than actual generation speed.

---

## 6. Recommendations

### Likely AFM Bug (Fix in Code)

1. **CRITICAL: Stop sequences match inside `<think>` reasoning content.** Stop sequences should only apply to the visible `content` portion of the response, not the `reasoning_content` inside `<think>...</think>` tags. This affects **22 of 28 passing tests**.
   - **File**: `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` — the streaming loop where stop sequence matching is applied
   - **Fix**: Skip stop-sequence matching while inside `<think>` tags (i.e., while `inThinkBlock` or equivalent state is true). Only start checking stop sequences after `</think>` is emitted.

2. **`response_format: "json_object"` causes Jinja template error.** Two tests fail with `TemplateException error 1`. The `--guided-json` CLI flag works fine, so this is specific to the `response_format` API field.
   - **File**: Likely in the chat template rendering path where `response_format` is passed to Jinja context
   - **Fix**: Check how `response_format: "json_object"` is being passed to the Qwen3.5 chat template. It may be setting an unsupported template variable.

3. **Anomalous latency on short stop matches.** Line 6 (`stop: ["."]`) takes 46 seconds to generate 6 tokens. This suggests the stop-sequence buffer may be waiting for more content before flushing when matching single characters.
   - **File**: Stop sequence buffer logic in `MLXChatCompletionsController.swift`
   - **Fix**: Investigate whether short stop sequences cause the buffer to spin or wait for timeout before confirming a match.

### Model Quality Issue

- None identified. When the model is allowed to complete (lines 7, 8, 14, 15, 21), output quality is excellent. This is entirely an AFM server-side stop sequence handling bug.

### Working Well

| Label | Notes |
|-------|-------|
| `stop-cli-only` | Full correct output, stop not applied to thinking |
| `stop-cli-multi` | Correct code + END output |
| `stop-guided-json-no-match` | Full JSON response (stop never matched) |
| `stop-guided-json-brace` | Good description + hex code |
| `stop-no-match` | Correct "4" answer (stop never matched) |
| `stop-seed-run1/2` | Deterministic reproduction works |
| `stop-streaming` / `stop-non-streaming` | Consistent behavior (both have the thinking bug) |

---

## Summary

The test suite reveals **one critical AFM bug**: stop sequences are being matched against `<think>` reasoning content instead of only against the visible response. This causes 22/28 passing tests to produce empty content. The fix should be localized to the stop-sequence matching logic in the streaming controller — skip matching while inside think tags. Two additional tests fail due to a separate `response_format: "json_object"` Jinja template bug.

<!-- AI_SCORES [{"i":0,"s":2},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":2},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":1},{"i":17,"s":1},{"i":18,"s":2},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":5},{"i":22,"s":2},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2}] -->
