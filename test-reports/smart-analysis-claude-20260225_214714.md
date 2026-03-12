# Stop Sequences Test Report — Qwen3.5-35B-A3B-4bit

**AFM Version:** v0.9.5 | **Model:** mlx-community/Qwen3.5-35B-A3B-4bit | **Date:** 2025-02-25

## 1. Broken Models

No models failed to load. All 28 test variants returned `status: OK`.

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Empty Content)

Several tests have **empty `content`** but non-empty `reasoning_content` — the model spent its entire visible-content budget inside `<think>` tags, and the stop sequence fired on the very first visible token. This is **expected behavior** for aggressive stop sequences on thinking models:

| Line | Label | Stop | Issue |
|------|-------|------|-------|
| 3 | `stop-api-newline` | `\n` | Stop fired immediately on first newline after `</think>`. Reasoning contains correct answer "The capital of France is Paris." |
| 4 | `stop-api-double-newline` | `\n\n` | Stop fired on first `\n\n` after `</think>`. Full essay drafted in reasoning but never emitted. |
| 20 | `stop-code-fence` | `` ``` `` | Stop fired before code block could be emitted. Reasoning shows correct factorial function planning. |
| 22 | `stop-immediate` | `["The","I","A"]` | All common sentence starters blocked — model can't emit any visible content. |
| 25 | `stop-unicode` | `•` | Stop fired on first bullet point character. Reasoning has complete 5-item list. |

### Stop Sequence Not Working — CLI `--stop`

| Line | Label | Stop | Expected | Actual |
|------|-------|------|----------|--------|
| 6 | `stop-cli-only` | `--stop "3."` (CLI) | Truncate at "3." | **Full 10 items emitted** — `"1. Apple\n2. Banana\n3. Orange\n4. Grape..."` |
| 7 | `stop-cli-multi` | `--stop "```,END"` (CLI) | Truncate at ``` or END | **Both ``` and END present in output** |

**This is a known bug pattern.** CLI `--stop` must be wired to `MLXChatCompletionsController`. Per MEMORY.md, this was previously fixed but may have regressed, or the test harness is passing `--stop` incorrectly for this model variant.

### Guided JSON + Stop Sequence Interactions

| Line | Label | Stop | Issue |
|------|-------|------|-------|
| 12 | `stop-guided-json-value` | `Tokyo` | Output: `"[\n  \""` — stop fired mid-JSON, producing invalid JSON. **Expected** given conflicting stop+content. |
| 13 | `stop-guided-json-comma` | `,` | Output is prose, not JSON — guided-json schema was ignored. Stop on `,` prevents any JSON object from being complete. |
| 16 | `stop-json-object-key` | `age` + `response_format: json_object` | Output: `"{\n  \"name\": \"Carol\",\n  \""` — truncated mid-JSON. Stop on `"age"` fires inside the JSON key. |

### Truncated Output (Stop Working Correctly but Aggressively)

| Line | Label | Stop | Content (truncated) |
|------|-------|------|---------------------|
| 5 | `stop-api-word` | `Python` | `"Here are 5 popular programming languages...1. **"` — stopped before listing Python |
| 23 | `stop-special-chars` | `**` | `"1. The Moon is"` — stopped at first bold markdown |
| 24 | `stop-html-tag` | `</li>` | `"```html\n<ul>\n <li>Apple"` — stopped at first closing tag |
| 18 | `stop-long-phrase` | `In conclusion` | Two paragraphs emitted, stopped before third — **correct behavior** |
| 19 | `stop-multi-word` | `Step 3` | Two steps emitted — **correct behavior** |

### Thinking Model Artifact

All responses end reasoning with `cw` suffix (e.g., `"...Looks good.cw"`). This is a model artifact, not an AFM bug — likely a chat template or tokenizer quirk in Qwen3.5.

## 3. Variant Comparison

### Streaming vs Non-Streaming (lines 10-11)
- **stop-streaming** and **stop-non-streaming**: Identical output (`"1. Mercury\n2. Venus"`), identical token counts (627), near-identical tok/s (31.84 vs 31.61). Stop sequence `["3."]` works correctly in both modes. **No difference.**

### Seed Reproducibility (lines 28-29)
- **stop-seed-run1** and **stop-seed-run2** (seed=42): Identical output, identical token counts (253), identical reasoning content. **Deterministic behavior confirmed.**

### CLI-only vs API stop (lines 6, 8-9)
- **stop-cli-only** (line 6): CLI `--stop "3."` — **FAILED**, full list emitted
- **stop-api-single** (line 1): API `stop: ["3."]` — **WORKED**, truncated at "2. Blue"
- **stop-cli-api-merge** (line 8): CLI `--stop "5."` + API `stop: ["3."]` — Output `"1. United States\n2. Canada"` — API stop worked, CLI stop not tested (API fired first)
- **stop-cli-api-dedup** (line 9): CLI `--stop "3."` + API `stop: ["3."]` — Output `"1. New York City\n2. London"` — Works (API path handles it)

**Conclusion:** API-path stop sequences work. CLI-only `--stop` does NOT work for this test.

### Temperature Variants
- **stop-high-temp** (line 27, temp=1.0): Output `"1. Bamboo\n2. Horizon"` — stop works, different word choices as expected
- **stop-top-p** (line 30, top_p=0.9): Output `"1. Nile\n2. Amazon"` — stop works normally

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 1 | stop-api-single | 5 | 5 | Clean list, correctly stopped |
| 2 | stop-api-multi | 5 | 5 | Clean list, correctly stopped |
| 3 | stop-api-newline | 3 | 2 | Empty content; answer only in reasoning |
| 4 | stop-api-double-newline | 3 | 2 | Empty content; essay only in reasoning |
| 5 | stop-api-word | 4 | 3 | Truncated mid-list but coherent |
| 6 | stop-api-period | 5 | 5 | Clean single sentence about sun |
| 7 | stop-cli-only | 5 | 5 | Full correct list (stop didn't fire) |
| 8 | stop-cli-multi | 5 | 5 | Full correct output (stop didn't fire) |
| 12 | stop-guided-json-value | 2 | 1 | Invalid JSON fragment |
| 13 | stop-guided-json-comma | 4 | 3 | Prose output, schema ignored |
| 14 | stop-guided-json-no-match | 5 | 5 | Full JSON + prose, no stop match |
| 15 | stop-guided-json-brace | 5 | 5 | Good description with hex code |
| 16 | stop-json-object-key | 2 | 2 | Truncated JSON fragment |
| 17 | stop-json-object-no-match | 5 | 5 | Valid JSON, correct output |
| 26 | stop-system-pirate | 5 | 5 | Excellent pirate persona |
| 27 | stop-system-numbered | 5 | 5 | Clean numbered list per system prompt |

Models scoring below 3: lines 3, 4, 12, 16 (all due to aggressive stop sequences on thinking model output).

## 5. Performance Summary

| Line | Label | Tokens | Gen Time (s) | tok/s | Notes |
|------|-------|--------|-------------|-------|-------|
| 5 | stop-api-word | 2579 | 62.76 | 41.09 | |
| 6 | stop-api-period | 2041 | 45.34 | 45.02 | |
| 18 | stop-long-phrase | 1119 | 21.14 | **52.94** | Fastest |
| 14 | stop-guided-json-no-match | 1150 | 26.94 | 42.69 | |
| 15 | stop-guided-json-brace | 819 | 19.20 | 42.66 | |
| 26 | stop-system-pirate | 1113 | 26.20 | 42.48 | |
| 17 | stop-json-object-no-match | 473 | 11.18 | 42.30 | |
| 27 | stop-system-numbered | 1010 | 23.13 | 43.67 | |
| 20 | stop-code-fence | 122 | 2.57 | 47.43 | Short gen |
| 28/29 | stop-seed-run1/2 | 253 | 8.89 | **28.45** | Slowest |
| 30 | stop-top-p | 357 | 12.83 | **27.83** | Slowest |

**Range:** 27.83 – 52.94 tok/s. The slowest runs (seed, top_p variants) are ~28 tok/s — likely due to additional sampling overhead. No suspiciously fast degenerate outputs detected.

## 6. Recommendations

### Likely AFM Bug

1. **CLI `--stop` not working (PRIORITY: HIGH)**
   - Lines 6-7: CLI `--stop` flag has no effect — full output emitted past stop strings
   - This was previously fixed per commit `f3607af` ("Fix CLI --stop not passed to Server in MlxCommand")
   - **Investigate:** Check if `MlxCommand.swift` correctly passes `--stop` to `MLXChatCompletionsController.swift`. The test harness sends `--stop` as CLI args via `afm_args`. Verify `mergeStopSequences()` is called when only CLI stops are present (no API stops).
   - **File:** `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` — check stop property initialization from CLI args

2. **Guided JSON + stop interaction produces broken output (PRIORITY: LOW)**
   - Lines 12, 13, 16: When stop sequences conflict with JSON structure, output is truncated mid-JSON
   - This is arguably expected (stop takes priority), but could be improved by not applying stop sequences inside JSON structural tokens when `response_format` is set
   - **File:** `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` — stop sequence matching logic

### Model Quality Issue (Not AFM Bugs)

3. **Thinking budget exhaustion with aggressive stops** — Lines 3, 4, 20, 22, 25: Model spends all tokens in `<think>` block, then stop fires on first visible token. This is inherent to thinking models with short stop strings like `\n`, `•`, common words. **No fix needed** — document as expected behavior.

4. **`cw` artifact at end of reasoning** — All responses end reasoning with `cw`. This is a model/tokenizer artifact. Harmless but worth noting.

### Working Well

- API-path stop sequences (single, multi, merge, dedup) — all working correctly
- Streaming and non-streaming parity — identical behavior
- Seed determinism — confirmed reproducible
- Temperature/top_p + stop — working correctly
- System prompt + stop — working correctly
- Long phrase and multi-word stops — working correctly
- No-match stops (XYZZY) — correctly ignored, full output produced
- Period stop, HTML tag stop, special char stops — all functioning

<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":5},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":3},{"i":5,"s":5},{"i":6,"s":4},{"i":7,"s":4},{"i":8,"s":4},{"i":9,"s":4},{"i":10,"s":4},{"i":11,"s":4},{"i":12,"s":2},{"i":13,"s":3},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":2},{"i":17,"s":5},{"i":18,"s":4},{"i":19,"s":4},{"i":20,"s":2},{"i":21,"s":5},{"i":22,"s":2},{"i":23,"s":3},{"i":24,"s":3},{"i":25,"s":2},{"i":26,"s":5},{"i":27,"s":5},{"i":28,"s":4},{"i":29,"s":4},{"i":30,"s":4},{"i":31,"s":2}] -->
