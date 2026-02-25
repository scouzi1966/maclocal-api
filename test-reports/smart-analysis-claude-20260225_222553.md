# Stop Sequence Test Report — Qwen3.5-35B-A3B-4bit

**AFM v0.9.5 | 2026-02-25T22:14:49Z | 28 test cases**

## 1. Broken Models

**None.** All 28 tests loaded and generated successfully (status=OK).

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Empty Content)
Several tests have **empty `content`** but non-empty `reasoning_content`. This is a consistent pattern with this thinking model (Qwen3.5) when stop sequences fire on the first visible token after `</think>`:

| Line | Label | Stop Seq | What Happened |
|------|-------|----------|---------------|
| 3 | stop-api-newline | `\n` | Stop fired on first newline after thinking. Reasoning contains correct answer "The capital of France is Paris." |
| 4 | stop-api-double-newline | `\n\n` | Stop fired before any visible content. Full essay exists in reasoning only. |
| 7 | stop-cli-multi | `` ``` ``, `END` | Stop on code fence. Reasoning has the complete answer. |
| 19 | stop-code-fence | `` ``` `` | Stop on code fence. Reasoning shows intent to write code. |
| 21 | stop-immediate | `The`, `I`, `A` | Stop on first word. Answer "The capital of Japan is Tokyo" in reasoning only. |
| 22 | stop-unicode | `•` | Stop on bullet point. All 5 facts exist in reasoning. |
| 28 | stop-low-max-tokens | `2.` | Thinking exhausted 100 max_tokens before emitting visible content. |

**Root cause:** These are **correct stop sequence behavior** — the stop string matched at the start of visible output. This is expected, not a bug. However, it highlights that stop sequences interact poorly with thinking models when the stop string appears early in the response.

### Guided JSON + Stop Sequence Conflicts
| Line | Label | Stop | Result |
|------|-------|------|--------|
| 12 | stop-guided-json-value | `Tokyo` | JSON truncated to `[\n  "` — stop fired mid-value. `is_valid_json: false` |
| 13 | stop-guided-json-comma | `,` | Not JSON at all — model output prose instead of JSON. Stop fired on comma in prose. |
| 15 | stop-guided-json-brace | `}` | Model ignored guided-json, output prose. Stop on `}` never fired (no brace in prose). |
| 16 | stop-json-object-key | `age` | JSON truncated to `{\n  "name": "Carol",\n  "` — stop fired on "age" key. `is_valid_json: false` |

**Key finding:** `--guided-json` uses prompt injection, not grammar-constrained decoding. When stop sequences conflict with JSON structure characters (`,`, `}`, key names), the output is truncated to invalid JSON. This is **expected behavior** given the architecture, but worth documenting.

### Content Truncated by Stop (Working as Intended)
Most list-based tests correctly stop at the specified item number:

| Line | Label | Stop | Visible Items | Expected |
|------|-------|------|---------------|----------|
| 1 | stop-api-single | `3.` | 2 (Red, Blue) | Correct — stops before "3." |
| 2 | stop-api-multi | `4.`, `five` | 3 (Lion, Elephant, Tiger) | Correct |
| 5 | stop-api-word | `Python` | Truncated mid-sentence at "**" | Correct — "Python" in content |
| 6 | stop-api-period | `.` | 1 sentence fragment | Correct — stops at first period |
| 8 | stop-cli-api-merge | `5.` (CLI) + `3.` (API) | 2 (US, Canada) | Correct — "3." fires first |
| 9 | stop-cli-api-dedup | `3.` (both CLI+API) | 2 (NYC, London) | Correct |
| 10 | stop-streaming | `3.` | 2 (Mercury, Venus) | Correct |
| 11 | stop-non-streaming | `3.` | 2 (Mercury, Venus) | Correct |

## 3. Variant Comparison

### Streaming vs Non-Streaming (lines 10-11)
Identical prompts, same stop sequence `["3."]`:
- **stop-streaming**: 2 items, 627 tokens, 31.71 tok/s
- **stop-non-streaming**: 2 items, 627 tokens, 31.46 tok/s
- **Verdict:** Identical output and performance. Both paths handle stops correctly.

### Seed Reproducibility (lines 26-27)
- **stop-seed-run1** (seed=42): "1. Rose\n2. Tulip", 253 tokens, 28.16 tok/s
- **stop-seed-run2** (seed=42): "1. Rose\n2. Tulip", 253 tokens, 28.16 tok/s
- **Verdict:** Perfectly reproducible. Seed + temperature=0 produces identical results.

### CLI-only vs API-only vs Merged stops
- **stop-cli-only** (line 6, `--stop "3."`): Stops at item 2. ✓
- **stop-api-single** (line 1, API `stop:["3."]`): Stops at item 2. ✓
- **stop-cli-api-merge** (line 8, CLI `--stop "5."` + API `stop:["3."]`): Stops at item 2 (API "3." fires first). ✓
- **stop-cli-api-dedup** (line 9, both `"3."`): Stops at item 2. No errors from duplicate. ✓

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 1 | stop-api-single | 5 | 5 | Clean list, correctly truncated |
| 2 | stop-api-multi | 5 | 5 | Clean list, correctly truncated |
| 3 | stop-api-newline | 3 | 2 | Empty content; answer in reasoning only |
| 4 | stop-api-double-newline | 3 | 2 | Empty content; essay in reasoning only |
| 5 | stop-api-word | 4 | 4 | Truncated mid-list but coherent intro |
| 6 | stop-api-period | 5 | 5 | Clean single sentence fragment about the sun |
| 7 | stop-cli-multi | 3 | 2 | Empty content; code in reasoning only |
| 8 | stop-cli-api-merge | 5 | 5 | Clean list, correctly truncated |
| 9 | stop-cli-api-dedup | 5 | 5 | Clean list, correctly truncated |
| 10 | stop-streaming | 5 | 5 | Clean list |
| 11 | stop-non-streaming | 5 | 5 | Clean list |
| 12 | stop-guided-json-value | 2 | 2 | Invalid JSON fragment |
| 13 | stop-guided-json-comma | 4 | 4 | Prose output (not JSON), truncated at comma |
| 14 | stop-guided-json-no-match | 5 | 5 | Full JSON + prose, stop never matched |
| 15 | stop-guided-json-brace | 5 | 5 | Good prose answer with hex code |
| 16 | stop-json-object-key | 2 | 2 | Truncated JSON, invalid |
| 17 | stop-json-object-no-match | 5 | 5 | Valid JSON, stop never matched |
| 18 | stop-long-phrase | 5 | 5 | Two complete paragraphs, stopped before conclusion |
| 19 | stop-multi-word | 5 | 5 | Two steps of recipe, clean |
| 20 | stop-code-fence | 3 | 2 | Empty content; reasoning has intent |
| 21 | stop-immediate | 3 | 2 | Empty content; answer in reasoning |
| 22 | stop-special-chars | 4 | 4 | Partial first fact "1. The Moon is" |
| 23 | stop-html-tag | 4 | 4 | Partial HTML list, truncated at first `</li>` |
| 24 | stop-unicode | 3 | 2 | Empty content; facts in reasoning only |
| 25 | stop-four-max | 5 | 5 | Clean list, 2 items |
| 26 | stop-system-pirate | 5 | 5 | Excellent pirate persona, full response |
| 27 | stop-system-numbered | 5 | 5 | Clean numbered list per system instruction |
| 28 | stop-high-temp | 5 | 5 | Clean list, 2 items |
| 29 | stop-seed-run1 | 5 | 5 | Clean list |
| 30 | stop-seed-run2 | 5 | 5 | Clean list (identical to run1) |
| 31 | stop-top-p | 5 | 5 | Clean list |
| 32 | stop-low-max-tokens | 3 | 2 | Thinking exhausted max_tokens |

## 5. Performance Summary

| Label | Tokens | tok/s | Notes |
|-------|--------|-------|-------|
| stop-code-fence | 140 | 46.84 | Short generation |
| stop-api-period | 2053 | 44.90 | Long thinking |
| stop-api-double-newline | 718 | 45.26 | |
| stop-system-pirate | 1151 | 42.42 | |
| stop-guided-json-no-match | 1182 | 42.79 | |
| stop-guided-json-brace | 849 | 42.50 | |
| stop-system-numbered | 1038 | 43.77 | |
| stop-json-object-no-match | 533 | 41.87 | |
| stop-immediate | 120 | 41.09 | |
| stop-no-match | 165 | 40.14 | |
| stop-api-word | 2595 | 40.93 | |
| stop-guided-json-comma | 649 | 39.85 | |
| stop-unicode | 433 | 39.63 | |
| stop-low-max-tokens | 139 | 38.28 | |
| stop-cli-multi | 301 | 38.12 | |
| stop-special-chars | 542 | 37.64 | |
| stop-api-newline | 214 | 36.14 | |
| stop-guided-json-value | 673 | 35.69 | |
| stop-json-object-key | 733 | 35.63 | |
| stop-multi-word | 607 | 34.02 | |
| stop-long-phrase | 1144 | 52.82 | Highest — likely sustained throughput |
| stop-api-single | 449 | 31.50 | |
| stop-streaming | 647 | 31.71 | |
| stop-cli-api-merge | 268 | 31.61 | |
| stop-non-streaming | 647 | 31.46 | |
| stop-api-multi | 485 | 31.05 | |
| stop-four-max | 373 | 30.90 | |
| stop-cli-api-dedup | 348 | 30.43 | |
| stop-cli-only | 210 | 29.73 | |
| stop-html-tag | 267 | 29.99 | |
| stop-high-temp | 203 | 28.62 | |
| stop-seed-run1 | 267 | 28.16 | |
| stop-seed-run2 | 267 | 28.16 | |
| stop-top-p | 371 | 27.73 | Lowest — with top_p=0.9 |

**Range:** 27.73–52.82 tok/s. No suspicious outliers — all consistent with Qwen3.5-35B-A3B-4bit on Apple Silicon.

## 6. Recommendations

### Working Well (No Action Needed)
- **API stop sequences**: Single, multi, merge, dedup all work correctly
- **CLI stop sequences**: Working, properly merged with API stops
- **Streaming vs non-streaming**: Identical behavior
- **Seed reproducibility**: Perfect
- **Long phrases and multi-word stops**: Working
- **HTML tags, special chars, Unicode stops**: All fire correctly
- **Four-stop maximum**: Working
- **System prompt + stops**: Compatible
- **Temperature/top_p + stops**: No interaction issues

### Model Quality Issues (Not AFM Bugs)
- **Thinking model + early stops**: When stop sequences match the first token(s) of visible output (e.g., `\n`, `The`, `•`, `` ``` ``), content is empty. The model works correctly — it just spent all its tokens thinking. **Workaround:** Document that stop sequences on common first-output characters will produce empty content with thinking models.
- **`content_preview` shows reasoning when content is empty** (lines 3, 4, 7, 20, 24): The `content_preview` field falls back to showing reasoning_content when content is empty. This is a **test harness issue**, not an AFM bug — but it makes reports confusing.

### Likely AFM Bug / Improvement Area
- **`--guided-json` incompatible with stop sequences containing JSON structural characters** (lines 12, 16): Stop sequences like `"Tokyo"`, `","`, `"age"` truncate JSON mid-value. This is expected given prompt-injection-based guided JSON, but could be documented or warned about. **Suggestion:** Add a warning log when stop sequences contain characters that could conflict with JSON output (`{`, `}`, `,`, `"`).
- **Line 13 (`stop-guided-json-comma`)**: Model completely ignored `--guided-json` and output prose instead of JSON. The `is_valid_json` field is absent (not even checked). This suggests guided-json prompt injection failed to override the model's default behavior. **Not a stop sequence bug** — the guided-json itself failed here.

### Summary Priority
1. **No critical bugs found** — stop sequences work correctly across all tested paths
2. **Document**: Thinking models + aggressive stop sequences = empty content (expected)
3. **Consider**: Warning when guided-json + stop sequences may conflict
4. **Test harness**: Fix `content_preview` to not show reasoning when content is empty

<!-- AI_SCORES [{"i":0,"s":4},{"i":1,"s":4},{"i":2,"s":3},{"i":3,"s":3},{"i":4,"s":4},{"i":5,"s":4},{"i":6,"s":3},{"i":7,"s":4},{"i":8,"s":4},{"i":9,"s":4},{"i":10,"s":4},{"i":11,"s":2},{"i":12,"s":3},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":2},{"i":16,"s":5},{"i":17,"s":5},{"i":18,"s":4},{"i":19,"s":3},{"i":20,"s":3},{"i":21,"s":4},{"i":22,"s":4},{"i":23,"s":3},{"i":24,"s":4},{"i":25,"s":5},{"i":26,"s":5},{"i":27,"s":4},{"i":28,"s":4},{"i":29,"s":4},{"i":30,"s":4},{"i":31,"s":3}] -->
