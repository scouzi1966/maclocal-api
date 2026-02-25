# Stop Sequence Test Report — Qwen3.5-35B-A3B-4bit

**Test suite:** `mlx-model-test.sh --prompts Scripts/test-stop-sequences.txt --smart claude,codex`
**AFM version:** v0.9.5 | **Date:** 2026-02-25 | **Model:** `mlx-community/Qwen3.5-35B-A3B-4bit`

---

## 1. Broken Models

**None.** All 28 test variants returned `status: OK`. No load failures or crashes.

---

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Empty Content)
Several tests produced **empty `content`** but valid `reasoning_content`, indicating the model spent its token budget in the thinking phase before the stop sequence triggered on the first output token:

| Line | Label | Stop Seq | Issue |
|------|-------|----------|-------|
| 3 | `stop-api-newline` | `\n` | Stop `\n` fired on first line of output — content empty, reasoning has correct answer |
| 4 | `stop-api-double-newline` | `\n\n` | Stop `\n\n` fired at first paragraph break — content empty, full essay in reasoning |
| 19 | `stop-code-fence` | `` ``` `` | Stop fired on code fence — no code delivered, only reasoning |
| 21 | `stop-immediate` | `"The","I","A"` | Stop `"The"` fired on first word of answer — content empty |
| 23 | `stop-unicode` | `•` | Stop `•` fired on first bullet — content empty |
| 28 | `stop-low-max-tokens` | `"2."` | max_tokens=100 exhausted in thinking — content empty |

**Root cause:** The model's `<think>` reasoning tokens count toward `completion_tokens` but the stop sequence only applies to the visible `content` output. When the first visible token matches a stop, content is empty. This is **expected behavior** for stop sequences, not a bug — but the interaction with thinking models is worth documenting.

### Truncated Output Due to Stop Sequences (Working as Designed)
These are stop sequences correctly truncating output — confirming the feature works:

| Line | Label | Stop | Content (truncated at) |
|------|-------|------|----------------------|
| 1 | `stop-api-single` | `"3."` | `"1. Red\n2. Blue"` — stopped before item 3 ✓ |
| 2 | `stop-api-multi` | `"4.","five"` | `"1. Lion\n2. Elephant\n3. Tiger"` — stopped before item 4 ✓ |
| 5 | `stop-api-period` | `"."` | `"The Sun is a star at the center of our solar system..."` — stopped at first period ✓ |
| 22 | `stop-special-chars` | `"**"` | `"1. The Moon is"` — stopped before bold markdown ✓ |
| 24 | `stop-html-tag` | `"</li>"` | `"```html\n<ul>\n <li>Apple"` — stopped before closing tag ✓ |

### Guided JSON Issues
| Line | Label | Issue |
|------|-------|-------|
| 12 | `stop-guided-json-value` | Stop `"Tokyo"` truncated JSON mid-value: `"[\n  \""` — `is_valid_json: false`. Guided JSON + stop interaction issue. |
| 13 | `stop-guided-json-comma` | Stop `","` prevented JSON output entirely — got markdown profile instead. `--guided-json` schema was ignored. |
| 15 | `stop-guided-json-brace` | Stop `"}"` prevented JSON closing — got readable text instead of JSON. |
| 16 | `stop-json-object-key` | Stop `"age"` truncated JSON: `"{\n  \"name\": \"Carol\",\n  \""` — `is_valid_json: false` |

**Key finding:** `--guided-json` does NOT constrain this thinking model's output format. The model ignores the JSON schema and produces markdown/prose. Stop sequences that conflict with JSON syntax produce broken fragments. This is a **known limitation** — guided JSON likely needs integration at the token-sampling level (like llama.cpp's grammar sampling), not just prompt injection.

### CLI Stop Flag Ignored
| Line | Label | CLI Flag | Expected | Actual |
|------|-------|----------|----------|--------|
| 6 | `stop-cli-only` | `--stop "3."` | Stop at item 2 | **All 10 items output** |
| 7 | `stop-cli-multi` | `--stop "```,END"` | Stop at code fence or END | **Both ``` and END present in output** |

**This is a bug.** The `--stop` CLI flag appears to be non-functional. API-level `stop` works (lines 1-2 confirm), but CLI-passed stop sequences are ignored. Investigate `main.swift` or `Server.swift` for how `--stop` is parsed and passed to the request.

---

## 3. Variant Comparison

### Streaming vs Non-Streaming (lines 10-11)
Both `stop-streaming` and `stop-non-streaming` with identical stop `["3."]`:
- **Identical output:** `"1. Mercury\n2. Venus"` 
- **Identical token counts:** 627 completion tokens
- **Nearly identical speed:** 31.42 vs 31.17 tok/s
- **Verdict:** Streaming/non-streaming parity confirmed ✓

### Seed Reproducibility (lines 27-28)
`stop-seed-run1` and `stop-seed-run2` with `seed: 42`:
- **Identical output:** `"1. Rose\n2. Tulip"`
- **Identical reasoning_content** (character-for-character match)
- **Identical token counts:** 253 tokens each
- **Identical speed:** 28.54 vs 28.52 tok/s
- **Verdict:** Deterministic seeding works perfectly ✓

### CLI + API Stop Merge (line 8)
`stop-cli-api-merge`: CLI `--stop "5."` + API `stop: ["3."]`
- Output: `"1. United States\n2. Canada"` — stopped at `"3."` (API stop)
- CLI stop `"5."` was never reached, so can't confirm merge. But API stop worked.

### CLI + API Dedup (line 9)
`stop-cli-api-dedup`: CLI `--stop "3."` + API `stop: ["3."]`
- Output: `"1. New York City\n2. London"` — stopped at `"3."` ✓
- Deduplication appears fine (no error from duplicate stop strings).

### Temperature Impact (line 26)
`stop-high-temp` at `temperature: 1.0` vs `stop-api-single` at `temperature: 0.0`:
- Both stopped correctly at `"3."`
- High temp: 28.53 tok/s vs low temp: 31.12 tok/s (slightly slower, within normal variance)

---

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 1 | stop-api-single | 5 | 5 | Clean truncation, correct content |
| 2 | stop-api-multi | 5 | 5 | Clean truncation, correct content |
| 3 | stop-api-newline | 4 | 4 | Empty content, but reasoning has perfect answer |
| 4 | stop-api-double-newline | 4 | 4 | Empty content, reasoning has full essay |
| 5 | stop-api-period | 5 | 5 | Clean single-sentence truncation |
| 6 | stop-cli-only | 5 | 5 | Full correct output (stop was ignored) |
| 7 | stop-cli-multi | 5 | 5 | Full correct output (stop was ignored) |
| 8 | stop-cli-api-merge | 5 | 5 | Clean truncation |
| 9 | stop-cli-api-dedup | 5 | 5 | Clean truncation |
| 10 | stop-streaming | 5 | 5 | Clean truncation |
| 11 | stop-non-streaming | 5 | 5 | Clean truncation |
| 12 | stop-guided-json-value | 1 | 2 | Broken JSON fragment `"[\n  \""` |
| 13 | stop-guided-json-comma | 4 | 3 | Good prose but ignored JSON schema |
| 14 | stop-guided-json-no-match | 5 | 5 | Full JSON + text output, stop never matched |
| 15 | stop-guided-json-brace | 5 | 5 | Good prose answer, stop prevented JSON |
| 16 | stop-json-object-key | 1 | 2 | Broken JSON fragment |
| 17 | stop-json-object-no-match | 5 | 5 | Valid JSON output ✓ |
| 18 | stop-long-phrase | 5 | 5 | Two strong paragraphs, stopped before conclusion |
| 19 | stop-multi-word | 5 | 5 | Two good steps, clean stop |
| 20 | stop-code-fence | 3 | 2 | Empty content — reasoning planned but no code delivered |
| 21 | stop-no-match | 5 | 5 | Perfect brief answer |
| 22 | stop-immediate | 3 | 2 | Empty content — answer in reasoning only |
| 23 | stop-special-chars | 2 | 3 | Only `"1. The Moon is"` — truncated mid-sentence |
| 24 | stop-html-tag | 3 | 3 | Partial HTML — has opening `<li>` but stopped before close |
| 25 | stop-unicode | 3 | 2 | Empty content — all facts in reasoning only |
| 26 | stop-system-pirate | 5 | 5 | Excellent pirate persona, detailed response |
| 27 | stop-system-numbered | 5 | 5 | Clean numbered list, stopped at item 4 |
| 28 | stop-high-temp | 5 | 5 | Clean truncation at item 3 |
| 29 | stop-seed-run1 | 5 | 5 | Clean truncation |
| 30 | stop-seed-run2 | 5 | 5 | Identical to run1 |
| 31 | stop-top-p | 5 | 5 | Clean truncation |
| 32 | stop-low-max-tokens | 3 | 2 | Thinking exhausted max_tokens=100, no content |
| 33 | stop-four-max | 5 | 5 | Clean truncation at item 3 |

---

## 5. Performance Summary

| Line | Label | Tokens | Gen Time | tok/s | Notes |
|------|-------|--------|----------|-------|-------|
| 20 | stop-code-fence | 122 | 2.60s | **46.87** | Fastest — short reasoning |
| 5 | stop-api-period | 2041 | 45.90s | 44.47 | |
| 4 | stop-api-double-newline | 695 | 15.45s | 44.98 | |
| 26 | stop-system-pirate | 1113 | 26.19s | 42.50 | |
| 15 | stop-guided-json-brace | 819 | 19.48s | 42.03 | |
| 14 | stop-guided-json-no-match | 1150 | 27.61s | 41.66 | |
| 7 | stop-cli-multi | 326 | 7.80s | 41.81 | |
| 17 | stop-json-object-no-match | 473 | 11.31s | 41.82 | |
| 22 | stop-immediate | 111 | 2.70s | 41.08 | |
| 6 | stop-cli-only | 309 | 7.53s | 41.04 | |
| 5 | stop-api-word | 2579 | 63.26s | 40.77 | Longest generation |
| 21 | stop-no-match | 134 | 3.32s | 40.38 | |
| 25 | stop-unicode | 420 | 10.62s | 39.54 | |
| 13 | stop-guided-json-comma | 628 | 15.94s | 39.41 | |
| 32 | stop-low-max-tokens | 100 | 2.57s | 38.84 | |
| 23 | stop-special-chars | 525 | 13.94s | 37.65 | |
| 3 | stop-api-newline | 199 | 5.58s | 35.69 | |
| 12 | stop-guided-json-value | 657 | 18.68s | 35.17 | |
| 16 | stop-json-object-key | 711 | 20.17s | 35.25 | |
| 19 | stop-multi-word | 585 | 17.15s | 34.12 | |
| 1 | stop-api-single | 435 | 13.98s | 31.12 | |
| 10 | stop-streaming | 627 | 19.95s | 31.42 | |
| 11 | stop-non-streaming | 627 | 20.12s | 31.17 | |
| 8 | stop-cli-api-merge | 253 | 8.07s | 31.37 | |
| 2 | stop-api-multi | 471 | 15.37s | 30.64 | |
| 9 | stop-cli-api-dedup | 334 | 11.03s | 30.27 | |
| 24 | stop-html-tag | 249 | 8.15s | 30.55 | |
| 33 | stop-four-max | 359 | 11.59s | 30.97 | |
| 28 | stop-high-temp | 419 | 14.69s | 28.53 | |
| 29 | stop-seed-run1 | 253 | 8.87s | 28.54 | |
| 30 | stop-seed-run2 | 253 | 8.87s | 28.52 | |
| 31 | stop-top-p | 357 | 12.78s | **27.94** | Slowest |
| 27 | stop-system-numbered | 1010 | 23.28s | 43.38 | |
| 18 | stop-long-phrase | 1119 | 21.13s | **52.95** | Fastest overall |

**Range:** 27.94 – 52.95 tok/s. No suspicious outliers — variance is normal for a 35B-A3B MoE model. Seeded runs show excellent consistency (28.54 vs 28.52).

---

## 6. Recommendations

### Likely AFM Bug

1. **`--stop` CLI flag is non-functional** (Priority: HIGH)
   - Lines 6-7: CLI-passed stop sequences are completely ignored
   - **Action:** Check how `--stop` flag is parsed in `main.swift` and whether it's passed through to the `ChatCompletionRequest.stop` field in the server handler. Likely the CLI argument isn't being forwarded to the API request.

2. **`--guided-json` has no effect on thinking models** (Priority: MEDIUM)
   - Lines 12-13, 15-16: Model ignores JSON schema, produces prose
   - **Action:** This is a known limitation of prompt-injection-based guided JSON. Document that `--guided-json` does not work reliably with thinking models (Qwen3.5). Consider implementing grammar-based constrained decoding for JSON.

3. **Stop sequences on content interact poorly with thinking models** (Priority: LOW/DOCUMENT)
   - Stop sequences like `\n`, `•`, `The` that match the first output token result in empty `content` with full `reasoning_content`
   - **Action:** Document this behavior. Consider whether stop sequences should also apply within reasoning tokens, or add a flag to control this.

### Model Quality Issue

- **None.** The model itself performs well across all tests. Reasoning is thorough and correct. Output quality is high when not truncated by stop sequences.

### Working Well

- **API-level stop sequences** — All API `stop` values work correctly (single, multi, period, newline, phrases, HTML tags, special chars, unicode)
- **Streaming/non-streaming parity** — Identical outputs confirmed
- **Seed determinism** — Perfect reproducibility with `seed: 42`
- **System prompts** — Pirate persona and numbered-list constraints both work perfectly
- **Temperature/top_p** — No degradation at high temperature
- **Stop deduplication** — No errors when CLI and API specify the same stop string
- **`response_format: json_object`** — Works when stop doesn't conflict (line 17: valid JSON)
- **Multi-stop (4 max)** — Correctly stops on first match among 4 sequences

<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":5},{"i":2,"s":5},{"i":3,"s":3},{"i":4,"s":3},{"i":5,"s":4},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":5},{"i":12,"s":2},{"i":13,"s":3},{"i":14,"s":5},{"i":15,"s":4},{"i":16,"s":2},{"i":17,"s":5},{"i":18,"s":4},{"i":19,"s":4},{"i":20,"s":2},{"i":21,"s":5},{"i":22,"s":3},{"i":23,"s":2},{"i":24,"s":3},{"i":25,"s":2},{"i":26,"s":5},{"i":27,"s":4},{"i":28,"s":4},{"i":29,"s":5},{"i":30,"s":5},{"i":31,"s":4},{"i":32,"s":2},{"i":33,"s":5}] -->
