# Stop Sequences Test Report — Qwen3.5-35B-A3B-4bit

**AFM Version:** v0.9.5-050e836 | **Test Suite:** `test-stop-sequences.txt` | **Date:** 2026-02-25

## 1. Broken Models

**None.** All 28 test variants loaded and generated successfully (status=OK).

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Empty Content)

Several tests produced empty `content` because the model spent its entire token budget in `<think>` reasoning and the stop sequence fired on the first visible token:

| Line | Label | Stop Sequence | Issue |
|------|-------|--------------|-------|
| 3 | `stop-api-newline` | `\n` | Content empty — stop `\n` fired immediately on first visible output line break |
| 4 | `stop-api-double-newline` | `\n\n` | Content empty — stop `\n\n` fired on paragraph break before visible text |
| 20 | `stop-code-fence` | `` ``` `` | Content empty — stop fired before code block output began |
| 22 | `stop-immediate` | `["The","I","A"]` | Content empty — stop `The` fired on first word of visible response |
| 25 | `stop-unicode` | `•` | Content empty — stop `•` fired on first bullet point |
| 29 | `stop-low-max-tokens` | `2.` | Content empty — max_tokens=100 exhausted in reasoning, stop never reached visible content |

**Key pattern:** The stop sequence matching correctly applies to ALL output including `reasoning_content`. This is **by design** for some cases (stop on `\n` truncating after first line) but problematic for thinking models where reasoning text contains the stop string before the visible response begins.

### Stop Sequences Firing Inside Reasoning Content

This is the **critical finding**: Stop sequences match against reasoning/thinking text, not just visible content. For example:
- `stop-api-newline` (line 3): The reasoning contains many `\n` characters — stop fires in reasoning before visible content
- `stop-api-double-newline` (line 4): Same issue with `\n\n`
- `stop-immediate` (line 22): Stop `"The"` matches inside reasoning text `"The capital of Japan is Tokyo.cw"`

**Wait — re-examining:** The content_preview for lines 3 and 4 shows the reasoning content leaking into content_preview, suggesting the content field is truly empty and the model's visible output was intercepted.

### CLI `--stop` Not Working

| Line | Label | CLI Args | Expected | Actual |
|------|-------|----------|----------|--------|
| 6 | `stop-cli-only` | `--stop "3."` | Stop at "3." | **Full 10-item list output** — stop NOT applied |
| 7 | `stop-cli-multi` | `--stop "```,END"` | Stop at ``` or END | **Both ``` and END present in output** — stop NOT applied |

**This is a known AFM bug** (documented in MEMORY.md). CLI `--stop` is not being wired to the MLX controller.

### Truncated JSON Output

| Line | Label | Stop | Content |
|------|-------|------|---------|
| 12 | `stop-guided-json-value` | `Tokyo` | `"[\n  \""` — invalid JSON, stop fired mid-value |
| 13 | `stop-guided-json-comma` | `,` | Markdown output instead of JSON — guided-json not constraining output |
| 16 | `stop-json-object-key` | `age` | `"{\n  \"name\": \"Carol\",\n  \""` — invalid JSON, stop mid-key |

### Partial/Truncated Visible Content

| Line | Label | Stop | Content Produced | Expected |
|------|-------|------|-----------------|----------|
| 5 | `stop-api-word` | `Python` | `"Here are 5 popular programming languages...1. **"` | Truncated mid-response at word "Python" |
| 18 | `stop-long-phrase` | `In conclusion` | 2 paragraphs only | Correct — stopped before 3rd paragraph |
| 19 | `stop-multi-word` | `Step 3` | Steps 1-2 only | Correct behavior |
| 23 | `stop-special-chars` | `**` | `"1. The Moon is"` | Truncated at first bold markdown |
| 24 | `stop-html-tag` | `</li>` | `"```html\n<ul>\n <li>Apple"` | Truncated at first closing li tag |

## 3. Variant Comparison

### Streaming vs Non-Streaming (lines 10-11)
Both `stop-streaming` and `stop-non-streaming` with stop=`["3."]` produced identical results:
- Content: `"1. Mercury\n2. Venus"` — correctly stopped before "3."
- Tokens: 627 each, ~31.8 tok/s — performance parity confirmed

### Seed Reproducibility (lines 27-28)
`stop-seed-run1` and `stop-seed-run2` with seed=42 produced **identical output** (content, token counts, reasoning). Deterministic generation confirmed.

### CLI-only vs API Stop (lines 6 vs 1)
- **API stop** `["3."]` (line 1): Correctly stops at "3." → `"1. Red\n2. Blue"`
- **CLI stop** `--stop "3."` (line 6): **Does NOT stop** → full 10-item list output

### CLI+API Merge (line 8)
- CLI `--stop "5."` + API `stop=["3."]`: Output is `"1. United States\n2. Canada"` — API stop `"3."` fired correctly, but CLI `"5."` was NOT applied (would be irrelevant here since API stop fires first)

### CLI+API Dedup (line 9)
- CLI `--stop "3."` + API `stop=["3."]`: Output `"1. New York City\n2. London"` — API stop worked, CLI redundant

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 1 | stop-api-single | 5 | 5 | Clean list, correctly stopped |
| 2 | stop-api-multi | 5 | 5 | Clean list, correctly stopped |
| 3 | stop-api-newline | 3 | 2 | Empty content, reasoning has answer |
| 4 | stop-api-double-newline | 3 | 2 | Empty content, reasoning has full essay |
| 5 | stop-api-word | 4 | 3 | Truncated mid-sentence but correct stop |
| 6 | stop-cli-only | 5 | 5 | Full output (CLI stop didn't fire) |
| 7 | stop-cli-multi | 5 | 5 | Full output (CLI stop didn't fire) |
| 12 | stop-guided-json-value | 2 | 2 | Invalid JSON fragment |
| 13 | stop-guided-json-comma | 4 | 4 | Good content but not JSON format |
| 16 | stop-json-object-key | 2 | 2 | Invalid JSON fragment |
| 20 | stop-code-fence | 3 | 2 | Empty content, reasoning shows intent |
| 22 | stop-immediate | 3 | 2 | Empty content, reasoning has answer |
| 25 | stop-unicode | 3 | 2 | Empty content, reasoning has full list |
| 29 | stop-low-max-tokens | 2 | 1 | Thinking budget exhausted, no visible output |

## 5. Performance Summary

| Line | Label | Tokens | tok/s | Notes |
|------|-------|--------|-------|-------|
| 20 | stop-code-fence | 122 | 47.67 | Fast — early stop |
| 5 | stop-api-word | 2579 | 41.18 | Extensive reasoning |
| 4 | stop-api-double-newline | 695 | 45.13 | Normal |
| 18 | stop-long-phrase | 1119 | 53.67 | **Fastest** — possible benchmark |
| 6 | stop-cli-only | 309 | 42.18 | Normal |
| 27 | stop-seed-run1 | 253 | 28.91 | Slowest tier |
| 28 | stop-seed-run2 | 253 | 28.94 | Matches run1 |
| 30 | stop-top-p | 357 | 28.42 | **Slowest** — top_p overhead? |
| 29 | stop-low-max-tokens | 100 | 38.82 | Short generation |

**Range:** 28.42 – 53.67 tok/s. No suspicious outliers. Performance is consistent for Qwen3.5-35B-A3B-4bit on Apple Silicon.

## 6. Recommendations

### Likely AFM Bugs

1. **CLI `--stop` not wired to MLX controller** (HIGH PRIORITY)
   - Lines 6, 7 prove CLI stops are ignored
   - Fix in: `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` — need `stop` property and `mergeStopSequences()` 
   - This is a **known bug** per project memory

2. **Stop sequences match inside `<think>` reasoning content** (HIGH PRIORITY)
   - Lines 3, 4, 22, 25 show stops firing on reasoning text before visible output
   - Fix in: Stop sequence buffer should only activate after `</think>` tag transition (same pattern as the existing think-model stop fix documented in MEMORY.md)
   - Affects: `\n`, `\n\n`, common words like "The", special chars `•`, `**`

3. **Guided-JSON + stop sequences produce invalid JSON** (MEDIUM)
   - Lines 12, 16: Stop fires mid-JSON-value, producing invalid fragments
   - Consider: Skip stop-sequence matching inside guided-json mode, or validate JSON completeness before stopping

### Model Quality Issues

- None. Qwen3.5-35B-A3B-4bit generates high-quality, coherent responses across all prompts when stops don't interfere.

### Working Well

- API stop sequences on visible (non-thinking) content: lines 1, 2, 8, 9, 10, 11, 18, 19, 26, 27, 28, 30
- Streaming/non-streaming parity: identical results
- Seed reproducibility: deterministic
- Stop on period (line 5): correctly truncates at sentence boundary
- Multi-stop (line 2): correctly picks earliest match
- Four-stop max (line 26): all four stops processed correctly
- No-match stop (lines 14, 17, 21): model generates full output when stop string absent

<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":5},{"i":2,"s":3},{"i":3,"s":2},{"i":4,"s":3},{"i":5,"s":5},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":2},{"i":12,"s":4},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":2},{"i":16,"s":5},{"i":17,"s":4},{"i":18,"s":4},{"i":19,"s":2},{"i":20,"s":5},{"i":21,"s":2},{"i":22,"s":3},{"i":23,"s":3},{"i":24,"s":2},{"i":25,"s":5},{"i":26,"s":5},{"i":27,"s":5},{"i":28,"s":5},{"i":29,"s":2}] -->
