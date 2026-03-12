# Stop Sequences Test Report — Qwen3.5-35B-A3B-4bit

**AFM Version:** v0.9.5 | **Date:** 2026-02-25 | **Model:** mlx-community/Qwen3.5-35B-A3B-4bit

## 1. Broken Models

| Line | Label | Error | Classification |
|------|-------|-------|---------------|
| 16 | `stop-json-object-key` | `Jinja.TemplateException error 1` | **AFM bug** |
| 17 | `stop-json-object-no-match` | `Jinja.TemplateException error 1` | **AFM bug** |

Both failures occur when `response_format: "json_object"` is combined with stop sequences. The Jinja template error suggests the chat template rendering fails when JSON object mode is requested. This is **not a model issue** — the model loads fine in all other tests. The bug is in how AFM passes `response_format` to the Jinja template engine.

**Likely code path:** `MLXModelService.swift` or vendor template rendering code — the `response_format: json_object` option likely injects a system prompt or modifies template variables in a way that breaks Qwen3.5's Jinja template.

## 2. Anomalies & Red Flags

### Thinking-Budget Exhaustion (Empty Content)
Several tests produced empty `content` because the model spent its entire token budget in `<think>` reasoning:

| Line | Label | completion_tokens | max_tokens | Issue |
|------|-------|------------------|------------|-------|
| 3 | `stop-api-newline` | 199 | 4096 | Stop `\n` fired inside reasoning output, truncating before actual response |
| 4 | `stop-api-double-newline` | 695 | 4096 | Stop `\n\n` fired inside reasoning, no content emitted |
| 21 | `stop-code-fence` | 122 | 4096 | Stop `` ``` `` fired during reasoning before code block |
| 23 | `stop-immediate` | 111 | 4096 | Stop `["The","I","A"]` caught first word of response |
| 25 | `stop-unicode` | 420 | 4096 | Stop `•` fired inside reasoning |
| 30 | `stop-low-max-tokens` | 100 | 100 | Budget exhausted in reasoning (expected) |

**Critical finding:** Stop sequences are matching against `reasoning_content` (the `<think>` block), not just the visible `content`. Lines 3, 4, 21, 25 all have stop strings that appear naturally in the model's thinking process. The stop sequence implementation should only match against tokens *after* the `</think>` tag.

### Truncated by Stop Sequences (Working as Designed but Partial Output)
These tests correctly stopped at the specified sequence, producing intentionally partial output:

| Line | Label | Content | Stop | Notes |
|------|-------|---------|------|-------|
| 1 | `stop-api-single` | `"1. Red\n2. Blue"` | `"3."` | Correct — stopped before item 3 |
| 2 | `stop-api-multi` | `"1. Lion\n2. Elephant\n3. Tiger"` | `["4.","five"]` | Correct — stopped before item 4 |
| 5 | `stop-api-word` | `"Here are 5 popular...1. **"` | `"Python"` | Stopped mid-list, content truncated at "Python" |
| 6 | `stop-api-period` | `"The Sun is...on Earth"` | `"."` | Stopped at first period — only one sentence fragment |
| 24 | `stop-special-chars` | `"1. The Moon is"` | `"**"` | Stopped at first bold markdown |

### Guided JSON Issues
| Line | Label | Content | Issue |
|------|-------|---------|-------|
| 12 | `stop-guided-json-value` | `'[\n  "'` | Stop `"Tokyo"` fired, but `is_valid_json: false` — guided JSON + stop sequences conflict |
| 13 | `stop-guided-json-comma` | Prose output, not JSON | Model ignored guided-json schema, produced markdown profile instead |

### CLI Stop `--stop "3."` Not Working (Line 6)
Line 6 (`stop-cli-only`): The `--stop "3."` was passed via `afm_args`, but the content shows all 10 items (`1. Apple` through `10. Peach`). **The CLI `--stop` flag was ignored.** Compare with line 1 (`stop-api-single`) where the API-level `stop: ["3."]` correctly truncated at item 2.

## 3. Variant Comparison

### Streaming vs Non-Streaming (Lines 10-11)
- `stop-streaming` (line 10): `"1. Mercury\n2. Venus"` — stopped at `"3."` ✓
- `stop-non-streaming` (line 11): `"1. Mercury\n2. Venus"` — stopped at `"3."` ✓
- Identical output and reasoning. tok/s: 28.47 (streaming) vs 31.26 (non-streaming). Both working correctly.

### Seed Reproducibility (Lines 27-28)
- `stop-seed-run1` and `stop-seed-run2`: **Identical** content, reasoning, token counts (253), and timing (8.99s). Seed=42 produces deterministic output. ✓

### CLI vs API Stop (Lines 6 vs 1)
- API `stop: ["3."]` (line 1): Correctly stops at item 2. ✓
- CLI `--stop "3."` (line 6): **Does NOT stop** — outputs all 10 items. **Bug.**

### CLI+API Merge (Line 8)
- CLI `--stop "5."` + API `stop: ["3."]`: Stopped at item 2 (API stop `"3."` fired first). API stops work; unclear if CLI stop would have fired at `"5."`.

## 4. Quality Assessment

| Line | Label | Coherence | Relevance | Notes |
|------|-------|-----------|-----------|-------|
| 1 | stop-api-single | 5 | 5 | Clean list, correctly truncated |
| 2 | stop-api-multi | 5 | 5 | Clean list, correctly truncated |
| 3 | stop-api-newline | 4 | 4 | Empty content but reasoning has correct answer |
| 4 | stop-api-double-newline | 4 | 4 | Empty content but reasoning has full essay |
| 5 | stop-api-word | 4 | 4 | Truncated mid-response but content was on-track |
| 6 | stop-api-period | 4 | 4 | Good single sentence fragment |
| 6* | stop-cli-only | 5 | 5 | Perfect output (but stop was supposed to fire) |
| 7 | stop-cli-multi | 5 | 5 | Perfect code block + END |
| 8 | stop-cli-api-merge | 5 | 5 | Correctly truncated |
| 9 | stop-cli-api-dedup | 5 | 5 | Correctly truncated |
| 10 | stop-streaming | 5 | 5 | Correctly truncated |
| 11 | stop-non-streaming | 5 | 5 | Correctly truncated |
| 14 | stop-guided-json-no-match | 5 | 5 | Full correct output, stop never matched |
| 15 | stop-guided-json-brace | 5 | 5 | Good description with hex code |
| 18 | stop-long-phrase | 5 | 5 | Excellent 2-paragraph essay |
| 19 | stop-multi-word | 5 | 5 | Clean 2-step recipe |
| 22 | stop-no-match | 5 | 5 | Perfect "4" answer |
| 26 | stop-system-pirate | 5 | 5 | Excellent pirate voice |
| 27 | stop-system-numbered | 5 | 5 | Clean numbered list |

Models scoring below 3: None (excluding FAIL lines).

## 5. Performance Summary

| Line | Label | Tokens | Time (s) | tok/s | Notes |
|------|-------|--------|----------|-------|-------|
| 18 | stop-long-phrase | 1144 | 21.13 | **52.96** | Highest tok/s |
| 20 | stop-code-fence | 140 | 2.59 | 47.16 | |
| 4 | stop-api-double-newline | 718 | 15.57 | 44.64 | |
| 6 | stop-api-period | 2053 | 46.21 | 44.17 | |
| 15 | stop-guided-json-brace | 849 | 19.17 | 42.71 | |
| 14 | stop-guided-json-no-match | 1182 | 27.10 | 42.43 | |
| 26 | stop-system-pirate | 1151 | 26.26 | 42.38 | |
| 27 | stop-system-numbered | 1038 | 23.13 | 43.66 | |
| 6* | stop-cli-only | 348 | 7.45 | 41.47 | |
| 7 | stop-cli-multi | 361 | 7.87 | 41.45 | |
| 23 | stop-immediate | 120 | 2.70 | 41.09 | |
| 5 | stop-api-word | 2595 | 63.84 | 40.39 | |
| 22 | stop-no-match | 165 | 3.35 | 40.04 | |
| 13 | stop-guided-json-comma | 649 | 15.83 | 39.67 | |
| 25 | stop-unicode | 433 | 10.70 | 39.24 | |
| 30 | stop-low-max-tokens | 139 | 2.61 | 38.29 | |
| 24 | stop-special-chars | 542 | 13.98 | 37.56 | |
| 3 | stop-api-newline | 214 | 5.61 | 35.46 | |
| 12 | stop-guided-json-value | 673 | 18.51 | 35.49 | |
| 19 | stop-multi-word | 607 | 17.29 | 33.84 | |
| 11 | stop-non-streaming | 647 | 20.05 | 31.26 | |
| 1 | stop-api-single | 449 | 14.17 | 30.70 | |
| 26a | stop-four-max | 373 | 11.68 | 30.75 | |
| 2 | stop-api-multi | 485 | 15.45 | 30.49 | |
| 9 | stop-cli-api-dedup | 348 | 11.23 | 29.75 | |
| 21 | stop-html-tag | 267 | 8.37 | 29.75 | |
| 28 | stop-high-temp | 301 | 9.83 | 29.00 | |
| 10 | stop-streaming | 647 | 22.02 | 28.47 | |
| 27a | stop-seed-run1 | 267 | 8.99 | 28.13 | |
| 27b | stop-seed-run2 | 267 | 8.99 | 28.15 | |
| 8 | stop-cli-api-merge | 268 | 9.10 | 27.80 | |
| 29 | stop-top-p | 371 | 13.00 | 27.45 | |

Range: 27.45–52.96 tok/s. All within normal range for this MoE model. No suspicious outliers.

## 6. Recommendations

### Likely AFM Bug (Fix in Code)

1. **`response_format: json_object` crashes with Jinja error** (Lines 16-17, Priority: HIGH)
   - Jinja template fails when `response_format` is set to `json_object`
   - Investigate how `response_format` modifies the chat template in `MLXModelService.swift` or `ChatCompletionsController`
   - May need to handle Qwen3.5's template differently for JSON mode

2. **Stop sequences fire inside `<think>` reasoning blocks** (Lines 3, 4, 21, 23, 25, Priority: HIGH)
   - Stop strings like `\n`, `\n\n`, `` ``` ``, `•`, `The` match against reasoning tokens
   - Fix: Only apply stop sequence matching to tokens emitted *after* `</think>` close tag
   - Code path: `MLXChatCompletionsController.swift` stop sequence detection logic

3. **CLI `--stop` flag ignored** (Line 6, Priority: MEDIUM)
   - `--stop "3."` via `afm_args` did not truncate output
   - API-level `stop` field works correctly (line 1)
   - Check CLI argument parsing in `main.swift` — the `--stop` flag may not be wired to the generation pipeline

4. **Guided JSON schema ignored** (Lines 12-13, Priority: LOW)
   - `--guided-json` with stop sequences produces prose instead of JSON
   - Model ignores the schema constraint when stop sequences are also present
   - May be a priority/interaction issue between guided generation and stop handling

### Model Quality Issue
- None. Qwen3.5-35B-A3B-4bit produces coherent, relevant output in all non-broken tests.

### Working Well
- API-level stop sequences with string/multi-string matching ✓
- Streaming and non-streaming parity ✓
- Seed reproducibility ✓
- System prompts respected ✓
- Temperature/top_p variants work ✓
- Stop at phrase boundaries ("In conclusion", "Step 3") ✓
- Combined CLI+API stop deduplication ✓

<!-- AI_SCORES [{"i":0,"s":4},{"i":1,"s":4},{"i":2,"s":3},{"i":3,"s":3},{"i":4,"s":3},{"i":5,"s":4},{"i":6,"s":4},{"i":7,"s":5},{"i":8,"s":4},{"i":9,"s":4},{"i":10,"s":4},{"i":11,"s":4},{"i":12,"s":2},{"i":13,"s":3},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":1},{"i":17,"s":1},{"i":18,"s":5},{"i":19,"s":4},{"i":20,"s":3},{"i":21,"s":5},{"i":22,"s":3},{"i":23,"s":4},{"i":24,"s":3},{"i":25,"s":5},{"i":26,"s":4},{"i":27,"s":4},{"i":28,"s":4},{"i":29,"s":4},{"i":30,"s":2}] -->
