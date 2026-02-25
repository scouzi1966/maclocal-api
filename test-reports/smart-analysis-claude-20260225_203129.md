# Stop Sequences Test Report — Qwen3.5-35B-A3B-4bit

**AFM Version:** v0.9.5-0cfba17 | **Model:** mlx-community/Qwen3.5-35B-A3B-4bit | **Date:** 2026-02-25

## 1. Broken Models

| Variant | Error | Type |
|---------|-------|------|
| `stop-cli-only` | `Unknown option '--stop'. Did you mean '--top-p'?` | **AFM bug** |
| `stop-cli-multi` | `Unknown option '--stop'. Did you mean '--top-p'?` | **AFM bug** |
| `stop-cli-api-merge` | `Unknown option '--stop'. Did you mean '--top-p'?` | **AFM bug** |
| `stop-cli-api-dedup` | `Unknown option '--stop'. Did you mean '--top-p'?` | **AFM bug** |
| `stop-json-object-key` | `Jinja.TemplateException error 1` | **Known AFM bug** |
| `stop-json-object-no-match` | `Jinja.TemplateException error 1` | **Known AFM bug** |

**CLI `--stop` flag (lines 7-10):** The `--stop` CLI flag is not recognized by the `afm mlx` subcommand. The ArgumentParser doesn't have a `--stop` option defined. This is a **known issue** per MEMORY.md — the flag was previously identified as needing to be wired to the MLX controller.

**Jinja template errors (lines 16-17):** These occur when `response_format: json_object` is combined with stop sequences. Per MEMORY.md, this is the known Qwen3.5 bug where JSON system message injection creates a second `.system()` message that the chat template rejects.

## 2. Anomalies & Red Flags

### Critical: Stop sequences fire inside `<think>` reasoning, not just visible content

Nearly every API stop sequence test produces **empty `content`** with reasoning truncated mid-thought. The stop strings are matching against the `reasoning_content` (inside `<think>` tags), not waiting for visible output.

| Variant | Stop string | Where it matched | Evidence |
|---------|------------|-----------------|----------|
| `stop-api-newline` | `\n` | First newline in reasoning | Only 6 tokens, reasoning = `"Thinking Process:"` |
| `stop-api-double-newline` | `\n\n` | First double-newline in reasoning | Only 6 tokens |
| `stop-api-period` | `.` | First period in reasoning (`"1"` → `"1."`) | Only 6 tokens, reasoning = `"Thinking Process:\n\n1"` |
| `stop-api-word` | `"Python"` | Would appear in reasoning about programming languages | 90 tokens, stopped in reasoning |
| `stop-immediate` | `"The"`, `"I"`, `"A"` | Common words in reasoning | 8 tokens |
| `stop-special-chars` | `**` | Markdown bold in reasoning | 7 tokens |
| `stop-unicode` | `•` | Bullet character in reasoning | 38 tokens |
| `stop-long-phrase` | `"In conclusion"` | Quoted in reasoning analysis | 44 tokens |
| `stop-code-fence` | `` ``` `` | Never reached visible content | 122 tokens of reasoning only |
| `stop-html-tag` | `</li>` | Appeared in reasoning HTML example | 107 tokens |
| `stop-system-pirate` | `"Arrr"` | Pirate vocabulary in reasoning | 374 tokens of reasoning only |

**This is the primary bug.** Stop sequences must only match against visible content (after `</think>`), not reasoning tokens. Per MEMORY.md, this was previously identified and fixed — but these results suggest the fix is incomplete or regressed.

### Thinking-Budget Exhaustion Cases

Several tests stopped early because the stop string was encountered in reasoning, producing no visible `content`. These are NOT budget exhaustion — they're stop-sequence-in-reasoning bugs.

### One legitimate success

| Variant | Stop | Result |
|---------|------|--------|
| `stop-no-match` (line 20) | `"XYZZY_NEVER_MATCH"` | Content = `"4"` — correct answer, stop never triggered |
| `stop-guided-json-no-match` (line 14) | `"XYZZY"` | Full response with JSON — stop never triggered, worked correctly |
| `stop-guided-json-brace` (line 15) | `"}"` | Full prose response — stop fired on `}` after visible content was emitted |

## 3. Variant Comparison

### Streaming vs Non-Streaming (lines 10-11)
Both `stop-streaming` and `stop-non-streaming` with stop=`["3."]` produced identical results: empty content, 161 reasoning tokens, stop fired in reasoning. **Both paths have the same bug.**

### Seed Reproducibility (lines 27-28)
`stop-seed-run1` and `stop-seed-run2` (seed=42) are **identical** — same token counts (80), same tok/s (~30.3), same content. Deterministic generation works.

### Temperature Effect (line 26)
`stop-high-temp` (temp=1.0) vs `stop-api-single` (temp=0.0) with same stop=`["3."]`: Both stopped in reasoning. Temperature doesn't affect the bug.

## 4. Quality Assessment

| Line | Variant | Coherence | Relevance | Notes |
|------|---------|-----------|-----------|-------|
| 1 | stop-api-single | 4 | 1 | Good reasoning, but no visible answer produced |
| 2 | stop-api-multi | 4 | 1 | Reasoning shows correct list started, but no content |
| 3 | stop-api-newline | 1 | 1 | Only "Thinking Process:" — effectively empty |
| 4 | stop-api-double-newline | 1 | 1 | Same — trivially truncated |
| 5 | stop-api-word | 3 | 1 | Decent reasoning analysis, no answer |
| 6 | stop-api-period | 1 | 1 | Only "Thinking Process:\n\n1" |
| 14 | stop-guided-json-no-match | 5 | 5 | Full, correct response |
| 15 | stop-guided-json-brace | 5 | 5 | Full, correct response with hex code |
| 20 | stop-no-match | 5 | 5 | Clean, correct "4" |

## 5. Performance Summary

| Line | Variant | Tokens | Time (s) | tok/s | Notes |
|------|---------|--------|----------|-------|-------|
| 15 | stop-guided-json-brace | 849 | 19.0 | 43.13 | Normal |
| 14 | stop-guided-json-no-match | 1182 | 26.6 | 43.18 | Normal |
| 20 | stop-no-match | 165 | 3.3 | 40.59 | Normal |
| 5 | stop-api-word | 106 | 2.3 | 39.66 | Normal |
| 19 | stop-multi-word | 133 | 3.4 | 33.16 | Normal |
| 25 | stop-system-numbered | 166 | 3.7 | 36.93 | Normal |
| 10 | stop-streaming | 181 | 4.7 | 33.97 | Normal |
| 11 | stop-non-streaming | 181 | 4.6 | 34.93 | Normal |
| 3 | stop-api-newline | 21 | 0.5 | 13.31 | Low — stopped very early |
| 6 | stop-api-period | 18 | 0.4 | 17.06 | Low — stopped very early |
| 23 | stop-special-chars | 24 | 0.4 | 16.03 | Low — stopped very early |

Low tok/s on short generations is expected (startup overhead dominates). No suspicious outliers.

## 6. Recommendations

### Likely AFM Bug — High Priority

1. **Stop sequences matching inside `<think>` reasoning (CRITICAL)**
   - **File:** `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift`
   - **Issue:** Stop sequence matching applies to ALL generated text including reasoning tokens inside `<think>...</think>`. It must only match against visible content after `</think>`.
   - **Fix:** In both streaming and non-streaming paths, the stop-sequence buffer/matcher must be gated by the `insideThink` state. Only start checking stop sequences once `</think>` has been emitted and visible content is being produced. Per MEMORY.md, a previous fix existed for this — check if it regressed or was incomplete.
   - **Affected tests:** Lines 1-6, 10-13, 18-19, 21-28 (nearly all)

2. **CLI `--stop` flag not implemented for MLX subcommand**
   - **File:** `Sources/MacLocalAPI/main.swift` (ArgumentParser definition for MLX subcommand)
   - **Issue:** `--stop` is not a recognized option. Per MEMORY.md, this was previously identified.
   - **Fix:** Add `@Option var stop: String?` to the MLX command, wire it through `mergeStopSequences()` to the controller.
   - **Affected tests:** Lines 7-10

3. **`response_format: json_object` + Qwen3.5 template error**
   - **File:** `Sources/MacLocalAPI/Models/MLXModelService.swift` (system message injection)
   - **Issue:** JSON format instruction creates a second system message that Qwen3.5's Jinja template rejects.
   - **Fix:** Append JSON instruction to existing system message instead of adding a new one. (Known issue per MEMORY.md.)
   - **Affected tests:** Lines 16-17

### Working Well

| Variant | Notes |
|---------|-------|
| `stop-no-match` (line 20) | Stop string never appears, full correct response |
| `stop-guided-json-no-match` (line 14) | Full response, guided JSON works when stop doesn't match |
| `stop-guided-json-brace` (line 15) | Stop fires on `}` in visible content correctly |
| `stop-seed-run1/run2` (lines 27-28) | Deterministic generation confirmed |

### Model Quality Issue

None — Qwen3.5-35B-A3B-4bit generates coherent, well-structured reasoning. The issues are entirely AFM server bugs.

---

**Bottom line:** The #1 fix needed is gating stop-sequence matching to only apply after `</think>` in thinking models. This single bug causes 20+ test failures. The CLI `--stop` flag and `json_object` template issues are secondary but also need fixing.

<!-- AI_SCORES [{"i":0,"s":2},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":1},{"i":7,"s":1},{"i":8,"s":1},{"i":9,"s":1},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":1},{"i":17,"s":1},{"i":18,"s":2},{"i":19,"s":2},{"i":20,"s":5},{"i":21,"s":2},{"i":22,"s":2},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2},{"i":30,"s":2}] -->
