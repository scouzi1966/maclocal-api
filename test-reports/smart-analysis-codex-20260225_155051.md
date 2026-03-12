## AFM Compatibility Report (Qwen3.5-35B-A3B-4bit variants)

### 1) Broken Models
All hard failures are **feature-path failures**, not model-load failures.

| Error type | Variants | Count | Likely cause | AFM bug vs model incompat |
|---|---|---:|---|---|
| CLI option parse error: `Unknown option '--stop'` | `stop-cli-only`, `stop-cli-multi`, `stop-cli-api-merge`, `stop-cli-api-dedup` | 4 | AFM CLI does not accept `--stop` though tests expect it | **Likely AFM bug / CLI-contract mismatch** |
| `response_format=json_object` + stop => `Jinja.TemplateException` 400 | `stop-json-object-key`, `stop-json-object-no-match` | 2 | Template/rendering path crashes when combining stop + json_object | **Likely AFM bug** |

### 2) Anomalies & Red Flags
- **Major regression:** most `status=OK` runs return **empty `content`** and only internal reasoning text in `reasoning_content`.
  - Representative snippet (first ~100 chars): `"Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Task: List 10 colors..."`
  - Affects nearly all stop tests (API, system, seed, top-p, streaming).
- **Stop-sequence over-triggering/truncation:** many outputs stop inside planning text:
  - `stop-api-newline`: snippet `"Thinking Process:"`
  - `stop-api-period`: snippet `"Thinking Process:\n\n1"`
  - `stop-immediate`: snippet `"Thinking Process:\n\n1.  **"`
- **Guided JSON not enforced consistently:**
  - `stop-guided-json-no-match` returns freeform prose + markdown JSON block, not schema-constrained object.
  - `stop-guided-json-brace` returns plain paragraph (`Blue is a primary color...`) despite guided-json schema.
- **No clear repetition loops/gibberish** observed.
- **Thinking-budget exhaustion pattern:** **not observed** (empty `content` was generally from stop-trigger behavior, not near-max-token reasoning-only runs).

### 3) Variant Comparison
| Variant pair/group | Better | Worse | Notes |
|---|---|---|---|
| `stop-streaming` vs `stop-non-streaming` | Tie | Tie | Nearly identical reasoning-only output; issue is independent of streaming mode. |
| `stop-seed-run1` vs `stop-seed-run2` | Tie | Tie | Deterministic (same output), but both reasoning-only and truncated. |
| Guided JSON variants | `stop-guided-json-brace` (has usable final content) | `stop-guided-json-value`, `stop-guided-json-comma` (empty content), `...no-match` (schema drift) | Structured-output path is inconsistent under stop constraints. |
| Stop matching | `stop-no-match` | Most explicit stop variants | When stop token never matches, model can return actual answer (`"4"`). |

### 4) Quality Assessment (Coherence/Relevance 1-5)
Flagged below 3 on either metric are marked `⚠`.

| Variant | Coh | Rel | Notes |
|---|---:|---:|---|
| stop-api-single | 3 | 2 ⚠ | Reasoning only, no user answer |
| stop-api-multi | 3 | 2 ⚠ | Reasoning only |
| stop-api-newline | 2 ⚠ | 1 ⚠ | Immediate truncation |
| stop-api-double-newline | 2 ⚠ | 1 ⚠ | Immediate truncation |
| stop-api-word | 3 | 2 ⚠ | Reasoning only |
| stop-api-period | 2 ⚠ | 1 ⚠ | Truncated at period stop |
| stop-streaming | 3 | 2 ⚠ | Reasoning only |
| stop-non-streaming | 3 | 2 ⚠ | Reasoning only |
| stop-guided-json-value | 2 ⚠ | 1 ⚠ | Truncated + invalid JSON (`is_valid_json=false`) |
| stop-guided-json-comma | 2 ⚠ | 1 ⚠ | Truncated reasoning only |
| stop-guided-json-no-match | 4 | 3 | Usable but ignores schema intent |
| stop-guided-json-brace | 4 | 4 | Good answer content, but not schema-shaped |
| stop-long-phrase | 2 ⚠ | 1 ⚠ | Truncated at `"In conclusion"` stop |
| stop-multi-word | 3 | 2 ⚠ | Stops before usable final answer |
| stop-code-fence | 3 | 2 ⚠ | Plans code, outputs none |
| stop-no-match | 4 | 5 | Correct final answer `"4"` |
| stop-immediate | 2 ⚠ | 1 ⚠ | Immediate stop hit |
| stop-special-chars | 2 ⚠ | 1 ⚠ | Immediate stop hit |
| stop-html-tag | 3 | 2 ⚠ | Stops in planning markup |
| stop-unicode | 2 ⚠ | 1 ⚠ | Immediate stop hit |
| stop-four-max | 3 | 2 ⚠ | Stops at numbered token |
| stop-system-pirate | 3 | 2 ⚠ | Persona planning only |
| stop-system-numbered | 3 | 2 ⚠ | Planning only |
| stop-high-temp | 3 | 2 ⚠ | Planning only |
| stop-seed-run1 | 3 | 2 ⚠ | Planning only |
| stop-seed-run2 | 3 | 2 ⚠ | Planning only |
| stop-top-p | 3 | 2 ⚠ | Planning only |
| stop-low-max-tokens | 2 ⚠ | 1 ⚠ | Stops at `"2."` before answer |

### 5) Performance Summary (sorted by tokens/sec)
| Variant | tok/s | Status | Flag |
|---|---:|---|---|
| stop-code-fence | 44.15 | OK | Suspiciously fast for reasoning-only/no final answer |
| stop-guided-json-no-match | 41.29 | OK | High throughput with schema drift |
| stop-guided-json-brace | 41.28 | OK | High throughput; structured-output mismatch |
| stop-no-match | 39.80 | OK | Healthy content return |
| stop-api-word | 38.38 | OK | Reasoning-only |
| stop-system-numbered | 35.37 | OK | Reasoning-only |
| stop-streaming | 33.59 | OK | Reasoning-only |
| stop-non-streaming | 32.20 | OK | Reasoning-only |
| stop-multi-word | 31.92 | OK | Truncated |
| stop-top-p | 31.40 | OK | Reasoning-only |
| stop-long-phrase | 31.28 | OK | Truncated |
| stop-guided-json-comma | 30.67 | OK | Truncated |
| stop-html-tag | 30.22 | OK | Truncated |
| stop-unicode | 30.02 | OK | Immediate stop |
| stop-api-multi | 28.43 | OK | Reasoning-only |
| stop-high-temp | 28.32 | OK | Reasoning-only |
| stop-seed-run2 | 28.31 | OK | Reasoning-only |
| stop-api-single | 28.26 | OK | Reasoning-only |
| stop-guided-json-value | 27.19 | OK | Invalid/truncated |
| stop-low-max-tokens | 26.61 | OK | Early stop |
| stop-system-pirate | 24.69 | OK | Reasoning-only |
| stop-seed-run1 | 24.13 | OK | Reasoning-only |
| stop-four-max | 22.39 | OK | Early stop |
| stop-immediate | 18.70 | OK | Immediate stop |
| stop-api-double-newline | 16.95 | OK | Immediate stop |
| stop-api-period | 15.95 | OK | Immediate stop |
| stop-special-chars | 15.74 | OK | Immediate stop |
| stop-api-newline | 12.93 | OK | Immediate stop |
| stop-cli-only | n/a | FAIL | CLI option error |
| stop-cli-multi | n/a | FAIL | CLI option error |
| stop-cli-api-merge | n/a | FAIL | CLI option error |
| stop-cli-api-dedup | n/a | FAIL | CLI option error |
| stop-json-object-key | n/a | FAIL | Jinja template crash |
| stop-json-object-no-match | n/a | FAIL | Jinja template crash |

### 6) Recommendations (Prioritized)

**Likely AFM bug**
1. **Stop token handling leaks/truncates on reasoning stream** across most API stop tests; user-visible `content` is empty while hidden reasoning is emitted.
2. **CLI/API parity:** implement or document `--stop` support; current CLI rejects expected flag.
3. **`response_format=json_object` + stop crash** (`Jinja.TemplateException`) should be fixed urgently.
4. **Guided JSON + stop interaction** is inconsistent (invalid JSON, schema bypass).

**Model quality issue**
1. No primary evidence this is model-quality-specific; failures are mostly harness/runtime behavior, not intrinsic text degeneration.

**Working well**
1. `stop-no-match` (returns correct final answer).
2. `stop-guided-json-brace` has coherent content for prompt semantics, though structured-output contract still questionable.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":1},{"i":8,"s":1},{"i":9,"s":1},{"i":10,"s":1},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":4},{"i":17,"s":1},{"i":18,"s":1},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":4},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":2}] -->
