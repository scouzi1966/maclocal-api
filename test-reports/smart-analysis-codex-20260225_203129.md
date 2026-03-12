## Compatibility QA Report (AFM `v0.9.5-0cfba17`, run: 2026-02-25)

### 1) Broken Models
All failures are the same model family (`mlx-community/Qwen3.5-35B-A3B-4bit`) under specific variants.

| Error type | Variants | Count | Likely cause | AFM bug vs model |
|---|---|---:|---|---|
| CLI option parse failure (`Unknown option '--stop'`) | `stop-cli-only`, `stop-cli-multi`, `stop-cli-api-merge`, `stop-cli-api-dedup` | 4 | Binary/test mismatch or CLI wiring regression (`main.swift` `MlxCommand`/dispatch) | **Likely AFM/tooling bug** (not model incompat) |
| HTTP 400 `Jinja.TemplateException` with `response_format=json_object` + `stop` | `stop-json-object-key`, `stop-json-object-no-match` | 2 | Template/message assembly bug in JSON-format path | **AFM bug** (`MLXModelService.swift` around JSON instruction injection) |

### 2) Anomalies & Red Flags
1. **Major degradation pattern: empty `content`, only `reasoning_content` in most OK runs** (26/28 OK lines).  
   Representative snippets:
   - `"Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Task: List 10 colors..."`  
   - `"Thinking Process:"`  
   This suggests stop strings are frequently matching planning text before final answer.
2. **Stop sequences likely applied at wrong stage/channel** (stopping before user-visible answer).  
   Seen in many stop tests (`stop-api-*`, `stop-system-*`, `stop-high-temp`, etc.).
3. **Guided JSON path unstable with stop**:
   - `stop-guided-json-value`: `is_valid_json=false`, truncated planning.
   - `stop-guided-json-comma`: truncated planning.
   - `stop-guided-json-no-match`: huge reasoning (1150 tokens), output ignores schema.
   - `stop-guided-json-brace`: coherent text but not schema-constrained JSON.
4. **Suspicious tok/s outliers tied to poor outputs**:
   - Very high: `stop-code-fence` 47.43 tok/s (empty content).
   - Very low: `stop-api-newline` 13.31 tok/s (immediate truncation).
5. **Thinking-budget exhaustion pattern**: **Not observed** (no empty-content case near `max_tokens` ceiling).

### 3) Variant Comparison
| Variant pair/group | Result |
|---|---|
| `stop-streaming` vs `stop-non-streaming` | Behavior identical (same reasoning-only truncation), similar speed (33.97 vs 34.93 tok/s). Good stream parity, same core bug. |
| `stop-seed-run1` vs `stop-seed-run2` (seed=42) | Deterministic and identical outputs/timing. |
| `stop-guided-json-*` | Mixed failures: early truncation, invalid JSON, or unconstrained plain text. No reliable schema adherence under stop constraints. |
| `stop-no-match` vs most `stop-*` | `stop-no-match` is the only clearly correct short answer (`"4"`). Many other stop strings trigger too early in reasoning/planning. |

### 4) Quality Assessment (Coherence/Relevance 1-5)
Flagged below 3 on either metric.

| Variant | Coherence | Relevance | Flag |
|---|---:|---:|---|
| stop-api-single | 2 | 1 | ⚠️ |
| stop-api-multi | 2 | 1 | ⚠️ |
| stop-api-newline | 1 | 1 | ⚠️ |
| stop-api-double-newline | 1 | 1 | ⚠️ |
| stop-api-word | 2 | 1 | ⚠️ |
| stop-api-period | 1 | 1 | ⚠️ |
| stop-streaming | 2 | 1 | ⚠️ |
| stop-non-streaming | 2 | 1 | ⚠️ |
| stop-guided-json-value | 1 | 1 | ⚠️ |
| stop-guided-json-comma | 2 | 1 | ⚠️ |
| stop-guided-json-no-match | 3 | 3 |  |
| stop-guided-json-brace | 4 | 4 |  |
| stop-long-phrase | 2 | 1 | ⚠️ |
| stop-multi-word | 2 | 1 | ⚠️ |
| stop-code-fence | 2 | 1 | ⚠️ |
| stop-no-match | 5 | 5 |  |
| stop-immediate | 1 | 1 | ⚠️ |
| stop-special-chars | 1 | 1 | ⚠️ |
| stop-html-tag | 2 | 1 | ⚠️ |
| stop-unicode | 2 | 1 | ⚠️ |
| stop-four-max | 2 | 1 | ⚠️ |
| stop-system-pirate | 2 | 1 | ⚠️ |
| stop-system-numbered | 2 | 1 | ⚠️ |
| stop-high-temp | 2 | 1 | ⚠️ |
| stop-seed-run1 | 2 | 1 | ⚠️ |
| stop-seed-run2 | 2 | 1 | ⚠️ |
| stop-top-p | 2 | 1 | ⚠️ |
| stop-low-max-tokens | 2 | 1 | ⚠️ |

### 5) Performance Summary (sorted by tok/s)
| Variant | tok/s | Note |
|---|---:|---|
| stop-code-fence | 47.43 | suspiciously fast + empty content |
| stop-guided-json-no-match | 43.18 | high throughput, long reasoning dump |
| stop-guided-json-brace | 43.13 | high |
| stop-no-match | 40.59 | healthy |
| stop-api-word | 39.66 | high but poor answering |
| stop-system-numbered | 36.93 |  |
| stop-system-pirate | 36.73 |  |
| stop-four-max | 35.30 |  |
| stop-non-streaming | 34.93 |  |
| stop-streaming | 33.97 |  |
| stop-multi-word | 33.16 |  |
| stop-top-p | 32.28 |  |
| stop-high-temp | 31.54 |  |
| stop-html-tag | 31.31 |  |
| stop-guided-json-comma | 31.28 |  |
| stop-guided-json-value | 30.94 |  |
| stop-seed-run2 | 30.38 | deterministic pair |
| stop-seed-run1 | 30.36 | deterministic pair |
| stop-api-multi | 30.12 |  |
| stop-api-single | 30.02 |  |
| stop-unicode | 29.59 |  |
| stop-long-phrase | 29.42 |  |
| stop-low-max-tokens | 27.48 |  |
| stop-immediate | 19.21 | short-circuit |
| stop-api-double-newline | 17.58 | short-circuit |
| stop-api-period | 17.06 | short-circuit |
| stop-special-chars | 16.03 | short-circuit |
| stop-api-newline | 13.31 | slowest + immediate truncation |

### 6) Recommendations (prioritized)

**Likely AFM bug**
1. **Fix stop handling vs reasoning separation**: stop matching is terminating before answer content.  
   Check `Sources/MacLocalAPI/Models/MLXModelService.swift` stop logic (`~351-376`, `~572-647`) and content/reasoning extraction path in `MLXChatCompletionsController`.
2. **Fix `json_object + stop` 400 crash**: reproduce with `stop-json-object-key/no-match`; patch template assembly/instruction injection in `MLXModelService.swift` (`~1187+`) to avoid `Jinja.TemplateException`.
3. **Resolve CLI `--stop` regression**: code defines `--stop` in `main.swift` (`MlxCommand`), but runtime reports unknown option. Verify installed binary (`/opt/homebrew/bin/afm`) matches source build and subcommand path used by `mlx-model-test.sh`.
4. **Strengthen guided-json enforcement under stop**: stop+schema combinations currently return invalid/unconstrained output. Add regression tests in `Scripts/test-stop-sequences.txt` + `Scripts/tests/test-structured-outputs.sh`.

**Model quality issue**
1. No clear model-inherent failure evidence here; issues look infrastructure-side.

**Working well**
1. `stop-no-match` (correct direct answer).
2. Streaming/non-streaming parity.
3. Seed reproducibility (`run1`/`run2`).

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":1},{"i":8,"s":1},{"i":9,"s":1},{"i":10,"s":1},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":4},{"i":17,"s":1},{"i":18,"s":1},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":2}] -->
