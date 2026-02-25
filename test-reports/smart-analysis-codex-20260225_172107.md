## 1) Broken Models

### Error Type: `Jinja.TemplateException error 1` (HTTP 400)
| Models | Likely Cause | AFM bug vs Model |
|---|---|---|
| `@ stop-json-object-key`, `@ stop-json-object-no-match` | Failure in `response_format=json_object` request templating when stop sequences are present | **Likely AFM bug** (same model works in other modes in this run) |

No model load/crash failures were observed; failures are request-time server errors.

## 2) Anomalies & Red Flags

1. **Systemic empty `content` with non-empty `reasoning_content` under stop tests**  
   - Affected: most API stop variants (`stop-api-*`, `stop-streaming`, `stop-non-streaming`, `stop-system-*`, etc.).  
   - Pattern snippet: `"Thinking Process:\n\n1. **Analyze the Request:** ..."`  
   - Interpretation: stop matcher is likely triggering on reasoning tokens before final answer text is emitted.

2. **Stop-token over-trigger on common tokens/symbols**  
   - Examples: stop strings like `"."`, `"\n"`, `"**"`, `"The"`, `"</li>"`, `"3."` lead to immediate/truncated outputs.  
   - Snippets: `"Thinking Process:"`, `"Thinking Process:\n\n1"`.

3. **Guided JSON behavior is inconsistent / bypassed**
   - `@ stop-guided-json-value`: `is_valid_json=false`, empty answer.  
   - `@ stop-guided-json-brace`: prose output despite guided-json schema.  
   - `@ stop-guided-json-no-match`: long freeform output, schema ignored.

4. **Thinking-budget exhaustion case (explicit)**
   - `@ stop-low-max-tokens`: `completion_tokens=100` with `max_tokens=100`, empty `content`, non-empty reasoning.  
   - This indicates model functioned but spent budget in reasoning; not a harness capture failure.

5. **Suspicious tok/s outliers**
   - Very high with degraded behavior: `@ stop-guided-json-brace` (42.61 tok/s), `@ stop-guided-json-no-match` (42.58 tok/s), `@ stop-code-fence` (46.72 tok/s) while producing no final answer or schema violations.
   - Very low with minimal output: `@ stop-api-period` (0.13 tok/s), `@ stop-api-double-newline` (0.37 tok/s).

## 3) Variant Comparison

| Variant Group | Observation | Better/Worse |
|---|---|---|
| `stop-streaming` vs `stop-non-streaming` | Near-identical timings/tokens and same empty-content behavior | Neither better |
| `stop-seed-run1` vs `stop-seed-run2` | Identical output and metrics (deterministic with seed) | Stable |
| `stop-cli-only` / `stop-cli-multi` vs API stop variants | CLI variants returned correct visible `content`; many API stop variants produced only reasoning | **CLI better** |
| `stop-cli-api-merge` / `stop-cli-api-dedup` | Still empty-content + reasoning truncation | Degraded vs CLI-only |
| Guided JSON variants | Mixed failures: empty, invalid JSON, or prose bypass | Unstable/inconsistent |

## 4) Quality Assessment (all `status=OK` entries)

| Label | Coherence | Relevance | Flag (<3) |
|---|---:|---:|---|
| stop-api-single | 2 | 2 | Yes |
| stop-api-multi | 2 | 2 | Yes |
| stop-api-newline | 1 | 1 | Yes |
| stop-api-double-newline | 1 | 1 | Yes |
| stop-api-word | 2 | 2 | Yes |
| stop-api-period | 1 | 1 | Yes |
| stop-cli-only | 5 | 5 | No |
| stop-cli-multi | 5 | 5 | No |
| stop-cli-api-merge | 2 | 2 | Yes |
| stop-cli-api-dedup | 2 | 2 | Yes |
| stop-streaming | 2 | 2 | Yes |
| stop-non-streaming | 2 | 2 | Yes |
| stop-guided-json-value | 1 | 1 | Yes |
| stop-guided-json-comma | 2 | 2 | Yes |
| stop-guided-json-no-match | 3 | 3 | No |
| stop-guided-json-brace | 4 | 4 | No |
| stop-long-phrase | 2 | 1 | Yes |
| stop-multi-word | 2 | 2 | Yes |
| stop-code-fence | 2 | 2 | Yes |
| stop-no-match | 5 | 5 | No |
| stop-immediate | 1 | 1 | Yes |
| stop-special-chars | 1 | 1 | Yes |
| stop-html-tag | 2 | 2 | Yes |
| stop-unicode | 2 | 2 | Yes |
| stop-four-max | 2 | 2 | Yes |
| stop-system-pirate | 2 | 2 | Yes |
| stop-system-numbered | 2 | 2 | Yes |
| stop-high-temp | 2 | 2 | Yes |
| stop-seed-run1 | 2 | 2 | Yes |
| stop-seed-run2 | 2 | 2 | Yes |
| stop-top-p | 2 | 2 | Yes |
| stop-low-max-tokens | 3 | 2 | Yes |

## 5) Performance Summary (sorted by tokens/sec)

| Label | tok/s | Status | Note |
|---|---:|---|---|
| stop-code-fence | 46.72 | OK | Fast but empty final answer |
| stop-guided-json-brace | 42.61 | OK | Fast, schema not followed |
| stop-guided-json-no-match | 42.58 | OK | Fast, schema bypass |
| stop-cli-multi | 41.53 | OK | Good output |
| stop-cli-only | 40.98 | OK | Good output |
| stop-no-match | 39.96 | OK | Good output |
| stop-low-max-tokens | 38.58 | OK | Budget exhausted in reasoning |
| stop-system-pirate | 19.57 | OK | Empty final answer |
| stop-high-temp | 14.48 | OK | Empty final answer |
| stop-html-tag | 13.63 | OK | Empty final answer |
| stop-cli-api-merge | 10.46 | OK | Empty final answer |
| stop-seed-run1 | 9.19 | OK | Empty final answer |
| stop-seed-run2 | 9.19 | OK | Empty final answer |
| stop-four-max | 8.89 | OK | Empty final answer |
| stop-top-p | 8.83 | OK | Empty final answer |
| stop-non-streaming | 8.21 | OK | Empty final answer |
| stop-streaming | 8.13 | OK | Empty final answer |
| stop-cli-api-dedup | 8.05 | OK | Empty final answer |
| stop-multi-word | 6.87 | OK | Empty final answer |
| stop-api-multi | 6.40 | OK | Empty final answer |
| stop-system-numbered | 6.23 | OK | Empty final answer |
| stop-api-single | 6.22 | OK | Empty final answer |
| stop-immediate | 5.66 | OK | Immediate stop |
| stop-guided-json-comma | 3.84 | OK | Empty final answer |
| stop-unicode | 3.57 | OK | Empty final answer |
| stop-long-phrase | 2.64 | OK | Empty final answer |
| stop-guided-json-value | 2.07 | OK | Invalid JSON |
| stop-api-word | 1.39 | OK | Empty final answer |
| stop-api-newline | 1.01 | OK | Immediate stop |
| stop-special-chars | 0.50 | OK | Immediate stop |
| stop-api-double-newline | 0.37 | OK | Immediate stop |
| stop-api-period | 0.13 | OK | Immediate stop |
| stop-json-object-key | N/A | FAIL | Jinja template error |
| stop-json-object-no-match | N/A | FAIL | Jinja template error |

## 6) Recommendations (prioritized)

### Likely AFM bug
1. **Fix stop-sequence application scope**: apply stop matching to final assistant output channel only (or post-reasoning text), not internal reasoning stream.  
2. **Fix `response_format=json_object` + stop interaction**: Jinja template path throws 400; inspect chat-completions templating/render path for combined structured-output + stop settings.  
3. **Guided JSON enforcement order**: enforce schema-constrained decoding before/while checking stop sequences; reject non-JSON completions reliably.  
4. **Add regression tests**: assert non-empty `content` for stop tests unless stop is intentionally immediate; add tests for newline/period/common-token stops.

### Model quality issue
1. **Qwen3.5-35B-A3B-4bit tends to emit long reasoning first** under these settings; if AFM exposes reasoning token stream to stop matcher, outputs degrade.  
2. Consider defaulting to “no-reasoning-output” mode or a lower reasoning budget for compatibility tests.

### Working well
1. `stop-cli-only`  
2. `stop-cli-multi`  
3. `stop-no-match`  
These can be used as current baselines while debugging API/structured-output stop handling.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":4},{"i":17,"s":1},{"i":18,"s":1},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":2},{"i":29,"s":2},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":3}] -->
