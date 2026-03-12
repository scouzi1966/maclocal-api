## 1) Broken Models
No load failures, crashes, or timeouts.  
All test entries had `status: OK` for `mlx-community/Qwen3.5-35B-A3B-4bit` variants.

## 2) Anomalies & Red Flags
| Variant | Issue | Snippet (first ~100 chars) | Likely Cause |
|---|---|---|---|
| `@ stop-api-single`, `@ stop-api-multi`, `@ stop-cli-api-merge`, `@ stop-cli-api-dedup`, `@ stop-streaming`, `@ stop-non-streaming`, `@ stop-four-max`, `@ stop-high-temp`, `@ stop-seed-run1`, `@ stop-seed-run2`, `@ stop-top-p` | Truncated numbered lists (usually stops at item 2) | `"1. ...\n2. ..."` | Stop sequence hits `"3."`/related patterns too early |
| `@ stop-api-newline`, `@ stop-api-double-newline`, `@ stop-code-fence`, `@ stop-immediate`, `@ stop-unicode` | Empty `content`, long `reasoning_content` only | `"Thinking Process: 1. **Analyze...**"` | Stop matching inside hidden reasoning / channel handling bug |
| `@ stop-guided-json-value`, `@ stop-guided-json-comma`, `@ stop-json-object-key` | Invalid/truncated JSON output under JSON-constrained tests | `"[  \""`, `"Here is a sample person profile..."`, `"{\n  \"name\": \"Carol\",\n  \""` | Guided JSON + stop handling conflict (AFM bug likely) |
| `@ stop-long-phrase` | Incomplete 3-paragraph essay (only 2 paragraphs) | `"Renewable energy represents..."` | Stop phrase `In conclusion` removed required last paragraph |
| `@ stop-multi-word` | Only steps 1–2 returned | `"Step 1: ...\n\nStep 2: ..."` | Stop sequence matched `Step 3` |
| `@ stop-special-chars`, `@ stop-html-tag` | Truncated markdown/HTML output | `"1. The Moon is"`, "```html\n<ul>\n <li>Apple" | Stop sequence matched formatting tokens (`**`, `</li>`) |
| `@ stop-low-max-tokens` | **Thinking-budget exhaustion pattern** (`content=""`, non-empty reasoning, `completion_tokens=100=max_tokens`) | `"Thinking Process: ... Options: Mount Everest, K2..."` | Token budget consumed in reasoning before final answer (not harness failure) |

## 3) Variant Comparison
- `@ stop-streaming` vs `@ stop-non-streaming`: identical truncation (`1. Mercury\n2. Venus`) and nearly identical speed. This is not a streaming transport issue.
- Best behavior: `@ stop-cli-only`, `@ stop-cli-multi`, `@ stop-no-match`, `@ stop-json-object-no-match`, `@ stop-system-pirate`.
- Worst degradations occur when stop strings are common substrings (`"3."`, `"Step 3"`, `"**"`, `"</li>"`, `"\n"`), causing premature termination before user-visible completion.
- JSON-constrained variants are inconsistent: one works (`@ stop-json-object-no-match`), others ignore/violate schema (`@ stop-guided-json-*`, `@ stop-json-object-key`).

## 4) Quality Assessment (Coherence / Relevance)
Flagged `<3` on either metric:

| Variant | Coherence | Relevance | Flag |
|---|---:|---:|---|
| `@ stop-api-single` | 3 | 2 | Yes |
| `@ stop-api-multi` | 3 | 2 | Yes |
| `@ stop-api-newline` | 2 | 1 | Yes |
| `@ stop-api-double-newline` | 2 | 1 | Yes |
| `@ stop-api-word` | 2 | 1 | Yes |
| `@ stop-api-period` | 3 | 2 | Yes |
| `@ stop-cli-api-merge` | 3 | 2 | Yes |
| `@ stop-cli-api-dedup` | 3 | 2 | Yes |
| `@ stop-streaming` | 3 | 2 | Yes |
| `@ stop-non-streaming` | 3 | 2 | Yes |
| `@ stop-guided-json-value` | 1 | 1 | Yes |
| `@ stop-guided-json-comma` | 2 | 1 | Yes |
| `@ stop-guided-json-no-match` | 4 | 2 | Yes |
| `@ stop-guided-json-brace` | 4 | 2 | Yes |
| `@ stop-json-object-key` | 1 | 1 | Yes |
| `@ stop-long-phrase` | 4 | 2 | Yes |
| `@ stop-multi-word` | 3 | 2 | Yes |
| `@ stop-code-fence` | 2 | 1 | Yes |
| `@ stop-immediate` | 2 | 1 | Yes |
| `@ stop-special-chars` | 2 | 1 | Yes |
| `@ stop-html-tag` | 2 | 1 | Yes |
| `@ stop-unicode` | 2 | 1 | Yes |
| `@ stop-four-max` | 3 | 2 | Yes |
| `@ stop-high-temp` | 3 | 2 | Yes |
| `@ stop-seed-run1` | 3 | 2 | Yes |
| `@ stop-seed-run2` | 3 | 2 | Yes |
| `@ stop-top-p` | 3 | 2 | Yes |
| `@ stop-low-max-tokens` | 2 | 2 | Yes |

Non-flagged (>=3 both): `@ stop-cli-only`, `@ stop-cli-multi`, `@ stop-json-object-no-match`, `@ stop-no-match`, `@ stop-system-pirate`, `@ stop-system-numbered`.

## 5) Performance Summary (sorted by tok/s)
| Variant | tok/s | Note |
|---|---:|---|
| `@ stop-long-phrase` | 53.67 | High; still truncated semantically |
| `@ stop-code-fence` | 47.67 | High with empty final content |
| `@ stop-api-double-newline` | 45.13 | High; no final content |
| `@ stop-api-period` | 45.08 | High; early stop |
| `@ stop-system-numbered` | 44.51 | Partial but usable |
| `@ stop-guided-json-brace` | 43.13 | Fast, ignored JSON constraint |
| `@ stop-system-pirate` | 43.12 | Good quality |
| `@ stop-guided-json-no-match` | 42.69 | Fast, schema not followed |
| `@ stop-json-object-no-match` | 42.60 | Good |
| `@ stop-cli-only` | 42.18 | Good |
| `@ stop-cli-multi` | 42.13 | Good |
| `@ stop-immediate` | 41.72 | Empty final content |
| `@ stop-api-word` | 41.18 | Long reasoning, truncated answer |
| `@ stop-no-match` | 40.82 | Good |
| `@ stop-unicode` | 40.09 | Empty final content |
| `@ stop-guided-json-comma` | 39.46 | Truncated at comma |
| `@ stop-low-max-tokens` | 38.82 | Thinking-budget exhaustion |
| `@ stop-special-chars` | 38.14 | Truncated at `**` |
| `@ stop-json-object-key` | 36.28 | Invalid/truncated JSON |
| `@ stop-api-newline` | 36.10 | Empty final content |
| `@ stop-guided-json-value` | 35.75 | Invalid JSON |
| `@ stop-multi-word` | 34.56 | Only 2 steps |
| `@ stop-streaming` | 31.89 | Truncated list |
| `@ stop-cli-api-merge` | 31.72 | Truncated list |
| `@ stop-api-single` | 31.65 | Truncated list |
| `@ stop-non-streaming` | 31.64 | Truncated list |
| `@ stop-four-max` | 31.54 | Truncated list |
| `@ stop-api-multi` | 31.10 | Truncated list |
| `@ stop-html-tag` | 30.88 | Truncated HTML |
| `@ stop-cli-api-dedup` | 30.28 | Truncated list |
| `@ stop-high-temp` | 30.27 | Truncated list |
| `@ stop-seed-run2` | 28.94 | Truncated list |
| `@ stop-seed-run1` | 28.91 | Truncated list |
| `@ stop-top-p` | 28.42 | Truncated list |

Outliers:
- Suspiciously fast with poor output: `@ stop-code-fence`, `@ stop-api-double-newline`, `@ stop-immediate`.
- Slowest group (`~28-30 tok/s`) mostly still truncated; speed is not the primary failure mode.

## 6) Recommendations (prioritized)
### Likely AFM bug
1. **Stop-sequence application scope is wrong**: stop appears to trigger on hidden reasoning or pre-answer planning, causing empty/truncated `content`.  
   - Check chat completion generation loop in `/v1/chat/completions` path: apply stop matching only on user-visible assistant text stream.
2. **Guided JSON + stop interaction is broken**: schema-constrained responses are frequently invalid/truncated.  
   - Inspect guided decoding integration and stop matcher ordering (constraint decoder should dominate; stop should not cut structural tokens mid-object).
3. **`response_format=json_object` not robust under stop settings** (`@ stop-json-object-key`).  
   - Ensure JSON object mode finalizes syntactically valid JSON before stop termination.
4. **Reasoning leakage risk**: `reasoning_content` is huge while `content` is empty in many tests.  
   - Verify separation policy and truncation behavior; avoid spending most budget in reasoning when final answer is required.

### Model quality issue
- Minimal evidence of intrinsic model quality failure. Most bad outputs are stop/config/runtime handling artifacts, not base model fluency issues.

### Working well
- `@ stop-cli-only`, `@ stop-cli-multi`, `@ stop-no-match`, `@ stop-json-object-no-match`, `@ stop-system-pirate` performed well and can be used as regression baselines.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":3},{"i":17,"s":2},{"i":18,"s":5},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":3},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":3}] -->
