## AFM Compatibility QA Report (from `/tmp/mlx-test-results.jsonl`)

### 1) Broken Models (load/crash)
No `status=FAIL` entries.  
No hard load failures, architecture incompatibilities, or server crashes in this run.

### 2) Anomalies & Red Flags

| Type | Affected lines (idx) | Evidence (first ~100 chars) | Likely cause | AFM bug vs model |
|---|---:|---|---|---|
| Stop over-trigger: numbered-list truncation at item 2/3 | 1,2,9,10,11,12,27,30,31,32,33 | `"1. New York City\n2. London"` / `"1. Mercury\n2. Venus"` | Stop matcher firing on tokens like `"3."` | **Likely AFM stop handling bug** |
| Stop over-trigger: empty `content`, only reasoning emitted | 3,4,21,23,26 | `content=""`, preview starts `"Thinking Process: ..."` | Stop applied before final channel output (possibly on reasoning tokens/newline/code fence/bullet) | **Likely AFM channel/stop integration bug** |
| Guided JSON not enforced / malformed outputs | 13,14,15,16,17 | `"[ \n  \""`, `"Here is a sample person profile..."`, `"{\n  \"name\": \"Carol\",\n  \""` | Guided decoder / response_format path not constraining output | **Likely AFM guided-json/json-object bug** |
| Truncated prose responses | 5,6,19,20,24,25,29 | `"Here are 5 popular programming languages..."` / `"The Sun is a star..."` | Stop sequence intersects normal text (period, words, markdown markers, tags) | **Likely AFM stop matching bug** |
| Thinking-budget exhaustion pattern | 34 | `completion_tokens=100, max_tokens=100, content=""`, reasoning partial | Max token budget consumed in reasoning before answer | **Config/harness issue**, not model failure |

Explicit thinking-budget case: **line 34** matches your rule exactly.

### 3) Variant Comparison

| Variant pair/group | Result |
|---|---|
| `stop-streaming` (11) vs `stop-non-streaming` (12) | Nearly identical truncation (`1.,2.` only). Streaming parity exists, but both wrong. |
| `stop-seed-run1` (31) vs `stop-seed-run2` (32) | Deterministic and identical outputs/timing (good reproducibility). |
| API stop vs CLI stop (`stop-api-*` vs `stop-cli-only`) | API stop variants truncate aggressively; `stop-cli-only` (7) returned full 10 items despite `--stop "3."` (inconsistent semantics). |
| `stop-cli-api-merge` (9) / `stop-cli-api-dedup` (10) | Both still truncate at 2 items; merge/dedup path likely mishandling effective stop set. |
| Guided JSON family (13,14,15,16,17,18) | Only `stop-json-object-no-match` (18) is clearly correct JSON. Others are malformed or schema-ignored. |
| System prompt variants (28,29) | Pirate style (28) works well; numbered-list system prompt (29) obeyed style but truncated at 3 items due stop trigger. |

### 4) Quality Assessment (Coherence/Relevance 1–5)

Flagged `<3` shown as `⚠`.

| idx | label | Coherence | Relevance |
|---:|---|---:|---:|
| 1 | stop-api-single | 2 ⚠ | 2 ⚠ |
| 2 | stop-api-multi | 2 ⚠ | 2 ⚠ |
| 3 | stop-api-newline | 2 ⚠ | 2 ⚠ |
| 4 | stop-api-double-newline | 2 ⚠ | 2 ⚠ |
| 5 | stop-api-word | 2 ⚠ | 2 ⚠ |
| 6 | stop-api-period | 3 | 2 ⚠ |
| 7 | stop-cli-only | 5 | 5 |
| 8 | stop-cli-multi | 5 | 5 |
| 9 | stop-cli-api-merge | 2 ⚠ | 2 ⚠ |
| 10 | stop-cli-api-dedup | 2 ⚠ | 2 ⚠ |
| 11 | stop-streaming | 2 ⚠ | 2 ⚠ |
| 12 | stop-non-streaming | 2 ⚠ | 2 ⚠ |
| 13 | stop-guided-json-value | 2 ⚠ | 2 ⚠ |
| 14 | stop-guided-json-comma | 2 ⚠ | 2 ⚠ |
| 15 | stop-guided-json-no-match | 3 | 3 |
| 16 | stop-guided-json-brace | 4 | 4 |
| 17 | stop-json-object-key | 2 ⚠ | 2 ⚠ |
| 18 | stop-json-object-no-match | 5 | 5 |
| 19 | stop-long-phrase | 4 | 3 |
| 20 | stop-multi-word | 2 ⚠ | 2 ⚠ |
| 21 | stop-code-fence | 2 ⚠ | 2 ⚠ |
| 22 | stop-no-match | 5 | 5 |
| 23 | stop-immediate | 2 ⚠ | 2 ⚠ |
| 24 | stop-special-chars | 2 ⚠ | 2 ⚠ |
| 25 | stop-html-tag | 2 ⚠ | 2 ⚠ |
| 26 | stop-unicode | 2 ⚠ | 2 ⚠ |
| 27 | stop-four-max | 2 ⚠ | 2 ⚠ |
| 28 | stop-system-pirate | 5 | 5 |
| 29 | stop-system-numbered | 4 | 3 |
| 30 | stop-high-temp | 2 ⚠ | 2 ⚠ |
| 31 | stop-seed-run1 | 2 ⚠ | 2 ⚠ |
| 32 | stop-seed-run2 | 2 ⚠ | 2 ⚠ |
| 33 | stop-top-p | 2 ⚠ | 2 ⚠ |
| 34 | stop-low-max-tokens | 3 | 2 ⚠ |

### 5) Performance Summary (sorted by tokens/sec)

| idx | label | tok/s | Note |
|---:|---|---:|---|
| 19 | stop-long-phrase | 52.95 | Fast but output truncated before required 3rd paragraph |
| 21 | stop-code-fence | 46.87 | Fast; empty `content` |
| 4 | stop-api-double-newline | 44.98 | Empty `content` |
| 6 | stop-api-period | 44.47 | Truncated to 1 sentence |
| 29 | stop-system-numbered | 43.38 | Truncated list |
| 28 | stop-system-pirate | 42.50 | Good quality |
| 16 | stop-guided-json-brace | 42.03 | Good prose, but ignores guided JSON |
| 18 | stop-json-object-no-match | 41.82 | Good |
| 8 | stop-cli-multi | 41.81 | Good |
| 15 | stop-guided-json-no-match | 41.66 | Schema mismatch |
| 23 | stop-immediate | 41.08 | Empty `content` |
| 7 | stop-cli-only | 41.04 | Good |
| 5 | stop-api-word | 40.77 | Truncated |
| 22 | stop-no-match | 40.38 | Good |
| 26 | stop-unicode | 39.54 | Empty `content` |
| 14 | stop-guided-json-comma | 39.41 | Truncated profile |
| 34 | stop-low-max-tokens | 38.84 | Thinking-budget exhaustion |
| 24 | stop-special-chars | 37.65 | Truncated on `**` |
| 3 | stop-api-newline | 35.69 | Empty `content` |
| 17 | stop-json-object-key | 35.25 | Broken JSON |
| 13 | stop-guided-json-value | 35.17 | Broken JSON |
| 20 | stop-multi-word | 34.12 | Truncated at Step 2 |
| 11 | stop-streaming | 31.42 | Truncated to 2 lines |
| 9 | stop-cli-api-merge | 31.37 | Truncated to 2 lines |
| 12 | stop-non-streaming | 31.17 | Truncated to 2 lines |
| 1 | stop-api-single | 31.12 | Truncated to 2 lines |
| 27 | stop-four-max | 30.97 | Truncated to 2 lines |
| 2 | stop-api-multi | 30.64 | Truncated to 3 lines |
| 25 | stop-html-tag | 30.55 | Truncated HTML |
| 10 | stop-cli-api-dedup | 30.27 | Truncated to 2 lines |
| 31 | stop-seed-run1 | 28.54 | Truncated |
| 30 | stop-high-temp | 28.53 | Truncated |
| 32 | stop-seed-run2 | 28.52 | Truncated |
| 33 | stop-top-p | 27.94 | Truncated |

Outliers:
- Suspiciously high with bad output: **19, 21, 4**.
- Slowest: **33, 31/32/30** (not extreme but also low-quality due truncation).

### 6) Recommendations (prioritized)

#### Likely AFM bug
1. **Stop sequence matching scope is wrong**: stop triggers appear to fire on content patterns (`"3."`, `"Step 3"`, `"**"`, `"</li>"`, `"Tokyo"`, `"age"`) and possibly reasoning channel.  
   - Check `/v1/chat/completions` decode loop stop matcher: apply stops only to emitted assistant `content` stream, not hidden reasoning buffers.
2. **Reasoning/content channel separation**: many cases emit only `reasoning_content` and empty `content`.  
   - Audit message assembly path where reasoning is captured; ensure final answer channel is not preempted by stop.
3. **Guided JSON / json_object enforcement**: malformed outputs with `is_valid_json=false` under guided-json and response_format tests.  
   - Inspect guided decoding constraint path (JSON schema/token masking), especially stop+guided interaction and early-stop finalization.
4. **CLI/API stop merge semantics** are inconsistent (`stop-cli-only` full output; merge/dedup truncates).  
   - Validate precedence and dedup logic for `--stop` + API `stop`.

#### Model quality issue
- None primary. Pattern is systematic across variants and looks infrastructure-related, not intrinsic model quality.

#### Working well (no immediate action)
- `stop-cli-only` (7), `stop-cli-multi` (8), `stop-json-object-no-match` (18), `stop-no-match` (22), `stop-system-pirate` (28).  
- `stop-seed-run1/2` deterministic behavior is good (31,32), though both are still functionally truncated due stop behavior.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":4},{"i":17,"s":2},{"i":18,"s":5},{"i":19,"s":3},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":3},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":3}] -->
