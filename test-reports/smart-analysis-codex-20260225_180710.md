## AFM Compatibility QA Report (from `/tmp/mlx-test-results.jsonl`)

### 1) Broken Models
Only one model family was tested (`mlx-community/Qwen3.5-35B-A3B-4bit`) with many variants.

| Error type | Variants | AFM bug vs model incompat |
|---|---|---|
| `400 mlx_error: Jinja.TemplateException error 1` | `@ stop-json-object-key` (line 17), `@ stop-json-object-no-match` (line 18) | **Likely AFM bug** (server-side templating/response_format path), not model architecture incompatibility |

### 2) Anomalies & Red Flags
Most anomalies are stop-sequence interaction failures (output cut mid-answer), not load/runtime crashes.

- **Reasoning-only / empty `content` due stop collisions** (model “works” but user-visible output is blank):  
  - line 3 `stop-api-newline` snippet: `"Thinking Process:  1.  **Analyze the Request:** ..."`  
  - line 4 `stop-api-double-newline` snippet: `"Thinking Process:  1.  **Analyze the Request:** ..."`  
  - line 21 `stop-code-fence` snippet: `"The user wants me to write a Python function that computes factorial..."`  
  - line 23 `stop-immediate` snippet: `"Thinking Process: ... Answer: The capital of Japan is Tokyo.cw"`  
  - line 26 `stop-unicode` snippet: `"Thinking Process:  1.  **Analyze the Request:** ..."`
- **Severely truncated visible output** (often 1-2 list items only): lines 1,2,9,10,11,12,20,24,25,27,30,31,32,33.
- **Guided JSON not respected / invalid JSON output**:  
  - line 13 `stop-guided-json-value`: `content` is `[\n  "` (`is_valid_json=false`)  
  - line 14 `stop-guided-json-comma`: prose/markdown instead of schema JSON  
  - line 16 `stop-guided-json-brace`: prose paragraph, not JSON object
- **Thinking-budget exhaustion pattern (explicit)**:  
  - line 34 `stop-low-max-tokens`: `content=""`, non-empty `reasoning_content`, `completion_tokens=100` with `max_tokens=100` => budget consumed in reasoning before final answer.

### 3) Variant Comparison
- **API stop vs CLI stop behavior is inconsistent**:
  - `stop-api-single` (line 1) truncates at item 2 (`stop=["3."]` effective).
  - `stop-cli-only` (line 7, `afm_args=--stop "3."`) still returns full 10-item list.
  - Suggests CLI stop application/parsing mismatch vs API stop path.
- **Streaming vs non-streaming**:
  - lines 11/12 produce identical truncated content; non-streaming is faster (31.26 vs 28.47 tok/s).
- **Seed reproducibility**:
  - lines 31/32 (`seed=42`) are effectively identical in output/timing (good determinism).
- **System-prompt variants**:
  - `stop-system-pirate` (line 28) performs well stylistically.
  - `stop-system-numbered` (line 29) obeys format but truncated before item 4 by stop rule.

### 4) Quality Assessment (Coherence/Relevance)
Flagged below 3 on either metric with `⚠`.

| Line | Variant | Coherence | Relevance | Note |
|---:|---|---:|---:|---|
|1|stop-api-single|4|2 ⚠|Truncated at 2 colors|
|2|stop-api-multi|4|2 ⚠|Truncated at 3 animals|
|3|stop-api-newline|2 ⚠|2 ⚠|No visible answer|
|4|stop-api-double-newline|2 ⚠|2 ⚠|No visible answer|
|5|stop-api-word|2 ⚠|2 ⚠|Cut at `Python`|
|6|stop-api-period|3|2 ⚠|1 sentence only|
|7|stop-cli-only|5|5|Good|
|8|stop-cli-multi|5|5|Good|
|9|stop-cli-api-merge|4|2 ⚠|Only 2 countries|
|10|stop-cli-api-dedup|4|2 ⚠|Only 2 cities|
|11|stop-streaming|4|2 ⚠|Only 2 objects|
|12|stop-non-streaming|4|2 ⚠|Only 2 objects|
|13|stop-guided-json-value|2 ⚠|2 ⚠|Invalid/truncated JSON|
|14|stop-guided-json-comma|2 ⚠|2 ⚠|Schema not followed|
|15|stop-guided-json-no-match|3|3|Usable but schema drift|
|16|stop-guided-json-brace|3|3|Prompt answered, schema drift|
|19|stop-long-phrase|4|2 ⚠|Missing 3rd paragraph|
|20|stop-multi-word|4|2 ⚠|Only Step 1-2|
|21|stop-code-fence|2 ⚠|2 ⚠|Reasoning only|
|22|stop-no-match|5|5|Correct (`4`)|
|23|stop-immediate|2 ⚠|2 ⚠|Reasoning only|
|24|stop-special-chars|2 ⚠|2 ⚠|Cut at bold marker|
|25|stop-html-tag|2 ⚠|2 ⚠|Partial HTML only|
|26|stop-unicode|2 ⚠|2 ⚠|Reasoning only|
|27|stop-four-max|4|2 ⚠|Only items 1-2|
|28|stop-system-pirate|5|5|Good|
|29|stop-system-numbered|4|3|Partial list but useful|
|30|stop-high-temp|4|2 ⚠|Only 2 words/items|
|31|stop-seed-run1|4|2 ⚠|Only 2 flowers|
|32|stop-seed-run2|4|2 ⚠|Only 2 flowers|
|33|stop-top-p|4|2 ⚠|Only 2 rivers|
|34|stop-low-max-tokens|3|2 ⚠|Thinking-budget exhaustion|

### 5) Performance Summary (sorted by tokens/sec)

| Rank | Line | Variant | tok/s | Flag |
|---:|---:|---|---:|---|
|1|19|stop-long-phrase|52.96|⚠ fast + truncated|
|2|21|stop-code-fence|47.16|⚠ fast + no visible output|
|3|4|stop-api-double-newline|44.64|⚠ no visible output|
|4|6|stop-api-period|44.17|⚠ truncated|
|5|29|stop-system-numbered|43.66| |
|6|16|stop-guided-json-brace|42.71|⚠ schema drift|
|7|15|stop-guided-json-no-match|42.43|⚠ schema drift|
|8|28|stop-system-pirate|42.38| |
|9|7|stop-cli-only|41.47| |
|10|8|stop-cli-multi|41.45| |
|11|23|stop-immediate|41.09|⚠ no visible output|
|12|5|stop-api-word|40.39|⚠ truncated|
|13|22|stop-no-match|40.04| |
|14|14|stop-guided-json-comma|39.67|⚠ invalid format|
|15|26|stop-unicode|39.24|⚠ no visible output|
|16|34|stop-low-max-tokens|38.29|⚠ budget exhaustion|
|17|24|stop-special-chars|37.56|⚠ truncated|
|18|13|stop-guided-json-value|35.49|⚠ invalid JSON|
|19|3|stop-api-newline|35.46|⚠ no visible output|
|20|20|stop-multi-word|33.84|⚠ truncated|
|21|12|stop-non-streaming|31.26|⚠ truncated|
|22|27|stop-four-max|30.75|⚠ truncated|
|23|1|stop-api-single|30.70|⚠ truncated|
|24|2|stop-api-multi|30.49|⚠ truncated|
|25|10|stop-cli-api-dedup|29.75|⚠ truncated|
|26|25|stop-html-tag|29.75|⚠ truncated|
|27|30|stop-high-temp|29.00|⚠ truncated|
|28|11|stop-streaming|28.47|⚠ truncated|
|29|32|stop-seed-run2|28.15|⚠ truncated|
|30|31|stop-seed-run1|28.13|⚠ truncated|
|31|9|stop-cli-api-merge|27.80|⚠ truncated|
|32|33|stop-top-p|27.45|⚠ truncated|
|33|17|stop-json-object-key|N/A|FAIL|
|34|18|stop-json-object-no-match|N/A|FAIL|

### 6) Recommendations (prioritized)

**Likely AFM bug**
1. Fix `response_format=json_object` + `stop` handling (`Jinja.TemplateException`, lines 17-18).  
   - Investigate chat template rendering path for `json_object` mode and stop-list injection.
2. Fix stop-sequence application scope so it does not erase all user-visible output while reasoning continues (lines 3,4,21,23,26,34).  
   - Check stop matcher placement relative to reasoning/content channel split.
3. Fix guided JSON enforcement under stop constraints (lines 13,14,16).  
   - Validate schema-constrained decoding path before emitting final content.
4. Verify CLI/API stop merge/parsing consistency (lines 7 vs 1/9/10).  
   - Confirm `--stop "a,b"` parsing and dedup/merge semantics.

**Model quality issue**
- No strong evidence of inherent model degradation; failures are mostly control-plane/decoding/stop-handling artifacts.

**Working well**
- `stop-cli-only` (line 7), `stop-cli-multi` (line 8), `stop-no-match` (line 22), `stop-system-pirate` (line 28), seed determinism pair (lines 31-32).

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":3},{"i":17,"s":1},{"i":18,"s":1},{"i":19,"s":2},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":3},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":3}] -->
