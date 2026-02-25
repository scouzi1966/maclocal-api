## Broken Models
No `status=FAIL` entries (no load failures, crashes, or timeouts).

### Functional breakages (status OK but behaviorally broken)
| Error type | Lines | Likely cause | AFM bug vs model |
|---|---:|---|---|
| Stop triggered before useful answer (often due reasoning stream hitting stop) | 3,4,5,6,13,17,20,21,23,24,25,26,34 | Stop matching appears to run against internal reasoning channel or pre-visible text | **Likely AFM bug** |
| CLI `--stop` not applied | 7 | CLI stop merge/application path inconsistent vs API stop | **Likely AFM bug** |
| Guided JSON not enforced (invalid/non-JSON output) | 13,14,16,17 | Guided decoding + stop/format interaction broken | **Likely AFM bug** |

## Anomalies & Red Flags
- **Reasoning-budget / hidden-reasoning dominance**: many runs spend most tokens in `reasoning_content`, with tiny/empty `content`.  
  - Example line 3 snippet: `"Thinking Process: ... The capital of France is Paris..."` with empty `content`.
- **Premature truncation by stop sequences**:
  - Line 5: `"Here are 5 popular programming languages and a brief description of each:\n\n1.  **"`  
  - Line 20: `"Step 1: Boil fresh water... \n\nStep 2: Place a tea bag..."`  
  - Line 25: `"```html\n<ul>\n <li>Apple"`
- **Invalid JSON under JSON-oriented prompts**:
  - Line 13: `content="[\\n  \""` (`is_valid_json=false`)
  - Line 17: `content="{\n  \"name\": \"Carol\",\n  \""` (`is_valid_json=false`)
- **Thinking-budget exhaustion pattern (explicit)**:
  - Line 34: `content=""`, non-empty `reasoning_content`, `completion_tokens=100` with `max_tokens=100` (model functioning, but token budget consumed before user-visible answer).

## Variant Comparison
- **API stop works in basic cases**: line 1 (`stop ["3."]`) and line 2 (`stop ["4.","five"]`) truncate at expected boundaries.
- **CLI stop regression**: line 7 (`--stop "3."`) did **not** truncate (returned full 1..10), unlike API stop behavior.
- **CLI+API merge/dedup**: lines 9/10 behave like API stop wins/works (truncate at 2 items).
- **Streaming vs non-streaming parity**: lines 11 and 12 are effectively identical (good).
- **Seed determinism parity**: lines 31 and 32 are identical (good).
- **High-temp/top-p variants** still truncate early due stop interaction, not sampling instability (lines 30,33).

## Quality Assessment (Coherence/Relevance)
(Flagged when either score <3)

| Line | Label | Coherence | Relevance | Flag |
|---:|---|---:|---:|---|
| 1 | stop-api-single | 4 | 2 | ⚠ |
| 2 | stop-api-multi | 4 | 2 | ⚠ |
| 3 | stop-api-newline | 2 | 2 | ⚠ |
| 4 | stop-api-double-newline | 2 | 2 | ⚠ |
| 5 | stop-api-word | 2 | 2 | ⚠ |
| 6 | stop-api-period | 3 | 2 | ⚠ |
| 7 | stop-cli-only | 5 | 5 | |
| 8 | stop-cli-multi | 5 | 5 | |
| 9 | stop-cli-api-merge | 4 | 2 | ⚠ |
| 10 | stop-cli-api-dedup | 4 | 2 | ⚠ |
| 11 | stop-streaming | 4 | 2 | ⚠ |
| 12 | stop-non-streaming | 4 | 2 | ⚠ |
| 13 | stop-guided-json-value | 1 | 1 | ⚠ |
| 14 | stop-guided-json-comma | 2 | 2 | ⚠ |
| 15 | stop-guided-json-no-match | 4 | 3 | |
| 16 | stop-guided-json-brace | 4 | 4 | |
| 17 | stop-json-object-key | 1 | 1 | ⚠ |
| 18 | stop-json-object-no-match | 5 | 5 | |
| 19 | stop-long-phrase | 4 | 3 | |
| 20 | stop-multi-word | 3 | 2 | ⚠ |
| 21 | stop-code-fence | 2 | 2 | ⚠ |
| 22 | stop-no-match | 5 | 5 | |
| 23 | stop-immediate | 2 | 2 | ⚠ |
| 24 | stop-special-chars | 2 | 2 | ⚠ |
| 25 | stop-html-tag | 2 | 2 | ⚠ |
| 26 | stop-unicode | 2 | 2 | ⚠ |
| 27 | stop-four-max | 4 | 2 | ⚠ |
| 28 | stop-system-pirate | 5 | 5 | |
| 29 | stop-system-numbered | 4 | 3 | |
| 30 | stop-high-temp | 4 | 2 | ⚠ |
| 31 | stop-seed-run1 | 4 | 2 | ⚠ |
| 32 | stop-seed-run2 | 4 | 2 | ⚠ |
| 33 | stop-top-p | 4 | 2 | ⚠ |
| 34 | stop-low-max-tokens | 2 | 2 | ⚠ |

## Performance Summary (sorted by tokens/sec)
| Line | Label | tok/s | Flag |
|---:|---|---:|---|
| 19 | stop-long-phrase | 52.94 | suspiciously fast + truncated to 2/3 paras |
| 21 | stop-code-fence | 47.43 | fast but empty content |
| 4 | stop-api-double-newline | 45.57 | fast with empty content |
| 6 | stop-api-period | 45.02 | fast with under-complete output |
| 29 | stop-system-numbered | 43.67 | normal |
| 15 | stop-guided-json-no-match | 42.69 | normal |
| 16 | stop-guided-json-brace | 42.66 | normal |
| 28 | stop-system-pirate | 42.48 | normal |
| 18 | stop-json-object-no-match | 42.30 | normal |
| 8 | stop-cli-multi | 41.90 | normal |
| 7 | stop-cli-only | 41.64 | normal |
| 23 | stop-immediate | 41.28 | fast with empty content |
| 5 | stop-api-word | 41.09 | normal but heavily degenerate reasoning |
| 22 | stop-no-match | 40.67 | normal |
| 14 | stop-guided-json-comma | 40.18 | normal but broken output |
| 26 | stop-unicode | 39.85 | normal but empty content |
| 24 | stop-special-chars | 37.56 | normal but truncated |
| 34 | stop-low-max-tokens | 36.19 | max-token exhaustion |
| 3 | stop-api-newline | 36.15 | empty content |
| 17 | stop-json-object-key | 35.99 | truncated invalid JSON |
| 13 | stop-guided-json-value | 35.72 | truncated invalid JSON |
| 20 | stop-multi-word | 33.80 | truncated at Step 3 stop |
| 11 | stop-streaming | 31.84 | truncated at item 2 |
| 12 | stop-non-streaming | 31.61 | truncated at item 2 |
| 9 | stop-cli-api-merge | 31.60 | truncated at item 2 |
| 27 | stop-four-max | 31.35 | truncated at item 2 |
| 1 | stop-api-single | 31.28 | expected stop truncation |
| 2 | stop-api-multi | 31.20 | expected stop truncation |
| 30 | stop-high-temp | 31.12 | truncated at item 2 |
| 10 | stop-cli-api-dedup | 30.63 | truncated at item 2 |
| 25 | stop-html-tag | 30.23 | truncated HTML |
| 32 | stop-seed-run2 | 28.46 | slower |
| 31 | stop-seed-run1 | 28.45 | slower |
| 33 | stop-top-p | 27.83 | slowest |

## Recommendations (Prioritized)

### Likely AFM bug
1. **Fix stop matching scope**: stop sequences should apply to user-visible assistant content, not internal reasoning stream.  
   - Affects lines 3,4,5,6,13,17,20,21,23,24,25,26.
2. **Fix CLI/API stop merge logic**: ensure CLI `--stop` is always active and merged deterministically with API `stop`.  
   - Regression signal: line 7 vs lines 1/2/9/10.
3. **Harden guided JSON + stop interplay**: guided decoder must preserve schema-valid output even with stop sequences.  
   - Failures: lines 13,14,17; add post-generation JSON validation and retry/correction path.
4. **Add channel-aware token accounting**: prevent hidden reasoning from consuming output budget in constrained runs; optionally cap reasoning tokens.  
   - Especially line 34 max-token exhaustion.

### Model quality issue
1. **Excessive verbose reasoning / indecision loops** (line 5 especially) inflates token usage and increases stop-trigger surface.  
   - Mitigate with prompt/template changes (`reasoning effort low`, constrained CoT) or disable reasoning for stop-regression tests.

### Working well
- 8 (`stop-cli-multi`), 18 (`stop-json-object-no-match`), 22 (`stop-no-match`), 28 (`stop-system-pirate`), plus parity checks 11/12 and determinism 31/32.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":3},{"i":2,"s":3},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":2},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":3},{"i":10,"s":3},{"i":11,"s":3},{"i":12,"s":3},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":4},{"i":16,"s":4},{"i":17,"s":2},{"i":18,"s":5},{"i":19,"s":3},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":3},{"i":28,"s":5},{"i":29,"s":3},{"i":30,"s":3},{"i":31,"s":3},{"i":32,"s":3},{"i":33,"s":3},{"i":34,"s":2}] -->
