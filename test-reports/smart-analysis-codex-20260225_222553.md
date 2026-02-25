## AFM Compatibility Triage Report (run: 2026-02-25)

### 1) Broken Models
No `status=FAIL` entries in this JSONL run (0 load failures / 0 crashes).

| Error type | Count | AFM bug vs model incompatibility |
|---|---:|---|
| Load failure | 0 | N/A |
| Server crash | 0 | N/A |
| Timeout | 0 | N/A |

### 2) Anomalies & Red Flags
Main issue pattern is **stop-sequence over-trigger**, causing truncated or empty `content` while long `reasoning_content` is present.

| Line | Variant | Red flag | Snippet (first ~100 chars) | Likely cause |
|---:|---|---|---|---|
| 1 | stop-api-single | Truncated after item 2 | `1. Red\n2. Blue` | stop token `3.` matched too early |
| 2 | stop-api-multi | Truncated after item 3 | `1. Lion\n2. Elephant\n3. Tiger` | stop token `4.` |
| 3 | stop-api-newline | Empty content, reasoning-only | `Thinking Process: ... The capital of France is Paris.` | stop token `\n` likely matching immediately |
| 4 | stop-api-double-newline | Empty content, reasoning-only | `Thinking Process: ...` | stop token `\n\n` |
| 5 | stop-api-word | Hard truncation mid-format | `Here are 5 popular programming languages... 1.  **` | stop token `Python`/format collision |
| 6 | stop-api-period | 1 sentence instead of 3 | `The Sun is a star at the center...` | stop token `.` |
| 8 | stop-cli-multi | Empty content | `Thinking Process: ... code block ... END` | stop on `````/`END` before visible answer |
| 13 | stop-guided-json-value | Invalid JSON output | `[\n  "` | stop token `Tokyo` interrupts guided JSON |
| 14 | stop-guided-json-comma | Prose + truncation, not JSON | `Here is a sample person profile for Alice...` | stop token `,` incompatible with JSON fields |
| 16 | stop-guided-json-brace | Ignores guided JSON schema | `Blue is a primary color... **Hex Code:**` | stop token `}` conflicts with object close |
| 17 | stop-json-object-key | Invalid partial JSON | `{\n  "name": "Carol",\n  "` | stop token `age` hit mid-object |
| 19 | stop-long-phrase | Missing 3rd paragraph | `Renewable energy represents...` | stop phrase `In conclusion` appears at start of p3 |
| 21 | stop-code-fence | No code emitted | `The user wants me to write a Python function...` | stop token ``` ``` blocks output start |
| 23 | stop-immediate | Empty content | `Thinking Process: ... Answer: The capital of Japan is Tokyo.` | stop list includes common start words (`The`,`I`,`A`) |
| 24 | stop-special-chars | Incomplete sentence | `1. The Moon is` | stop token `**` |
| 25 | stop-html-tag | Broken HTML | ````html\n<ul>\n <li>Apple` | stop token `</li>` |
| 26 | stop-unicode | Empty content | `Thinking Process: ... • ...` | stop token `•` |
| 34 | stop-low-max-tokens | **Thinking-budget exhaustion** | `Thinking Process: ... Options: Mount Everest, K2, ...` | `completion_tokens=100` equals `max_tokens=100`; model used budget in reasoning and never emitted answer |

Additional systemic red flag:
- Most entries leak very long `reasoning_content`; visible `content` is often truncated/empty. For OpenAI-compat behavior, this is usually undesirable unless explicitly requested.

### 3) Variant Comparison
- **`stop-streaming` (11) vs `stop-non-streaming` (12):** near-identical output and speed; no streaming-specific regression.
- **`stop-seed-run1` (31) vs `stop-seed-run2` (32):** identical output/timing; seed determinism appears stable.
- **API vs CLI stop variants:** both degrade similarly under aggressive stop strings; issue is likely shared stop-matching path, not transport.
- **JSON modes:**  
  - `stop-json-object-no-match` (18) works (valid JSON).  
  - `guided-json-*` and `stop-json-object-key` frequently fail/truncate under stop constraints.

### 4) Quality Assessment (Coherence / Relevance, 1-5)
Flagged `<3` on either metric are the priority.

| Line | Variant | Coh | Rel | Flag |
|---:|---|---:|---:|---|
| 1 | stop-api-single | 4 | 2 | Yes |
| 2 | stop-api-multi | 4 | 2 | Yes |
| 3 | stop-api-newline | 2 | 2 | Yes |
| 4 | stop-api-double-newline | 2 | 2 | Yes |
| 5 | stop-api-word | 2 | 2 | Yes |
| 6 | stop-api-period | 4 | 2 | Yes |
| 7 | stop-cli-only | 4 | 2 | Yes |
| 8 | stop-cli-multi | 2 | 1 | Yes |
| 9 | stop-cli-api-merge | 4 | 2 | Yes |
| 10 | stop-cli-api-dedup | 4 | 2 | Yes |
| 11 | stop-streaming | 4 | 2 | Yes |
| 12 | stop-non-streaming | 4 | 2 | Yes |
| 13 | stop-guided-json-value | 2 | 1 | Yes |
| 14 | stop-guided-json-comma | 2 | 1 | Yes |
| 15 | stop-guided-json-no-match | 4 | 3 | No |
| 16 | stop-guided-json-brace | 4 | 2 | Yes |
| 17 | stop-json-object-key | 2 | 1 | Yes |
| 18 | stop-json-object-no-match | 5 | 5 | No |
| 19 | stop-long-phrase | 4 | 2 | Yes |
| 20 | stop-multi-word | 4 | 2 | Yes |
| 21 | stop-code-fence | 2 | 1 | Yes |
| 22 | stop-no-match | 5 | 5 | No |
| 23 | stop-immediate | 2 | 1 | Yes |
| 24 | stop-special-chars | 1 | 1 | Yes |
| 25 | stop-html-tag | 1 | 1 | Yes |
| 26 | stop-unicode | 2 | 1 | Yes |
| 27 | stop-four-max | 4 | 2 | Yes |
| 28 | stop-system-pirate | 5 | 5 | No |
| 29 | stop-system-numbered | 5 | 3 | No |
| 30 | stop-high-temp | 4 | 2 | Yes |
| 31 | stop-seed-run1 | 4 | 2 | Yes |
| 32 | stop-seed-run2 | 4 | 2 | Yes |
| 33 | stop-top-p | 4 | 2 | Yes |
| 34 | stop-low-max-tokens | 2 | 2 | Yes |

### 5) Performance Summary (sorted by tokens/sec)
| Line | Variant | tok/s | Outlier note |
|---:|---|---:|---|
| 19 | stop-long-phrase | 52.82 | suspiciously high with truncated output |
| 21 | stop-code-fence | 46.84 | high; no visible answer |
| 4 | stop-api-double-newline | 45.26 | high; reasoning-only |
| 6 | stop-api-period | 44.90 | high; early stop |
| 29 | stop-system-numbered | 43.77 | high-normal |
| 15 | stop-guided-json-no-match | 42.79 | high-normal |
| 16 | stop-guided-json-brace | 42.50 | high-normal |
| 28 | stop-system-pirate | 42.42 | high-normal |
| 18 | stop-json-object-no-match | 41.87 | high-normal |
| 23 | stop-immediate | 41.09 | high; empty content |
| 5 | stop-api-word | 40.93 | high; severe reasoning bloat |
| 22 | stop-no-match | 40.14 | normal |
| 14 | stop-guided-json-comma | 39.85 | normal |
| 26 | stop-unicode | 39.63 | high; empty content |
| 34 | stop-low-max-tokens | 38.28 | normal; max-token exhaustion |
| 8 | stop-cli-multi | 38.12 | normal; empty content |
| 24 | stop-special-chars | 37.64 | normal; truncated |
| 3 | stop-api-newline | 36.14 | normal; empty content |
| 13 | stop-guided-json-value | 35.69 | normal; invalid JSON |
| 17 | stop-json-object-key | 35.63 | normal; invalid JSON |
| 20 | stop-multi-word | 34.02 | normal |
| 11 | stop-streaming | 31.71 | normal |
| 9 | stop-cli-api-merge | 31.61 | normal |
| 1 | stop-api-single | 31.50 | normal |
| 12 | stop-non-streaming | 31.46 | normal |
| 2 | stop-api-multi | 31.05 | normal |
| 27 | stop-four-max | 30.90 | normal |
| 10 | stop-cli-api-dedup | 30.43 | normal |
| 25 | stop-html-tag | 29.99 | normal |
| 7 | stop-cli-only | 29.73 | normal |
| 30 | stop-high-temp | 28.62 | slow-side normal |
| 31 | stop-seed-run1 | 28.16 | slow-side normal |
| 32 | stop-seed-run2 | 28.16 | slow-side normal |
| 33 | stop-top-p | 27.73 | slow-side normal |

### 6) Recommendations (prioritized)

#### Likely AFM bug
1. **Stop matching should apply only to surfaced assistant text, not hidden reasoning path.**  
   - Symptoms: many entries stop before any user-visible answer (`3,4,8,21,23,26`).  
   - Check code paths handling decode + stop in chat completions stream/non-stream merge (e.g., stop matcher in token assembly before role/channel separation).
2. **Guided JSON + stop sequence interaction is broken.**  
   - Failures: `13,14,16,17` (invalid/incomplete JSON despite guided/json_object modes).  
   - Check guided decoder + stop matcher precedence; schema-constrained closing tokens should not be cut by substring stop.
3. **`response_format=json_object` conformance regression under stop conditions.**  
   - `17` invalid JSON with `response_format=json_object`.  
   - Add post-decode JSON validity gate and retry/repair path before returning `status=OK`.
4. **Thinking-budget handling for reasoning models needs guardrails.**  
   - `34`: tokens exhausted in reasoning; no final answer.  
   - Add min-reserved completion budget or detect near-limit and force concise finalization.

#### Model quality issue
- None clearly model-inherent in this run; failures are dominated by stop/guided-decoding orchestration.

#### Working well
- `18 stop-json-object-no-match` (valid strict JSON output)
- `22 stop-no-match` (correct concise answer)
- `28 stop-system-pirate` (good system-prompt adherence)
- `11/12` streaming vs non-streaming parity
- `31/32` deterministic seed parity

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":2},{"i":2,"s":2},{"i":3,"s":2},{"i":4,"s":2},{"i":5,"s":2},{"i":6,"s":3},{"i":7,"s":2},{"i":8,"s":2},{"i":9,"s":2},{"i":10,"s":2},{"i":11,"s":2},{"i":12,"s":2},{"i":13,"s":2},{"i":14,"s":2},{"i":15,"s":3},{"i":16,"s":2},{"i":17,"s":2},{"i":18,"s":5},{"i":19,"s":3},{"i":20,"s":2},{"i":21,"s":2},{"i":22,"s":5},{"i":23,"s":2},{"i":24,"s":2},{"i":25,"s":2},{"i":26,"s":2},{"i":27,"s":2},{"i":28,"s":4},{"i":29,"s":3},{"i":30,"s":2},{"i":31,"s":2},{"i":32,"s":2},{"i":33,"s":2},{"i":34,"s":2}] -->
