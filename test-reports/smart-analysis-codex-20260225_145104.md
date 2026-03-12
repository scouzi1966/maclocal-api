## Broken Models

### Error Type: `Connection error` (likely AFM/harness bug, not model incompatibility)
| Line idx | Variant | Prompt |
|---:|---|---|
| 46 | `agent-cached-turn1` | Read `Sources/MacLocalAPI/main.swift` |
| 48 | `agent-cached-turn2` | Add `--timeout` CLI flag |
| 50 | `agent-cached-turn3` | Write timeout unit test |

No architecture/file-format incompatibility failures were observed.

## Anomalies & Red Flags

| Pattern | Lines | Variant(s) | Snippet (first ~100 chars) |
|---|---|---|---|
| Thinking-budget exhaustion (empty `content`, maxed tokens, long `reasoning_content`) | 16,18,27,29,51,52,62,64,66,88,90,92,95,109,120,123,125,126 | `seed-42-*`, `scientist`, `eli5`, `short-output`, `prefill-*`, `streaming-seeded`, `non-streaming-seeded`, `max-completion-tokens`, `developer-role`, `tool-call-multi`, `special-chars`, `code-python`, `code-swift` | `"Thinking Process:  1.  **Analyze the Request:** ..."` |
| `<think>` leakage in user-visible output | 69,70,85,86 | `raw-mode`, `think-raw` | `"<think>Thinking Process:  1.  **Analyze the Request:** ..."` |
| Tool/agent non-execution (returns plans/placeholders instead of action/result) | 40,42,44,112 | `agent-no-cache-*`, `tool-call-complex` | `"I'll read the file... <read_file ...>"`, `"I'll start by exploring..."` |
| Structured-output contract ignored | 35,36,37,38,77,79,80 | `guided-json-*`, `response-format-json/schema` | `"The capital of Japan is Tokyo."`, `"**Character Profile: Elias Thorne**..."` |
| Repetition/looping in reasoning (content missing or degraded) | 88,90,92,95,120,123,125,126 | multiple | repeated “Wait, I need to…” loops |

## Variant Comparison (A/B Highlights)

- `seed-42-run1` vs `seed-42-run2`: both deterministic; both fail identically on limerick (reasoning-only, token-budget exhaustion).
- `streaming-seeded` vs `non-streaming-seeded`: both fail identically on 4-line poem (reasoning-only, maxed at 4096).
- `prefill-default` vs `prefill-large-4096` vs `prefill-small-256`: same failure mode on architecture bottleneck prompt (reasoning-only, no final answer).
- `agent-no-cache-*` vs `agent-cached-*`: cached variants hard-fail (`Connection error`), no-cache variants return placeholder/planning text.
- `no-penalty` vs `with-penalty` vs `repetition-penalty` (bread essay): `no-penalty`/`with-penalty` are better; `repetition-penalty` degrades into verbosity/run-on style.
- `raw-mode`/`think-raw` degrade safety/format by leaking internal `<think>` blocks.

## Quality Assessment

### Variants below 3 on coherence or relevance (flagged)
| Variant | Coherence (1-5) | Relevance (1-5) | Why flagged |
|---|---:|---:|---|
| `response-format-schema` | 2 | 2 | Ignores required schema; one response is unrelated long profile text format-wise |
| `guided-json-simple` | 2 | 2 | Guided JSON constraints ignored |
| `guided-json-nested` | 2 | 2 | Guided nested schema ignored |
| `short-output` | 2 | 2 | Token cap consumed by reasoning; no usable answer |
| `code-swift` | 2 | 2 | Repeated reasoning loops; empty final content |
| `special-chars` | 2 | 2 | Fails exact-repeat task due reasoning-only exhaustion |

All other variants were generally 4–5 on coherence/relevance for their successful rows, except specific anomalous rows listed above.

## Performance Summary (all tested variants, sorted by avg tokens/sec)

| Variant | Runs | Avg tok/s | Min | Max | Fails |
|---|---:|---:|---:|---:|---:|
| scientist | 2 | 42.20 | 42.20 | 42.21 | 0 |
| eli5 | 2 | 42.16 | 42.11 | 42.21 | 0 |
| pirate | 2 | 42.09 | 41.96 | 42.21 | 0 |
| greedy | 2 | 41.83 | 41.18 | 42.47 | 0 |
| long-prompt | 2 | 41.80 | 41.21 | 42.39 | 0 |
| code-swift | 2 | 41.75 | 41.65 | 41.84 | 0 |
| response-format-schema | 2 | 41.73 | 41.07 | 42.38 | 0 |
| raw-mode | 2 | 41.58 | 40.77 | 42.39 | 0 |
| response-format-json | 2 | 41.58 | 41.07 | 42.09 | 0 |
| default | 2 | 41.57 | 40.90 | 42.24 | 0 |
| guided-json-simple | 2 | 41.56 | 40.88 | 42.25 | 0 |
| guided-json-nested | 2 | 41.56 | 41.00 | 42.13 | 0 |
| long-output | 2 | 41.50 | 40.78 | 42.21 | 0 |
| very-verbose | 2 | 41.48 | 40.94 | 42.03 | 0 |
| think-raw | 2 | 41.48 | 41.02 | 41.94 | 0 |
| small-kv | 2 | 41.44 | 40.84 | 42.03 | 0 |
| tool-call-none | 2 | 41.39 | 40.81 | 41.97 | 0 |
| stop-single | 2 | 41.39 | 40.99 | 41.78 | 0 |
| no-streaming | 2 | 41.38 | 40.44 | 42.31 | 0 |
| numbered-list | 2 | 41.33 | 40.80 | 41.87 | 0 |
| multilingual | 2 | 41.33 | 40.71 | 41.94 | 0 |
| math | 2 | 41.32 | 40.71 | 41.93 | 0 |
| response-format-text | 2 | 41.24 | 40.88 | 41.60 | 0 |
| think-normal | 2 | 41.22 | 40.62 | 41.81 | 0 |
| seed-42-run1 | 2 | 41.20 | 40.44 | 41.97 | 0 |
| minimal-prompt | 2 | 41.20 | 41.12 | 41.29 | 0 |
| streaming-seeded | 2 | 41.20 | 40.35 | 42.05 | 0 |
| no-penalty | 2 | 41.20 | 40.55 | 41.84 | 0 |
| max-completion-tokens | 3 | 41.15 | 39.74 | 42.09 | 0 |
| stop-multi | 2 | 41.14 | 41.00 | 41.27 | 0 |
| long-form | 2 | 41.14 | 40.31 | 41.96 | 0 |
| seed-42-run2 | 2 | 41.11 | 40.21 | 42.00 | 0 |
| json-output | 2 | 41.11 | 40.04 | 42.17 | 0 |
| prefill-small-256 | 2 | 41.08 | 40.11 | 42.05 | 0 |
| stop-newline | 2 | 41.08 | 40.92 | 41.23 | 0 |
| prefill-default | 2 | 41.07 | 40.19 | 41.95 | 0 |
| tool-call-multi | 3 | 41.06 | 39.89 | 42.37 | 0 |
| non-streaming-seeded | 2 | 41.03 | 40.00 | 42.07 | 0 |
| special-chars | 2 | 41.03 | 41.02 | 41.03 | 0 |
| strict-format | 2 | 41.02 | 40.72 | 41.32 | 0 |
| tool-call-complex | 3 | 41.01 | 40.22 | 41.84 | 0 |
| developer-role | 3 | 40.99 | 39.65 | 41.71 | 0 |
| top-k | 2 | 40.95 | 40.63 | 41.27 | 0 |
| tool-call-xml | 3 | 40.94 | 39.93 | 42.20 | 0 |
| code-python | 2 | 40.88 | 40.28 | 41.48 | 0 |
| prefill-large-4096 | 2 | 40.87 | 39.61 | 42.13 | 0 |
| tool-call-auto | 3 | 40.86 | 39.65 | 41.92 | 0 |
| logprobs | 2 | 40.86 | 40.81 | 40.90 | 0 |
| combined-samplers | 2 | 40.75 | 39.79 | 41.71 | 0 |
| high-temp | 2 | 40.74 | 40.38 | 41.10 | 0 |
| verbose | 2 | 40.66 | 40.01 | 41.31 | 0 |
| kv-quantized | 2 | 40.63 | 39.23 | 42.03 | 0 |
| top-p | 2 | 40.32 | 40.27 | 40.37 | 0 |
| min-p | 2 | 39.88 | 38.82 | 40.93 | 0 |
| with-penalty | 2 | 38.34 | 37.50 | 39.18 | 0 |
| repetition-penalty | 2 | 38.04 | 36.70 | 39.38 | 0 |
| short-output | 2 | 36.36 | 35.44 | 37.28 | 0 |
| agent-no-cache-turn3 | 2 | 32.45 | 27.96 | 36.95 | 0 |
| agent-no-cache-turn2 | 2 | 30.27 | 27.88 | 32.67 | 0 |
| agent-cached-turn1 | 2 | 27.99 | 27.99 | 27.99 | 1 |
| agent-cached-turn2 | 2 | 27.90 | 27.90 | 27.90 | 1 |
| agent-cached-turn3 | 2 | 27.88 | 27.88 | 27.88 | 1 |
| agent-no-cache-turn1 | 2 | 26.95 | 25.19 | 28.72 | 0 |

Outliers:
- Slow: all `agent-*` variants (26.95–32.45 tok/s).
- Suspiciously fast but bad: several ~42 tok/s rows that are reasoning-only/max-token loops (e.g., 16,18,88,90,92,109,123,125,126).

## Recommendations (Prioritized)

### Likely AFM bug
1. `agent-cached-turn*` connection failures (idx 46/48/50): caching mode appears to introduce transport instability.
2. Thinking-budget exhaustion with empty final content across many variants (idx 16,18,27,29,62,64,66,88,90,92,95,109,120,123,125,126): generation control likely failing to force final answer emission.
3. `<think>` leakage in raw/think modes (idx 69,70,85,86): internal reasoning should not be exposed in normal compatibility mode.

### Model quality / prompting issue
1. `seed-42-run*` limerick prompts loop in self-critique and never answer.
2. `short-output` (`max_tokens=50`) unsurprisingly produces reasoning-only truncation; not a server failure but poor UX default for reasoning models.
3. `repetition-penalty` long essay quality degrades relative to `no-penalty` / `with-penalty`.

### Working well (no immediate action)
1. Core QA prompts on `greedy/default/high-temp/top-k/top-p/min-p/combined-samplers` (capital + sky-blue tasks).
2. Most format/task prompts under `numbered-list`, `strict-format`, `math`, `multilingual`, `logprobs`, `long-output`.
3. Throughput consistency around ~40–42 tok/s for most non-agent variants.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":5},{"i":2,"s":5},{"i":3,"s":5},{"i":4,"s":5},{"i":5,"s":5},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":5},{"i":12,"s":5},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":2},{"i":17,"s":5},{"i":18,"s":2},{"i":19,"s":5},{"i":20,"s":4},{"i":21,"s":5},{"i":22,"s":4},{"i":23,"s":5},{"i":24,"s":3},{"i":25,"s":4},{"i":26,"s":4},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":2},{"i":30,"s":5},{"i":31,"s":5},{"i":32,"s":5},{"i":33,"s":5},{"i":34,"s":5},{"i":35,"s":3},{"i":36,"s":2},{"i":37,"s":3},{"i":38,"s":2},{"i":39,"s":5},{"i":40,"s":2},{"i":41,"s":5},{"i":42,"s":3},{"i":43,"s":5},{"i":44,"s":2},{"i":45,"s":5},{"i":46,"s":1},{"i":47,"s":5},{"i":48,"s":1},{"i":49,"s":5},{"i":50,"s":1},{"i":51,"s":2},{"i":52,"s":2},{"i":53,"s":5},{"i":54,"s":5},{"i":55,"s":5},{"i":56,"s":5},{"i":57,"s":5},{"i":58,"s":5},{"i":59,"s":5},{"i":60,"s":5},{"i":61,"s":5},{"i":62,"s":3},{"i":63,"s":5},{"i":64,"s":3},{"i":65,"s":5},{"i":66,"s":3},{"i":67,"s":5},{"i":68,"s":5},{"i":69,"s":3},{"i":70,"s":3},{"i":71,"s":5},{"i":72,"s":5},{"i":73,"s":5},{"i":74,"s":5},{"i":75,"s":5},{"i":76,"s":5},{"i":77,"s":3},{"i":78,"s":5},{"i":79,"s":2},{"i":80,"s":2},{"i":81,"s":5},{"i":82,"s":5},{"i":83,"s":5},{"i":84,"s":5},{"i":85,"s":3},{"i":86,"s":3},{"i":87,"s":5},{"i":88,"s":2},{"i":89,"s":5},{"i":90,"s":2},{"i":91,"s":5},{"i":92,"s":2},{"i":93,"s":5},{"i":94,"s":5},{"i":95,"s":2},{"i":96,"s":4},{"i":97,"s":5},{"i":98,"s":5},{"i":99,"s":5},{"i":100,"s":5},{"i":101,"s":5},{"i":102,"s":4},{"i":103,"s":4},{"i":104,"s":5},{"i":105,"s":4},{"i":106,"s":4},{"i":107,"s":5},{"i":108,"s":4},{"i":109,"s":2},{"i":110,"s":5},{"i":111,"s":4},{"i":112,"s":3},{"i":113,"s":5},{"i":114,"s":4},{"i":115,"s":5},{"i":116,"s":5},{"i":117,"s":5},{"i":118,"s":5},{"i":119,"s":5},{"i":120,"s":2},{"i":121,"s":5},{"i":122,"s":5},{"i":123,"s":2},{"i":124,"s":5},{"i":125,"s":2},{"i":126,"s":2},{"i":127,"s":5},{"i":128,"s":5},{"i":129,"s":5},{"i":130,"s":4},{"i":131,"s":5},{"i":132,"s":5}] -->
