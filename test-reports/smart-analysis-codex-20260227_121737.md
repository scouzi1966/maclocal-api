## AFM Compatibility Report (JSONL: `mlx-model-report-20260227_065710.jsonl`)

### 1) Broken Models
No hard failures (`status=FAIL`) were present, so there are no load/crash failures to group by architecture/file/timeout.

### 2) Anomalies & Red Flags

| Type | Lines | Evidence (first ~100 chars) | Likely Cause |
|---|---:|---|---|
| Thinking-budget exhaustion (empty `content`, near max tokens) | 16, 18, 25, 27, 29, 95, 106, 120, 123, 125, 126 | `"Thinking Process:\n\n1. **Analyze the Request:** ..."` | Model spent full budget in reasoning; no final answer emitted. |
| Empty/truncated output from stop handling | 72, 74, 75, 76 | `line72 content: "1. Apple\n2. Banana"` / `line74 content: ""` | Stop-sequence interaction appears to terminate before usable output. |
| Raw/think leakage into user-visible `content` | 69, 70, 85, 86 | `"<think>Thinking Process:\n\n1. **Analyze the Request:** ..."` | AFM output post-processing not stripping internal reasoning in raw modes. |
| Structured-output mode not enforced (`guided-json` / `response_format`) | 35, 36, 37, 38, 77, 79, 80 | `"The capital of Japan is Tokyo."` under schema/json modes | AFM structured decoding/validator path likely bypassed or weakly enforced. |
| Tool-call mode not actually invoking tools | 103, 106, 109 | `"I don't have access to real-time data..."` despite tool schemas in prior lines | Tool parser/dispatch integration issue in chat-completions pipeline. |
| Repetitive/degenerate reasoning loops | 25, 27, 29, 95, 106, 120, 123, 125, 126 | repeated `"Wait, ..."` loops in reasoning until token cap | Decoding instability under persona/system-format constraints. |
| Truncated long-form answer | 54, 24 | content ends mid-thought | Hit `max_tokens`; partial usability only. |

Explicit required pattern callout: **thinking-budget exhaustion** is present multiple times (not harness failure): non-empty `reasoning_content`, empty `content`, `completion_tokens` ~= `max_tokens`.

### 3) Variant Comparison

- **Sampler A/Bs (`greedy/default/high-temp/top-p/top-k/min-p/combined`)**: Final user-facing answers are consistently correct for simple factual + sky-blue prompts; no major quality regression from temperature/top-p/top-k changes.
- **`seed-42-run1/run2`**: Deterministic reproduction of the same failure on limerick prompt (both hit 4096 reasoning tokens, no final content).
- **Penalty variants (`with-penalty`, `repetition-penalty`)**: Slower than baseline and long essays more likely to degrade/truncate.
- **Streaming vs non-streaming seeded** (`streaming-seeded` vs `non-streaming-seeded`): Essentially identical outputs and behavior.
- **Prefill step sizes** (`prefill-default`, `prefill-large-4096`, `prefill-small-256`): No observable quality difference in outputs.
- **Tool-call parser modes** (`auto/xml/multi`): setup prompts look fine, but real weather/time requests do not execute tools reliably; xml mode includes a max-token loop failure (line 106).

### 4) Quality Assessment (Coherence / Relevance)

- **Generally strong (4–5/5)**: Core QA/math/translation/blog responses and most sampler variants.
- **Flagged (<3 on either metric):**
  - 16, 18, 25, 27, 29, 40, 42, 44, 46, 48, 50, 51, 52, 69, 72, 74, 75, 76, 85, 95, 106, 120, 123, 125, 126.
- Main failure modes: empty final content, endless internal loops, stop-triggered truncation, strict-format noncompliance.

### 5) Performance Summary (by variant label, avg tokens/sec, high→low)

| Variant | Avg tok/s | Note |
|---|---:|---|
| code-swift | 122.5 | suspiciously fast but both major prompts failed (empty content) |
| code-python | 120.7 | one severe loop failure + one good code response |
| max-completion-tokens | 116.6 | generally fast; one config-ack line non-critical |
| guided-json-nested | 115.5 | fast but structured mode not enforced |
| guided-json-simple | 115.1 | fast but structured mode not enforced |
| developer-role | 115.0 | mixed; one severe loop |
| tool-call-xml | 114.3 | includes severe loop on weather request |
| response-format-json | 114.0 | fast; json mode partly ignored |
| long-form | 113.7 | good quality |
| tool-call-multi | 113.6 | tool execution weak on real query |
| prefill-default | 113.6 | stable |
| prefill-large-4096 | 113.6 | stable |
| prefill-small-256 | 113.3 | stable |
| numbered-list | 113.2 | stable |
| no-streaming | 112.8 | stable |
| long-output | 112.3 | one truncated long response |
| think-raw | 112.3 | reasoning leakage in content |
| seed-42-run1 | 112.4 | one hard degeneration |
| no-penalty | 111.8 | mostly good |
| think-normal | 111.8 | good |
| verbose | 111.7 | good |
| math | 111.4 | good |
| default | 111.3 | good |
| small-kv | 111.3 | good |
| kv-quantized | 111.3 | good |
| seed-42-run2 | 111.0 | one hard degeneration |
| raw-mode | 111.0 | reasoning leakage |
| strict-format | 110.1 | one strict-format pass, one pass |
| long-prompt | 110.3 | good |
| top-k | 110.5 | good |
| non-streaming-seeded | 110.0 | good |
| streaming-seeded | 109.9 | good |
| special-chars | 109.8 | severe loop on special-char repeat |
| high-temp | 109.4 | good |
| response-format-schema | 109.2 | schema mode not enforced |
| very-verbose | 109.1 | good |
| multilingual | 108.8 | good |
| tool-call-none | 108.6 | expected no live tools |
| top-p | 108.1 | good |
| logprobs | 108.1 | good |
| greedy | 107.9 | good |
| stop-multi | 107.0 | one empty-output failure |
| min-p | 106.6 | good |
| response-format-text | 105.3 | good |
| minimal-prompt | 104.4 | good |
| combined-samplers | 103.9 | good |
| stop-newline | 99.3 | both outputs empty |
| stop-single | 95.0 | second prompt truncated to 2 items |
| with-penalty | 90.7 | slow outlier |
| repetition-penalty | 91.3 | slow outlier |
| agent-cached-turn2 | 92.1 | mixed/incomplete agent actions |
| agent-cached-turn3 | 86.0 | mixed/incomplete agent actions |
| agent-no-cache-turn3 | 83.7 | mixed/incomplete agent actions |
| agent-cached-turn1 | 81.0 | mixed/incomplete agent actions |
| agent-no-cache-turn2 | 80.4 | mixed/incomplete agent actions |
| short-output | 79.2 | frequent budget truncation |
| agent-no-cache-turn1 | 68.6 | slowest + incomplete tool behavior |

Outliers:
- **Suspiciously fast + bad quality**: `code-swift`, `code-python`, several persona-loop cases.
- **Unusually slow**: agent-no-cache/cached variants, short-output, penalty variants.

### 6) Recommendations (prioritized)

#### Likely AFM bug
1. **Structured output enforcement** (`guided-json`, `response_format json/json_schema`)  
   - Failing lines: 35, 36, 37, 38, 77, 79, 80  
   - Check schema-constrained decoding/validator path in chat completion response assembly.
2. **Stop-sequence handling** (premature empty/truncated output)  
   - Failing lines: 72, 74, 75, 76  
   - Inspect stop matcher interaction with reasoning/content channel split.
3. **Tool-calling execution path** (tools declared but not invoked on actionable prompts)  
   - Failing lines: 103, 106, 109  
   - Inspect tool-call parser + dispatcher wiring in `/v1/chat/completions`.
4. **Reasoning leakage / raw-mode isolation**  
   - Failing lines: 69, 70, 85, 86  
   - Ensure `<think>`/reasoning tokens are filtered from user-facing `content` unless explicitly requested.
5. **Loop/degen under persona/system constraints**  
   - Failing lines: 25, 27, 29, 95, 106, 120, 123, 125, 126  
   - Add repetition guardrails or EOS heuristics for reasoning channel.

#### Model quality issue
1. **Reasoning-over-answer behavior under some prompts** (limerick/persona/special-char repeats)  
   - Lines: 16, 18, 25, 27, 29, 120  
   - Workaround: reduce reasoning budget, force concise response mode, cap hidden-thought tokens.

#### Working well (no action needed)
- Core sampler variants for factual/science prompts (1–15, 31–34, 53, 55–60, 67, 71, 81–84, 87–94, 97–101, 110, 113, 115–119, 121–122, 127–132) are generally coherent and relevant.

<!-- AI_SCORES [{"i":0,"s":3},{"i":1,"s":5},{"i":2,"s":5},{"i":3,"s":5},{"i":4,"s":5},{"i":5,"s":5},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":5},{"i":12,"s":5},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":2},{"i":17,"s":5},{"i":18,"s":2},{"i":19,"s":5},{"i":20,"s":4},{"i":21,"s":5},{"i":22,"s":4},{"i":23,"s":5},{"i":24,"s":3},{"i":25,"s":2},{"i":26,"s":5},{"i":27,"s":2},{"i":28,"s":5},{"i":29,"s":2},{"i":30,"s":5},{"i":31,"s":5},{"i":32,"s":5},{"i":33,"s":5},{"i":34,"s":5},{"i":35,"s":3},{"i":36,"s":2},{"i":37,"s":3},{"i":38,"s":2},{"i":39,"s":5},{"i":40,"s":2},{"i":41,"s":5},{"i":42,"s":2},{"i":43,"s":5},{"i":44,"s":2},{"i":45,"s":5},{"i":46,"s":2},{"i":47,"s":5},{"i":48,"s":2},{"i":49,"s":5},{"i":50,"s":2},{"i":51,"s":2},{"i":52,"s":2},{"i":53,"s":5},{"i":54,"s":3},{"i":55,"s":5},{"i":56,"s":5},{"i":57,"s":5},{"i":58,"s":5},{"i":59,"s":5},{"i":60,"s":5},{"i":61,"s":5},{"i":62,"s":5},{"i":63,"s":5},{"i":64,"s":5},{"i":65,"s":5},{"i":66,"s":5},{"i":67,"s":5},{"i":68,"s":4},{"i":69,"s":2},{"i":70,"s":3},{"i":71,"s":5},{"i":72,"s":2},{"i":73,"s":5},{"i":74,"s":2},{"i":75,"s":2},{"i":76,"s":2},{"i":77,"s":4},{"i":78,"s":5},{"i":79,"s":4},{"i":80,"s":3},{"i":81,"s":5},{"i":82,"s":5},{"i":83,"s":5},{"i":84,"s":5},{"i":85,"s":2},{"i":86,"s":3},{"i":87,"s":5},{"i":88,"s":4},{"i":89,"s":5},{"i":90,"s":4},{"i":91,"s":5},{"i":92,"s":4},{"i":93,"s":5},{"i":94,"s":5},{"i":95,"s":2},{"i":96,"s":5},{"i":97,"s":5},{"i":98,"s":5},{"i":99,"s":5},{"i":100,"s":5},{"i":101,"s":5},{"i":102,"s":4},{"i":103,"s":3},{"i":104,"s":5},{"i":105,"s":4},{"i":106,"s":2},{"i":107,"s":5},{"i":108,"s":4},{"i":109,"s":3},{"i":110,"s":5},{"i":111,"s":4},{"i":112,"s":4},{"i":113,"s":5},{"i":114,"s":3},{"i":115,"s":5},{"i":116,"s":5},{"i":117,"s":5},{"i":118,"s":5},{"i":119,"s":5},{"i":120,"s":2},{"i":121,"s":5},{"i":122,"s":5},{"i":123,"s":2},{"i":124,"s":5},{"i":125,"s":2},{"i":126,"s":2},{"i":127,"s":5},{"i":128,"s":5},{"i":129,"s":5},{"i":130,"s":5},{"i":131,"s":5},{"i":132,"s":5}] -->
