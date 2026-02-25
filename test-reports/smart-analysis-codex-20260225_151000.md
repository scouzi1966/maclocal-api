1) **Broken Models**
- `agent-cached-turn1/2/3`: hard failures (`status: FAIL`, connection errors on task turns).
- `scientist` (capital prompt): empty `content`, maxed output in `reasoning_content` loop.
- `eli5` (capital prompt): same failure mode (empty `content`, runaway reasoning).
- `seed-42-run1` and `seed-42-run2` (limerick): empty `content`, hit `completion_tokens=4096`.
- `short-output` (both prompts): truncates into reasoning, no user-facing answer.
- `prefill-default`, `prefill-large-4096`, `prefill-small-256` (bottleneck prompt): empty `content`, maxed reasoning.
- `streaming-seeded` and `non-streaming-seeded` (4-line poem): empty `content`, maxed reasoning.
- `max-completion-tokens` (`max_completion_tokens: 100` prompt): runaway repetition, empty `content`.
- `special-chars` (repeat exactly): empty `content`, runaway reasoning.
- `code-python` (capital prompt) and `code-swift` (both prompts): runaway/empty `content`.
- `tool-call-multi` (weather+time): runaway, empty `content`.

2) **Anomalies**
- Severe hidden-reasoning leakage pattern across many rows (`reasoning_content` huge; some rows only reasoning, no answer).
- Multiple “`status: OK` but functionally failed” cases (empty `content` with 4096 completion tokens).
- Structured output non-compliance:
  - `response-format-json` (capital prompt) ignored JSON-object requirement.
  - `response-format-schema` rows ignored schema entirely.
  - `guided-json-simple/nested` task rows ignored guided schema.
- Tool-use non-compliance:
  - Weather/tool-call variants often asked follow-up or refused real-time instead of emitting tool calls.
- Prompt-role instability: certain role/system prompts trigger repetitive loops.

3) **Variant Comparison**
- **Best stability (basic QA/format tasks):** `default`, `high-temp`, `top-p`, `top-k`, `min-p`, `combined-samplers` on short factual prompts.
- **Determinism check:** `seed-42-run1` and `seed-42-run2` are reproducibly identical (including same catastrophic limerick failure).
- **Cache comparison:** `agent-cached-*` had transport failures; `agent-no-cache-*` returned partial/planning text instead of completing tool tasks.
- **Prefill step size (256 vs 4096):** similar behavior on easy prompt; both failed identically on hard architecture prompt (runaway reasoning).
- **Streaming vs non-streaming seeded:** same failure profile on poem prompt.
- **Raw/think modes:** answers produced but with `<think>` leakage (undesirable for production).

4) **Quality Assessment**
- Factual correctness is high on simple prompts when `content` is present.
- Constraint-following is inconsistent under stricter formatting/tool/schema conditions.
- Major quality risk is **answer-channel starvation** (all tokens spent in reasoning).
- Effective quality is bimodal: either clean/correct or catastrophic loop/truncation.

5) **Performance Summary**
- Throughput is generally stable around ~40–42 tok/s on successful generations.
- Long failing runs cluster around ~97–104s with `completion_tokens=4096`.
- Explicit hard-fail transport errors are low (3 rows), but behavioral failures are much higher.
- Load time mostly ~1s; some variants show 2s cold-start/load.

6) **Recommendations**
- Enforce strict guardrails: if `content` is empty after N tokens, abort and regenerate with reasoning disabled.
- Disable/strip reasoning in production paths; never allow reasoning to consume full budget.
- Add hard stop/loop detectors (repeated substring / repeated thought blocks).
- Strengthen schema/tool compliance tests (`response_format`, `guided-json`, function-call emission).
- Add per-variant canary suite for: exact-format prompts, tool-call prompts, and max-token stress prompts.
- For cached agent path, add retry/backoff and fallback to non-cached execution on connection error.
- Track “functional success” KPI (non-empty valid `content`) in addition to `status`.

<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":5},{"i":2,"s":4},{"i":3,"s":5},{"i":4,"s":5},{"i":5,"s":5},{"i":6,"s":5},{"i":7,"s":5},{"i":8,"s":5},{"i":9,"s":5},{"i":10,"s":5},{"i":11,"s":5},{"i":12,"s":5},{"i":13,"s":5},{"i":14,"s":5},{"i":15,"s":5},{"i":16,"s":1},{"i":17,"s":5},{"i":18,"s":1},{"i":19,"s":5},{"i":20,"s":4},{"i":21,"s":5},{"i":22,"s":4},{"i":23,"s":5},{"i":24,"s":2},{"i":25,"s":5},{"i":26,"s":5},{"i":27,"s":1},{"i":28,"s":5},{"i":29,"s":1},{"i":30,"s":5},{"i":31,"s":5},{"i":32,"s":5},{"i":33,"s":5},{"i":34,"s":5},{"i":35,"s":4},{"i":36,"s":1},{"i":37,"s":4},{"i":38,"s":1},{"i":39,"s":5},{"i":40,"s":2},{"i":41,"s":5},{"i":42,"s":2},{"i":43,"s":5},{"i":44,"s":2},{"i":45,"s":5},{"i":46,"s":1},{"i":47,"s":5},{"i":48,"s":1},{"i":49,"s":5},{"i":50,"s":1},{"i":51,"s":1},{"i":52,"s":1},{"i":53,"s":5},{"i":54,"s":4},{"i":55,"s":5},{"i":56,"s":5},{"i":57,"s":5},{"i":58,"s":5},{"i":59,"s":5},{"i":60,"s":5},{"i":61,"s":5},{"i":62,"s":1},{"i":63,"s":5},{"i":64,"s":1},{"i":65,"s":5},{"i":66,"s":1},{"i":67,"s":5},{"i":68,"s":5},{"i":69,"s":2},{"i":70,"s":2},{"i":71,"s":5},{"i":72,"s":5},{"i":73,"s":5},{"i":74,"s":5},{"i":75,"s":5},{"i":76,"s":5},{"i":77,"s":1},{"i":78,"s":5},{"i":79,"s":1},{"i":80,"s":1},{"i":81,"s":5},{"i":82,"s":5},{"i":83,"s":5},{"i":84,"s":5},{"i":85,"s":2},{"i":86,"s":2},{"i":87,"s":5},{"i":88,"s":1},{"i":89,"s":5},{"i":90,"s":1},{"i":91,"s":5},{"i":92,"s":1},{"i":93,"s":5},{"i":94,"s":5},{"i":95,"s":1},{"i":96,"s":4},{"i":97,"s":5},{"i":98,"s":5},{"i":99,"s":5},{"i":100,"s":5},{"i":101,"s":5},{"i":102,"s":4},{"i":103,"s":2},{"i":104,"s":5},{"i":105,"s":4},{"i":106,"s":2},{"i":107,"s":5},{"i":108,"s":4},{"i":109,"s":1},{"i":110,"s":5},{"i":111,"s":4},{"i":112,"s":3},{"i":113,"s":5},{"i":114,"s":4},{"i":115,"s":5},{"i":116,"s":5},{"i":117,"s":5},{"i":118,"s":5},{"i":119,"s":5},{"i":120,"s":1},{"i":121,"s":5},{"i":122,"s":5},{"i":123,"s":1},{"i":124,"s":5},{"i":125,"s":1},{"i":126,"s":1},{"i":127,"s":5},{"i":128,"s":5},{"i":129,"s":5},{"i":130,"s":4},{"i":131,"s":5},{"i":132,"s":5}] -->
