## Summary
- Total JSONL lines: **133** (`1` metadata + `132` test results)
- Reported status: **132 OK / 0 FAIL**
- Effective quality health: **Degraded** despite 100% status, due to multiple `OK` tests with empty `content` and feature non-compliance.

## Performance
- Throughput over 132 tests: **min 46.11 / avg 107.95 / max 125.37 tok/s**
- Low outliers: mainly agent/cache and short-output control tests (indices `39,41,43,45,47,49,51`)
- High band (~124-125 tok/s): long generations and structured tasks, consistent and stable.

## Issues Found
- **Empty final output with reasoning-only preview (`content=""` but thinking text in preview)**  
  - Tests: `seed-42-run1`(16), `seed-42-run2`(18), `pirate`(25), `scientist`(27), `eli5`(29), `short-output`(51,52), `stop-multi`(74), `stop-newline`(75,76), `tool-call-xml`(106), `special-chars`(120), `code-python`(123), `code-swift`(125,126)  
  - Severity: **critical**  
  - Likely root cause: response channeling bug (reasoning/text routed to preview/reasoning field, empty assistant content), possibly interacting with stop logic or persona/system formatting.
- **Stop-sequence behavior truncating/emptying user-visible output**  
  - Tests: `stop-single`(72, only 2/10 fruits), `stop-multi`(74 empty), `stop-newline`(75,76 empty)  
  - Severity: **critical**  
  - Likely root cause: stop patterns matching too early/common tokens or applied before first visible token commit.
- **Tool calling not executed in tool-enabled scenarios**  
  - Tests: `tool-call-auto`(103), `tool-call-xml`(106), `tool-call-multi`(109), `tool-call-complex`(112)  
  - Severity: **warning**  
  - Likely root cause: tool policy/parser not triggering call objects; model falls back to plain text guidance/refusal.
- **Structured output/schema compliance drift**  
  - Tests: `guided-json-simple`(36), `response-format-schema`(80), minor format miss in `numbered-list`(34)  
  - Severity: **warning**  
  - Likely root cause: weak enforcement for strict format/schema constraints; instruction hierarchy not consistently applied.
- **Logprobs feature appears non-functional**  
  - Tests: `logprobs`(55,56) with `logprobs_count=0`  
  - Severity: **warning**  
  - Likely root cause: backend not returning token logprobs despite mode request.
- **Instruction-following regression in developer-mode constraint**  
  - Tests: `developer-role`(95,96)  
  - Severity: **info/warning**  
  - Likely root cause: role conditioning conflict with generic assistant style template.

## Feature Coverage
- Sampling variants (`greedy/default/high-temp/top-p/top-k/min-p/combined`): **mostly pass**
- Seed determinism (`seed-42-run1/run2`): **partially pass** (deterministic failure on limerick: both empty)
- Penalties (`with/no/repetition`): **pass**
- Persona/style (`pirate/scientist/eli5`): **degraded** (capital prompt outputs empty in multiple persona runs)
- JSON/format controls (`json-output`, `guided-json*`, `response-format-*`, `strict-format`): **mixed**
- Stop sequences (`stop-*`): **failed/degraded**
- Thinking/raw modes (`raw-mode`, `think-normal`, `think-raw`): **pass** for configured raw behavior
- Streaming parity (`streaming-seeded` vs `non-streaming-seeded`): **pass** (matching poem)
- Context/cache agent turns (`agent-*`): **mixed** (planning/tool-intent responses without task completion)
- Tool calls (`tool-call-*`): **degraded**
- Logprobs: **failed/degraded** (no logprobs returned)
- Code generation (`code-python`, `code-swift`): **mixed** (Swift severe failure due empty output)

## Recommendations
1. **Fix output channel integrity first**: guarantee non-empty `content` when generation succeeds; add invariant checks (`status=OK` + `content.len>0` unless explicit null-output mode).
2. **Harden stop-sequence handling**: apply stop matching after minimum visible token threshold and add regression tests for premature-stop edge cases.
3. **Implement strict tool-call mode**: when tools are provided and query is tool-satisfiable, require function-call output or explicit tool-unavailable error object.
4. **Enforce schema/format constraints at decode time**: constrained decoding for JSON/schema tests; reject markdown wrappers when strict JSON requested.
5. **Validate logprobs pipeline** end-to-end and add assertion that `logprobs_count>0` in logprobs-enabled tests.
6. **Improve QA pass criteria**: supplement `status` with semantic validators (format checks, emptiness checks, tool-call checks, schema checks) before marking pass.
7. **Add per-feature fail gates** in CI so “all OK” cannot mask functional regressions.

[{"i":0,"s":10},{"i":1,"s":10},{"i":2,"s":10},{"i":3,"s":10},{"i":4,"s":10},{"i":5,"s":10},{"i":6,"s":10},{"i":7,"s":10},{"i":8,"s":10},{"i":9,"s":10},{"i":10,"s":10},{"i":11,"s":10},{"i":12,"s":10},{"i":13,"s":10},{"i":14,"s":10},{"i":15,"s":10},{"i":16,"s":1},{"i":17,"s":10},{"i":18,"s":1},{"i":19,"s":10},{"i":20,"s":9},{"i":21,"s":10},{"i":22,"s":9},{"i":23,"s":10},{"i":24,"s":9},{"i":25,"s":1},{"i":26,"s":9},{"i":27,"s":1},{"i":28,"s":9},{"i":29,"s":1},{"i":30,"s":9},{"i":31,"s":10},{"i":32,"s":10},{"i":33,"s":10},{"i":34,"s":7},{"i":35,"s":10},{"i":36,"s":6},{"i":37,"s":10},{"i":38,"s":9},{"i":39,"s":10},{"i":40,"s":6},{"i":41,"s":10},{"i":42,"s":6},{"i":43,"s":10},{"i":44,"s":7},{"i":45,"s":10},{"i":46,"s":6},{"i":47,"s":10},{"i":48,"s":6},{"i":49,"s":10},{"i":50,"s":6},{"i":51,"s":1},{"i":52,"s":1},{"i":53,"s":10},{"i":54,"s":9},{"i":55,"s":10},{"i":56,"s":10},{"i":57,"s":10},{"i":58,"s":9},{"i":59,"s":10},{"i":60,"s":9},{"i":61,"s":10},{"i":62,"s":9},{"i":63,"s":10},{"i":64,"s":9},{"i":65,"s":10},{"i":66,"s":9},{"i":67,"s":10},{"i":68,"s":8},{"i":69,"s":8},{"i":70,"s":8},{"i":71,"s":10},{"i":72,"s":4},{"i":73,"s":10},{"i":74,"s":1},{"i":75,"s":1},{"i":76,"s":1},{"i":77,"s":10},{"i":78,"s":10},{"i":79,"s":10},{"i":80,"s":5},{"i":81,"s":10},{"i":82,"s":9},{"i":83,"s":10},{"i":84,"s":10},{"i":85,"s":8},{"i":86,"s":8},{"i":87,"s":10},{"i":88,"s":10},{"i":89,"s":10},{"i":90,"s":10},{"i":91,"s":10},{"i":92,"s":8},{"i":93,"s":8},{"i":94,"s":10},{"i":95,"s":4},{"i":96,"s":5},{"i":97,"s":10},{"i":98,"s":9},{"i":99,"s":10},{"i":100,"s":9},{"i":101,"s":10},{"i":102,"s":7},{"i":103,"s":4},{"i":104,"s":10},{"i":105,"s":7},{"i":106,"s":1},{"i":107,"s":10},{"i":108,"s":7},{"i":109,"s":4},{"i":110,"s":10},{"i":111,"s":7},{"i":112,"s":5},{"i":113,"s":10},{"i":114,"s":9},{"i":115,"s":10},{"i":116,"s":9},{"i":117,"s":10},{"i":118,"s":10},{"i":119,"s":10},{"i":120,"s":1},{"i":121,"s":10},{"i":122,"s":10},{"i":123,"s":1},{"i":124,"s":9},{"i":125,"s":1},{"i":126,"s":1},{"i":127,"s":10},{"i":128,"s":10},{"i":129,"s":10},{"i":130,"s":9},{"i":131,"s":10},{"i":132,"s":10}]
