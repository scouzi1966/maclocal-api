# Test Inventory

Master table of every test case across all test scripts.

## Automated Assertions (`test-assertions.sh`)

| # | Section | Test Name | Tier | Gap Covered |
|---|---------|-----------|------|-------------|
| 1 | Preflight | Binary exists | all | Basic sanity |
| 2 | Preflight | Server reachable | all | Server up check |
| 3 | Lifecycle | /v1/models contains model ID | all | Model registration |
| 4 | Lifecycle | Basic completion returns content | all | End-to-end smoke |
| 5 | Stop | Stop string absent from output | all | Stop correctness |
| 6 | Stop | finish_reason is 'stop' | all | Stop reason reporting |
| 7 | Stop | Multi-word stop truncates correctly | all | Multi-word stops |
| 8 | Stop | Stop on newline produces single line | all | Special char stops |
| 9 | Stop | Multiple stop sequences | all | Multi-stop array |
| 10 | Stop | Empty stop array is no-op | all | Edge case |
| 11 | Stop | Streaming: stop string absent | all | Streaming stop parity |
| 12 | Stop | Stop '3.' truncates list | std+ | Numbered list stop |
| 13 | Stop | Stop doesn't fire on partial match | std+ | Partial match safety |
| 14 | Stop | Stop fires mid-word | std+ | Substring matching |
| 15 | Logprobs | ChoiceLogprobs JSON schema valid | all | Schema correctness |
| 16 | Logprobs | top_logprobs count <= requested | all | Count enforcement |
| 17 | Logprobs | logprobs=false returns null | all | Opt-out works |
| 18 | Logprobs | top_logprobs=99 returns 400 | all | Validation |
| 19 | Logprobs | Streaming logprobs present | std+ | Streaming logprobs |
| 20 | Logprobs | top_logprobs=0 empty arrays | std+ | Zero count edge case |
| 21 | Think | reasoning_content present | all* | Think extraction |
| 22 | Think | No <think> tags in content | all* | Tag cleanup |
| 23 | Think | Streaming reasoning_content | all* | Streaming think |
| 24 | Think | Stop + think interaction | all* | Stop/think independence |
| 25 | Think | reasoning_content meaningful length | all* | Non-trivial reasoning |
| 26 | Tools | Basic tool call + finish_reason | all | Tool call basics |
| 27 | Tools | tool_choice=none suppresses | all | Tool suppression |
| 28 | Tools | Tool arguments valid JSON | std+ | Arg parsing |
| 29 | Tools | Streaming tool calls | std+ | Streaming tools |
| 30 | Tools | Multi-tool call | std+ | Multi-tool |
| 31 | Tools | No tools = text response | std+ | No-tool fallback |
| 32 | Cache | First request cached_tokens=0 | std+ | Cache miss reporting |
| 33 | Cache | Second identical cached_tokens>0 | std+ | Cache hit reporting |
| 34 | Cache | Different prompt cached_tokens=0 | std+ | Cache invalidation |
| 35 | Cache | Streaming cached_tokens>0 | std+ | Streaming cache |
| 36 | Concurrent | Two simultaneous 200 | std+ | Concurrent safety |
| 37 | Concurrent | Three simultaneous 200 | std+ | Higher concurrency |
| 38 | Error | Empty messages 400 | all | Input validation |
| 39 | Error | Malformed JSON 400 | all | Parse error handling |
| 40 | Error | Missing messages 400 | all | Required field |
| 41 | Error | json_object returns valid JSON | all | Response format |
| 42 | Error | max_tokens respected | all | Token limit |
| 43 | Error | OPTIONS 200 (CORS) | all | CORS support |
| 44 | Error | developer role accepted | all | Role mapping |
| 45 | Perf | TTFT < 5s | full | Latency |
| 46 | Perf | tok/s > 1 | full | Throughput |
| 47 | Perf | Long context 2K no crash | full | Context stability |
| 48 | Perf | Very long context 4K no NaN | full | SDPA regression |
| 49 | AdaptiveXML | Tool call valid (with/without grammar) | std+ | afm_adaptive_xml parsing |
| 50 | AdaptiveXML | Required params present (grammar-hardened) | std+ | EBNF param enforcement |
| 51 | AdaptiveXML | Structured param types (grammar-hardened) | std+ | EBNF type enforcement |
| 52 | Grammar | Calculator tool call (non-streaming) | std+ | Grammar enforcement (strict:true) |
| 53 | Grammar | Calculator tool call (streaming) | std+ | Streaming grammar |
| 54 | Grammar | Two tools: correct selection | std+ | Grammar + tool routing |
| 55 | Grammar | Two tools: selects calculate | std+ | Grammar + multi-tool |
| 56 | Grammar | 3 required params enforced (send_email) | std+ | Grammar completeness |
| 57 | Grammar | Array param constrained | std+ | Grammar typed constraint |
| 58 | Grammar | Array param via streaming | std+ | Streaming grammar typed |
| 59 | Grammar | Complex schema: string+int+array+object | std+ | Deep structure |
| 60 | StrictWiring | X-Grammar-Constraints header (tool strict:true) | all | Downgrade header present/absent |
| 61 | StrictWiring | X-Grammar-Constraints header (schema strict:true) | all | Downgrade header present/absent |
| 62 | StrictWiring | No header when strict absent | all | No false positive header |
| 63 | StrictWiring | Streaming json_schema strict:true valid JSON | all | Streaming schema grammar |
| 64 | StrictWiring | Streaming tool strict:true valid tool call | all | Streaming tool grammar |
| 65 | StrictWiring | strict:false does not error | all | Best-effort control |

\* Think tests auto-skip if model doesn't support `<think>` tags.
† Section 13 Grammar tests require `--grammar-constraints` flag. Section 14 adapts assertions based on `--grammar-constraints` presence.

| 66 | Batch | POST /v1/files upload returns file ID | std+ | File upload |
| 67 | Batch | GET /v1/files/:id returns file metadata | std+ | File retrieval |
| 68 | Batch | POST /v1/batches creates batch | std+ | Batch creation |
| 69 | Batch | GET /v1/batches/:id polls to completed | std+ | Batch polling |
| 70 | Batch | Output JSONL contains both results (2/2) | std+ | Output retrieval |
| 71 | Batch | GET /v1/batches lists completed batch | std+ | Batch listing |
| 72 | Batch | DELETE /v1/files/:id removes uploaded file | std+ | File deletion |
| 73 | Batch | SSE multiplex: 2 non-streaming requests tagged | std+ | SSE multiplex |
| 74 | Batch | SSE multiplex: streaming interleaved with finish | std+ | SSE streaming |
| 75 | Batch | SSE multiplex rejects duplicate custom_ids | std+ | Validation |
| 76 | Batch | SSE multiplex rejects empty requests | std+ | Validation |

## Smart Analysis Tests (`test-edge-cases.txt`)

| Label | Section | What AI Judge Evaluates |
|-------|---------|------------------------|
| think-basic | Think | Basic reasoning extraction quality |
| think-long-chain | Think | Extended multi-step reasoning coherence |
| think-stop-interaction | Think + Stop | Stop only affects visible content, not reasoning |
| stop-sentence-boundary | Stop | Clean truncation at period |
| stop-multi-word | Stop | Multi-word phrase stop quality |
| stop-stream-parity-ns | Stop | Non-streaming baseline for parity |
| stop-stream-parity-s | Stop | Streaming output matches non-streaming |
| tool-single | Tools | Single tool call argument quality |
| tool-multi | Tools | Multi-tool argument quality |
| tool-complex-args | Tools | Complex typed arguments |
| logprobs-valid | Logprobs | All logprobs <= 0, correct count |
| logprobs-min | Logprobs | Minimal top_logprobs=1 |
| cache-warmup | Cache | Baseline timing for cache pair |
| cache-hit | Cache | Faster timing on identical repeat |
| long-context-1k | SDPA | ~1K token input coherence |
| long-context-2k | SDPA | ~2K token input coherence |
| long-context-4k | SDPA | ~4K token input, NaN/garbage detection |
| json-basic | JSON | json_object produces valid JSON |
| json-system-msg | JSON | json_object + system message regression |
| stream-parity-ns | Streaming | Non-streaming baseline |
| stream-parity-s | Streaming | Streaming assembled output matches |

## Swift Unit Tests (`swift test`)

Run automatically by `test-assertions.sh --tier unit` (and all higher tiers).

| File | Tests | Purpose |
|------|-------|---------|
| `StreamingUsageChunkTests.swift` | 4 | Non-streaming finish_reason, usage summary chunk empty choices, Foundation commonPrefixLength, terminal chunk shape |
| `ConcurrentBatchTests.swift` | 10 | RequestSlot init/uniqueID/append/thread-safety/elapsed, StreamChunk defaults/timing/cached, MLXServiceError messages, BatchScheduler constants |
| `NullableToolSchemaTests.swift` | — | Nullable tool schema parsing |
| `XMLToolCallParsingTests.swift` | — | XML tool call format parsing |
| `StrictGrammarWiringTests.swift` | 16 | hasStrictTools() (nil/empty/true/false/nil/mixed), RequestToolFunction strict decoding (true/false/absent/array), ResponseJsonSchema strict decoding (true/false/absent), ResponseFormat json_schema strict detection (true/false/json_object) |
| `BatchDispatchTests.swift` | 71 | BatchStore file/batch CRUD, concurrent access, type serialization round-trips, JSONL parsing edge cases, Vapor integration tests for BatchAPIController (file endpoints, batch endpoints, validation, e2e dispatch+polling) and BatchCompletionsController (SSE headers, streaming/non-streaming, error events, slot reservation, mixed mode) |

## OpenAI Compatibility Evals (`test-openai-compat-evals.py`)

End-to-end OpenAI-python SDK compatibility matrix. Requires running server or `--start-server`.

| Test Name | What It Validates |
|-----------|-------------------|
| `models_endpoint` | GET /v1/models returns non-empty data |
| `openai_python_nonstream` | openai-python non-streaming chat completion |
| `openai_python_stream` | openai-python streaming + `stream_options.include_usage` usage chunk |
| `openai_python_nonstream_logprobs` | Non-streaming logprobs via openai-python |
| `openai_python_stream_logprobs` | Streaming logprobs + usage-only final chunk (empty choices) |
| `vllm_bench_smoke` | vllm bench serve: 4 prompts, 0 failures |

## Guided JSON Evals (`test-guided-json-evals.py`)

Structured output / `--guided-json` eval bundle. Tests API `json_schema`, streaming, CLI, and SDK parse.

| Test Pattern | What It Validates |
|--------------|-------------------|
| `api_json_schema::*` | Non-streaming JSON schema response validates against schema |
| `api_stream_json_schema::*` | Streaming JSON schema + usage chunk + schema validation |
| `api_conflict_json_schema::*` | Conflicting schema (impossible constraint) → refusal or valid |
| `api_truncation_json_schema::*` | Low max_tokens → finish_reason=length, truncated JSON |
| `api_unsupported_json_schema::*` | Edge-case schemas → accepted or gracefully rejected |
| `openai_python_parse::*` | openai-python `beta.chat.completions.parse()` with Pydantic |
| `cli_guided_json::*` | CLI `--guided-json` single-prompt mode |
| `cli_guided_json::no_think::*` | CLI `--guided-json --no-think` mode |
| `cli_guided_json::invalid_schema_error` | Invalid JSON schema rejected with error |

## Concurrent Batch Validation (`Scripts/feature-mlx-concurrent-batch/`)

Correctness and performance validation for batched MLX generation. Requires running server.

| Script | Tests | Purpose |
|--------|-------|---------|
| `validate_responses.py` | 8 × B={1,2,4,8} | Known-answer questions at various batch sizes; catches corrupt KV cache/masks |
| `validate_mixed_workload.py` | 8 × B={1,2,4,8} | Mixed short-answer + long-decode (4K max_tokens) with GPU metrics via mactop |
| `validate_multiturn_prefix.py` | 5 convs × B={1,2,4,8} | Multi-turn conversations with long system prompts; validates prefix cache hits under concurrency |

## Promptfoo Grammar-Constraints Suite (`run-promptfoo-agentic.sh grammar-constraints`)

7 phases testing the full CLI flag × `strict: true` Cartesian matrix. Each phase starts a server with a different profile and runs promptfoo eval.

| Phase | Server Profile | Config | Tests | Covers |
|---|---|---|---|---|
| P1 | `default` (no grammar) | schema + tools | 12 | Downgrade path (strict:true silently downgraded) |
| P2 | `grammar-enabled` | schema + tools | 12 | Enforcement path (xgrammar active) |
| P3 | `grammar-enabled-adaptive-xml` | tools | 7 | Parser flag regression (grammar works without afm_adaptive_xml) |
| P4 | `grammar-enabled-concurrent` | schema + tools | 12 | Concurrent/batch path grammar |
| P5 | `grammar-enabled-prefix-cache` | schema + tools | 12 | Prefix caching + grammar interaction |
| P6 | `grammar-enabled` | mixed-strict | 1 | Schema + tool strict priority (json_schema wins) |
| P7 | `default` + `grammar-enabled` | header assertions | 4 | X-Grammar-Constraints: downgraded header via JS judge |

Custom assertion judge: `judges/assert-grammar-header.mjs` — validates presence/absence of `X-Grammar-Constraints` response header.


## Other Test Scripts

| Script | Tests | Purpose |
|--------|-------|---------|
| `test-sampling-params.sh` | 21 | Seed, temperature, top_p, top_k, min_p, presence_penalty |
| `test-structured-outputs.sh` | ~10 | JSON schema, json_object response format |
| `test-tool-call-parsers.py` | ~15 | Unit tests for XML/JSON tool call parsing |
| `regression-test.sh` | ~5 | Quick smoke test for CI |
