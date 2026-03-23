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

\* Think tests auto-skip if model doesn't support `<think>` tags.

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

## GPU Shader Profile (`gpu-profile-report.py`)

Full harness: mactop bandwidth + `--gpu-profile` + `--gpu-trace` + shader extraction + HTML report.

| Metric | Source | What It Measures |
|--------|--------|------------------|
| Decode tok/s | `--gpu-profile` | Token generation throughput |
| Prefill tok/s | `--gpu-profile` | Prompt processing throughput |
| Memory breakdown | `--gpu-profile` | Model weights vs KV cache vs peak |
| DRAM bandwidth (GB/s) | mactop via PTY | Read + write bandwidth timeline |
| GPU utilization (%) | mactop via PTY | GPU active percentage timeline |
| GPU power (W) | mactop via PTY | Power consumption timeline |
| Metal kernel names | xctrace Shader Timeline | Per-kernel shader inventory (e.g. `affine_qmv_fast`, `steel_gemm_fused`) |
| Command line | `--gpu-profile` | Exact reproducible command |

**Key files:**
- `Scripts/gpu-profile-report.py` — full harness (run with `python3 Scripts/gpu-profile-report.py [model]`)
- `Scripts/gpu-profile.sh` — individual profiling helpers (bandwidth, capture, trace, power)
- `Scripts/create-shader-template.py` — one-time setup: patches Instruments template for Shader Timeline
- Report output: `/tmp/afm-gpu-profile.html`, `/tmp/afm-metal.trace`, `/tmp/afm-gpu-samples.json`

**CLI flags (can be used on any `afm mlx` invocation):**
- `--gpu-profile` — zero-overhead per-request stats
- `--gpu-profile-bw` — + mactop bandwidth sampling (~5s)
- `--gpu-trace N` — xctrace Metal System Trace for N seconds
- `--gpu-capture <path>` — full Metal GPU capture (small models only)

## Other Test Scripts

| Script | Tests | Purpose |
|--------|-------|---------|
| `test-sampling-params.sh` | 21 | Seed, temperature, top_p, top_k, min_p, presence_penalty |
| `test-structured-outputs.sh` | ~10 | JSON schema, json_object response format |
| `test-tool-call-parsers.py` | ~15 | Unit tests for XML/JSON tool call parsing |
| `regression-test.sh` | ~5 | Quick smoke test for CI |
