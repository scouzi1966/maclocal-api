# Issue 192 Phase A: vLLM Metrics and GuideLLM Interoperability Plan

Status: revised planning checkpoint only. No feature code is included. Implementation remains blocked pending architecture re-review.

Issue: <https://github.com/scouzi1966/maclocal-api/issues/192>

## Scope and pinned compatibility targets

Issue 192 has two compatibility goals:

1. Expose metrics that an unmodified vLLM Playground can scrape while retaining every existing `afm:*` metric and its current meaning.
2. Qualify `/v1/chat/completions` and a true raw-prompt `/v1/completions` implementation with GuideLLM in streaming and non-streaming modes.

This is not a claim of general vLLM or OpenAI conformance. Implementation and fixtures are pinned to the revisions inspected on 2026-08-17:

- GuideLLM `97b3077c05a367599112fd7080082c2d32c14b7e`.
- vLLM Playground `76276229092455f9ef66748731e4a615f4d80720`.
- vLLM `9633933dd81228fbcae07969f20881ad0b7cb766`.

## Current-state evidence

### Metrics and telemetry

- `MetricsController` registers only `GET /metrics`, returns Prometheus text 0.0.4, and renders one `StatsAggregator` snapshot (`Sources/AFMServer/Controllers/MetricsController.swift:22`, line 26, line 64).
- The exposition contains only `afm:*`: live gauges begin at `Sources/AFMServer/Controllers/MetricsController.swift:82`, counters at line 128, finish reasons at line 160, histograms at line 173, and process metadata at line 252.
- `StatsAggregator` is currently a process singleton implemented in `AFMKitMLX`, while AFMServer owns HTTP rendering (`Sources/AFMKitMLX/Models/StatsAggregator.swift:18`, line 23, line 37). That concrete ownership cannot support sibling runtimes without a dependency violation.
- Existing state covers generated and computed prompt tokens, accepted/completed requests, request-level radix hit/miss events, and terminal reasons (`Sources/AFMKitMLX/Models/StatsAggregator.swift:126`). It has no full-prompt total, failure-status counter, preemption counter, token-level prefix metrics, speculative totals, per-token ITL, or rolling throughput.
- `observeRequest` derives E2E, queue, inference, prefill, TTFT, and one average TPOT observation per request (`Sources/AFMKitMLX/Models/StatsAggregator.swift:304`, line 340). It cannot reconstruct ITL.
- Batched prefix caching adds only the uncached suffix to the current AFM prompt counter and records cache hit/miss once per request (`Sources/AFMKitMLX/Models/BatchScheduler.swift:1359`). Pinned vLLM instead increments `prompt_tokens_total` with full prompt tokens, including cached tokens, while prompt throughput uses locally computed tokens.
- `BatchScheduler` owns running/waiting state, registers gauge readers at `Sources/AFMKitMLX/Models/BatchScheduler.swift:590`, accepts work at line 654, and records terminal observations at lines 1979-1992.
- Serial MLX paths update the same singleton (`Sources/AFMKitMLX/Models/MLXModelService.swift:2100`, line 2433, line 3821). Some speculative fast paths terminate without a complete token/latency observation, including lines 2177 and 2254.
- `afm:gpu_cache_usage_perc` is Metal working-set pressure including weights and intermediates, not KV occupancy (`Sources/AFMServer/Controllers/MetricsController.swift:103`). It cannot back `vllm:kv_cache_usage_perc`.
- DwarfStar already tracks accepted speculative tokens (`Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift:647`) but cannot import AFMKitMLX because both are sibling targets depending on AFMKitCore (`Package.swift:166`, line 188).
- No existing test directly exercises the aggregator, metrics renderer, Prometheus parsing, or AFM/vLLM value parity.

### OpenAI routes and semantics

- Both runtime controllers register only `POST /v1/chat/completions`; there is no `/v1/completions` route or raw completion DTO (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:90`, `Sources/AFMServer/Controllers/ChatCompletionsController.swift:40`, `Sources/AFMOpenAICompat/OpenAIRequest.swift:3`).
- The existing serving abstraction accepts message arrays. MLX's input path inserts instructions and applies a chat template, so it cannot provide raw completion semantics by wrapping a prompt in a chat message.
- Chat accepts `max_tokens` and `max_completion_tokens`, preferring the former (`Sources/AFMOpenAICompat/OpenAIRequest.swift:8`, line 66). Unknown GuideLLM extensions such as `ignore_eos` and `continuous_usage_stats` are ignored by Codable.
- Streaming usage defaults to enabled when `stream_options` is absent (`Sources/AFMOpenAICompat/OpenAIRequest.swift:60`), a behavior locked by `Tests/MacLocalAPITests/AgentFriendlyTier1Tests.swift:52` and out of scope to change incidentally.
- MLX success currently emits a finish chunk, optional usage-only chunk, then `[DONE]` (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1453`). MLX normally obtains runtime token counts but can fall back to estimates (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1279`).
- Streaming failures are currently converted to assistant text and `[DONE]` (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1514`; Foundation equivalent at `Sources/AFMServer/Controllers/ChatCompletionsController.swift:490`). GuideLLM can misclassify these as successful.
- Foundation usage remains heuristic (`Sources/AFMServer/Controllers/ChatCompletionsController.swift:192`, line 304), so Foundation cannot be claimed as GuideLLM-qualified.

### Discovery and external clients

- `/v1/models` is inline in `Server` (`Sources/AFMServer/Server.swift:321`). MLX is first, but its `created` value changes per request (`Sources/AFMServer/Server.swift:333`). Foundation also uses current time and appends discovery results without explicit sorting (`Sources/AFMServer/Server.swift:356`, line 369).
- Existing tests cover embedding-model presence rather than repeat-response equality or first-model suitability (`Tests/MacLocalAPITests/EmbeddingsControllerTests.swift:231`).
- Pinned GuideLLM maps `completions` and `chat_completions` to `/v1/completions` and `/v1/chat/completions`, and selects the first `/v1/models` ID if no model is supplied (`src/guidellm/backends/openai/http.py:45`, line 322 in the pinned checkout).
- Its streaming payloads request usage, send compatibility extras, and expect token-bearing SSE, usage, and exact `[DONE]` termination (`src/guidellm/backends/openai/request_handlers.py:485`, line 620, line 1010, line 1153). Pinned GuideLLM sends one prompt string for legacy completions.
- Pinned GuideLLM documents synchronous, throughput, concurrent, constant-rate, Poisson, and sweep profiles plus JSON/CSV/HTML outputs (`docs/getting-started/benchmark.md:105`, line 295).
- Pinned Playground always scrapes `<base-url>/metrics` and discards names without the `vllm:` prefix (`vllm_playground/app.py:622`, line 313). It has no separate metrics-path setting.
- The existing local harness covers discovery, chat, and optional `vllm bench serve`, but not raw completions, GuideLLM, or Prometheus parsing (`Scripts/feature-codex-optimize-api/test-openai-compat-evals.py:1`, line 120, line 302). `Scripts/test-assertions.sh` has reusable HTTP/SSE helpers at line 110.

## Approved representation decision

Append `vllm:*` compatibility families to the existing `GET /metrics` representation. Keep every existing AFM HELP/TYPE/sample family and meaning unchanged.

This is additive for Prometheus and is the only representation that works with the pinned unmodified Playground. Content negotiation is unsafe because neither Playground nor normal Prometheus scrapers request a vLLM-specific media profile. A separate endpoint cannot be the compatibility surface while Playground hardcodes `/metrics`. No content-negotiated or separate vLLM endpoint is planned.

AFM and vLLM rendering consume one immutable provider-neutral snapshot. There is no second registry and no scrape-time mutation except taking the atomic live-gauge snapshot.

## Selected telemetry architecture

This plan selects **cross-runtime qualification** rather than keeping the concrete collector in AFMKitMLX.

- `AFMKitCore` owns only `Sendable` event/value protocols and immutable observation types. It has no counters, Prometheus names, HTTP concepts, or dependency on a provider.
- `AFMKitServices` owns the provider-neutral process collector, locks/rolling windows, immutable snapshot, clock abstraction, and one-terminal enforcement. Proposed new files are `Sources/AFMKitServices/Telemetry/InferenceTelemetryCollector.swift` and `Sources/AFMKitServices/Telemetry/InferenceMetricsSnapshot.swift`.
- AFMKitMLX and AFMKitDwarfStar receive an `any AFMInferenceTelemetryObserving` during construction. They never import AFMKitServices; the CLI/server composition root injects the collector through the AFMKitCore protocol.
- AFMServer receives the same collector through a snapshot-source protocol. It owns only HTTP-side failure classification and deterministic Prometheus exposition.
- `Sources/AFMKitMLX/Models/StatsAggregator.swift` becomes a deprecated forwarding facade/type alias if public source compatibility requires `StatsAggregator.shared`. New provider and server code must not use that global facade.
- `Sources/AFMCLI/main.swift` creates exactly one collector per server process and injects it into provider/runtime and `Server`. `Package.swift` is updated only as needed for AFMKitCore protocol and AFMKitServices collector dependencies.
- Foundation is not adapted to the provider event contract in issue 192 and is not part of the qualified matrix below.

The observer methods are synchronous, nonblocking, `Sendable`, cancellation-safe, and limited to a short lock/atomic enqueue. They must never await, call provider code, or re-enter a scheduler actor. Event values include an opaque collector-issued request token so terminal deduplication does not require request IDs as Prometheus labels.

### Event ownership and one-terminal rule

A request becomes **accepted** only when a provider validates its runtime request, reserves/adopts its request token, and admits it to the provider's runnable/waiting queue. HTTP decode/auth/client-validation/capacity failures before that point are rejected, not accepted. Every accepted request gets exactly one provider-owned terminal event.

| Event/state | Sole owner | Recording rule |
| --- | --- | --- |
| HTTP body decode, auth, endpoint/model/client-field validation failure | AFMServer | Record one bounded rejection status; no provider request token and no accepted/completed event. |
| Capacity rejection before provider admission | AFMServer | Record `capacity`; no accepted request. Waiting after provider admission is not rejection. |
| Provider admission | Runtime/provider | Allocate telemetry token and emit `accepted` exactly once after successful queue admission. |
| Waiting/running gauges | Runtime scheduler | Publish an atomic snapshot; accepted queued work is waiting, scheduled work is running. |
| Full prompt token count | Runtime/provider tokenizer | Record exact model-input token IDs once per accepted request, including reusable cached prefix. |
| Computed prompt token count | Runtime/provider prefill | Record only prompt token positions actually computed after prefix reuse. |
| Output tokens and monotonic timestamps | Runtime/provider generation loop | Record actual generated token events; server text chunk boundaries are irrelevant. |
| Prefix queried/hit token counts | Runtime/provider cache owner | Record eligible queried tokens and reused tokens once, at cache resolution. |
| Logical KV positions | Runtime scheduler | Publish atomic active-position sum; waiting and retained radix entries contribute zero. |
| Speculative draft round/draft/accepted token counts | Runtime speculative implementation | Record one observation per draft round; collector derives rejected totals and rate. |
| Successful/failed runtime terminal | Runtime/provider | Atomically claim the request token's terminal state once, with bounded finish/failure reason and exact usage. |
| Client disconnect/cancel after acceptance | AFMServer requests; provider owns result | Server signals cancellation. Provider removes runtime/KV state and emits the sole `cancelled` terminal. Server must not emit a second terminal metric. |
| Wire HTTP status, JSON errors, SSE framing | AFMServer | Map provider result/error to wire state only; never increment accepted-runtime terminal counters. |

The collector rejects duplicate terminal events for the same telemetry token and tests assert balanced accepted/terminal counts for every exit path.

## Metric contract and cardinality

### Full versus computed prompt tokens

The snapshot stores two distinct monotonic values:

- `fullPromptTokensTotal`: exact tokenizer input length before prefix reuse, including cached tokens. This renders as pinned vLLM `vllm:prompt_tokens_total` and feeds the full request-size histogram.
- `computedPromptTokensTotal`: prompt positions actually prefetched/computed after cache reuse. This continues to render as the existing `afm:prompt_tokens_total` without changing AFM semantics and feeds the 10-second prompt-throughput gauge.

For a cache-hit request with full prompt `F`, reused prefix `H`, and computed suffix `C = F - H`, tests must prove: vLLM prompt total increases by `F`; AFM computed prompt total and prompt-throughput window increase by `C`; prefix queries increase by `F`; prefix hits increase by `H`; misses increase by `C`.

### Canonical and AFM-extension families

| Rendered metric | Source | Contract classification |
| --- | --- | --- |
| `vllm:num_requests_running` | provider live gauge | Pinned vLLM/Playground family. |
| `vllm:num_requests_waiting` | provider live gauge | Pinned vLLM/Playground family. |
| `vllm:num_preemptions_total` | explicit provider preemption events | Pinned family; cancellation never increments it. |
| `vllm:request_success_total{finished_reason}` | sole provider terminal | Pinned family with bounded `stop`, `length`, `tool_calls`, `abort`, `error`. |
| `vllm:prompt_tokens_total` | `fullPromptTokensTotal` | Pinned family; includes cached tokens. It is not an AFM prompt-total alias. |
| `vllm:generation_tokens_total` | exact generated tokens | Pinned family; value agrees with existing AFM generation total. |
| `vllm:request_prompt_tokens` | full prompt size | Pinned histogram. |
| `vllm:request_generation_tokens` | exact output size | Pinned histogram. |
| `vllm:avg_prompt_throughput_toks_per_s` | rolling computed prompt tokens | Exact legacy Playground key; documented 10-second monotonic window. |
| `vllm:avg_generation_throughput_toks_per_s` | rolling generated tokens | Exact legacy Playground key; same window. |
| `vllm:time_to_first_token_seconds` | accepted/queued to first token | Pinned histogram. |
| `vllm:request_time_per_output_token_seconds` | decode duration / output intervals | Pinned TPOT histogram. |
| `vllm:inter_token_latency_seconds` | adjacent output-token timestamps | Pinned ITL histogram, not a TPOT alias. |
| `vllm:e2e_request_latency_seconds` | accepted to terminal | Pinned histogram. |
| `vllm:kv_cache_usage_perc` | active logical positions / logical capacity | Pinned family with AFM approximation documented below. |
| `vllm:prefix_cache_queries_total` | eligible full prompt tokens queried | Pinned/Playground family. |
| `vllm:prefix_cache_hits_total` | reused prefix tokens | Pinned/Playground family. |
| `vllm:prefix_cache_hit_rate` | hits / queries | Exact Playground registry key; zero at zero queries. |
| `vllm:spec_decode_num_draft_tokens_total` | speculative draft tokens | Exact Playground counter sample. |
| `vllm:spec_decode_num_accepted_tokens_total` | accepted draft tokens | Exact Playground counter sample. |
| `vllm:spec_decode_num_drafts_total` | draft rounds | Exact Playground counter sample. |
| `vllm:spec_decode_acceptance_rate` | accepted / draft tokens | **Exact pinned Playground key**; zero when draft tokens are zero. |
| `afm:request_failures_total{status}` | server rejection plus provider failure classification | AFM issue-192 addition, not represented as upstream vLLM. Bounded statuses only. |
| `afm:prefix_cache_missed_tokens_total` | queries minus hits | AFM issue-192 addition, not upstream vLLM. Existing request-event radix counters remain unchanged. |
| `afm:spec_decode_num_rejected_tokens_total` | draft minus accepted | AFM issue-192 addition, not upstream vLLM. |
| `afm:request_throughput_requests_per_s` | rolling terminal events | AFM issue-192 addition, not upstream vLLM. |

Create pinned fixtures such as `Tests/MacLocalAPITests/Fixtures/vllm-9633933-metrics.prom` and `Tests/MacLocalAPITests/Fixtures/vllm-playground-7627622-registry.json`. Renderer tests lock canonical sample names, HELP/TYPE names, label sets, and bucket boundaries to those fixtures. Prometheus counters retain `_total` on rendered samples. AFM additions are tested separately and never described as upstream families.

Append only the upper histogram bounds added by pinned upstream. Do not remove old AFM buckets or change their cumulative values.

### Cardinality policy

- Canonical vLLM families use only process-derived `model_name` and fixed `engine="0"`, matching pinned upstream.
- `finished_reason` and AFM `status` use closed enums. Unknown values collapse to `unknown`; arbitrary strings are never labels.
- Do not label by request ID/token, API key, user, prompt, endpoint, sampling values, HTTP code/path, cache key, tool name, or error type/message.
- Runtime identity, if exposed, is one bounded AFM info gauge, not a label copied onto every series.
- Expected bounded series render stable zero samples so schema does not change with traffic.

### Logical KV utilization

`vllm:kv_cache_usage_perc` is the atomic scheduler snapshot:

```text
sum(active request logical KV positions) / (contextWindow * maxConcurrent)
```

- An active logical KV position is a prompt or generated sequence position currently addressable in a running request's model KV state. Reused prefix positions count while attached to an active request because they are logically addressable by that request.
- Waiting requests have no KV allocation and contribute zero.
- Completed/cancelled requests are removed atomically before the next snapshot.
- Retained radix/prefix snapshots not attached to active requests are excluded and remain visible only through existing AFM radix metrics.
- The value is clamped to `[0,1]`; zero capacity/configuration yields zero and a diagnostic rather than division by zero.
- HELP/docs call this logical slot occupancy, an AFM approximation of vLLM KV usage, and explicitly distinguish it from Metal working-set pressure.

Tests cover idle zero, admission before allocation, prefill/decode growth, concurrent active sums, completion, cancellation, retained-prefix exclusion, and the invariant `0 <= usage <= 1`.

## Provider-neutral raw-prompt generation contract

Add wire-independent types in `Sources/AFMKitCore/Providers/AFMRawTextGeneration.swift`:

- `AFMRawTextGenerationRequest`: one `prompt: String`, selected model, maximum output tokens, stop strings, supported sampling values, and seed. It contains no messages, roles, instructions, or HTTP/SSE fields.
- `AFMRawTextGenerationEvent`: text/token delta with provider monotonic timestamp, followed by exactly one terminal result containing bounded finish reason and exact `promptTokens`, `completionTokens`, and `totalTokens`; or one typed provider failure.
- `AFMRawTextGenerating`: capability plus a generation method returning the provider event stream. The same stream is collected for non-streaming responses, preventing separate usage/finish implementations.

Provider requirements:

1. Tokenize the raw prompt and construct the runtime equivalent of `UserInput(prompt:)` directly.
2. Do not insert system/default instructions, synthesize a user/assistant turn, invoke a chat template, or use `buildUserInput`'s message path.
3. Count the exact model input token IDs as full prompt usage even when a prefix is cached; report computed prompt tokens separately to telemetry.
4. Count actual generated token IDs, preserve stop/length semantics, and emit one provider terminal event.
5. Advertise the capability only when these semantics and exact usage are implemented.

AFMOpenAICompat owns `CompletionRequest`, response, chunk, usage, and error wire DTOs. AFMServer owns DTO validation and mapping provider events to OpenAI JSON/SSE. `/v1/completions` POST/OPTIONS is registered only when the selected runtime supplies `AFMRawTextGenerating`; unsupported runtimes do not silently route through chat.

### Prompt-array behavior

The request DTO decodes `prompt` as either string or array so it can return a stable protocol error. Issue 192 supports only one prompt string, matching pinned GuideLLM. Any array, including a one-element array, is rejected before provider admission with:

- HTTP `400`.
- `type: "invalid_request_error"`.
- `code: "unsupported_prompt_array"`.
- `param: "prompt"` when the existing error DTO supports it; otherwise include `prompt` in the stable message and add `param` as part of the DTO work.

No partial choices, usage, accepted telemetry, or SSE headers are emitted for an array.

## Exact runtime qualification matrix

| Runtime/path | vLLM metrics | Chat GuideLLM | Raw `/v1/completions` | Usage claim | Issue-192 result |
| --- | --- | --- | --- | --- | --- |
| MLX batch and serial | Required | Required, stream and non-stream | Required through `AFMRawTextGenerating` | Exact provider token IDs only; estimation disqualifies a run | Fully qualified baseline. |
| MLX MTP/EAGLE3 | Required including spec metrics | Required | Required through same raw contract | Exact provider token IDs | Qualified after the same matrix passes with each mode. |
| DwarfStar and DSpARK | Required through neutral observer, including DSpARK spec metrics | Required | Required through `AFMRawTextGenerating` | Exact provider token IDs | Cross-runtime qualification is required; do not register raw route or claim success until contract passes. |
| Foundation Models | Not qualified for vLLM provider metrics | Existing chat remains supported but not GuideLLM-qualified | Not registered | Current estimates are prohibited for qualification | Explicitly out of the issue-192 compatibility claim until native raw generation and usage exist. |
| Proxy/discovered remote backends | Existing proxy behavior only | Not qualified by this issue | Not registered unless a future provider implements the contract | No local accuracy claim | Out of scope. |

`/v1/models` remains available for every server, but GuideLLM documentation lists only a runtime/model row that passed this matrix. The loaded generative model remains first, IDs remain unchanged, its creation epoch is stable, and remaining entries are sorted.

## Wire success and error state machines

Chat and legacy completion streams are separate machines; shared framing helpers must not erase their distinct choice payloads.

### Chat SSE (`chat.completion.chunk`)

1. `validated`: request is decoded/validated before response headers and provider admission succeeds.
2. `opened`: emit the assistant role delta at most once if retained by current compatibility behavior.
3. `streaming`: emit zero or more `choices[0].delta.content`/reasoning/tool deltas with `finish_reason: null`.
4. `finished`: emit exactly one choice with empty delta and non-null `finish_reason`.
5. `usage` (only when requested): emit exactly one final event with `choices: []` and exact usage.
6. `done`: emit exactly one `data: [DONE]`, then close.

### Legacy SSE (`text_completion`)

1. `validated`: decode/validate the single raw prompt before headers; provider admission succeeds.
2. `streaming`: emit zero or more choices containing `text` and `index: 0`; never emit a role or `delta` object.
3. `finished`: emit exactly one choice with `text: ""`, `index: 0`, and non-null `finish_reason`.
4. `usage` (only when requested): emit exactly one event with `choices: []` and exact usage.
5. `done`: emit exactly one `data: [DONE]`, then close.

For both machines, a pre-header failure is a non-2xx OpenAI JSON error. A failure after headers is exactly one `data: {"error": ...}` event followed by connection close, with no success finish, usage event, assistant/text substitution, or `[DONE]`. The pinned GuideLLM parser itself is run against success and post-header-error fixtures and a live forced failure; qualification requires that it classify the latter as an error. If it does not, implementation remains unqualified rather than changing the declared state machine silently.

Non-streaming chat and raw completions use their respective normal object shapes and the provider's exact terminal usage/reason. There is no estimate fallback on a qualified path.

## Implementation sequence and exact likely files

1. Add event/value protocols and raw-prompt contracts in new `Sources/AFMKitCore/Telemetry/InferenceTelemetry.swift` and `Sources/AFMKitCore/Providers/AFMRawTextGeneration.swift`.
2. Add the collector/snapshot/clock in new `Sources/AFMKitServices/Telemetry/InferenceTelemetryCollector.swift` and `InferenceMetricsSnapshot.swift`; update `Package.swift` target dependencies.
3. Convert `Sources/AFMKitMLX/Models/StatsAggregator.swift` to a compatibility facade and instrument `BatchScheduler.swift`, `MLXModelService.swift`, `AFMMLXRuntime.swift`, and `AFMMLXProvider.swift` through injected protocols.
4. Instrument `Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift` and its model/runtime construction. Expose MTP/EAGLE3/DSpARK observations from AFM-owned sources. Patch only `Scripts/patches/Qwen3_5MoE.swift`, `DeepseekV4.swift`, or `Gemma4Eagle3.swift` if direct implementation proves the callback unavailable; never edit vendor sources directly.
5. Construct/inject one collector in `Sources/AFMCLI/main.swift` and inject its snapshot source into `Server`.
6. Refactor `Sources/AFMServer/Controllers/MetricsController.swift` into unchanged AFM-native rendering plus pinned vLLM rendering over one snapshot. Update `Scripts/grafana/README.md` and `Scripts/grafana/UPSTREAM.md`.
7. Add raw completion DTOs under `Sources/AFMOpenAICompat/CompletionRequest.swift` and `CompletionResponse.swift`; add `Sources/AFMServer/Controllers/CompletionsController.swift`; register capability-gated routes and update `OpenAPIController.swift` and `Server.swift` route output.
8. Correct chat and raw SSE framing/error behavior in `MLXChatCompletionsController.swift`, `ChatCompletionsController.swift` only where existing unqualified behavior must not be shared, and common response helpers/DTOs.
9. Extract deterministic model-list construction from `Sources/AFMServer/Server.swift` into a testable helper.
10. Add `Scripts/test-vllm-guidellm-compat.py`, focused deterministic coverage in `Scripts/test-assertions.sh`, and `docs/guidellm.md`. Heavy model/GuideLLM runs remain opt-in.

Release packaging/workflow cleanup is explicitly out of scope. Do not modify `.github/workflows/release.yml` or repair/remove its stale script references under issue 192.

## Automated test matrix

| Layer | Test | Required assertion |
| --- | --- | --- |
| Collector | Lifecycle/terminal dedupe | Accepted/terminal balances; duplicate terminal ignored/reported; cancellation race is exactly once. |
| Collector | Nonblocking observer | Concurrent event calls do not await or re-enter a scheduler; deterministic stress test has no lost counts. |
| Collector | Full/computed/cache split | On cache hit `F/H/C`, vLLM full `+F`, AFM computed and throughput `+C`, query `+F`, hit `+H`, miss `+C`. |
| Collector | Rolling throughput | Injected monotonic clock covers empty, partial 10-second window, expiry, and concurrency. |
| Collector | Speculation | Draft/accepted/derived rejected arithmetic and exact `vllm:spec_decode_acceptance_rate`; zero denominator is zero. |
| Collector | TTFT/TPOT/ITL | Known token timestamps produce distinct values; zero/one-token outputs fabricate no intervals. |
| Collector | KV logical occupancy | Admission/decode/completion/cancel bounds and retained-prefix exclusion. |
| Renderer | AFM preservation golden | Every pre-issue AFM HELP/TYPE/sample family and meaning remains unchanged. |
| Renderer | Pinned vLLM fixture | Exact canonical names, `_total` samples, HELP/TYPE, labels, and buckets match pinned fixtures. |
| Renderer | Playground registry | Every pinned registry key exists, especially `vllm:spec_decode_acceptance_rate`; AFM additions are not classified as upstream. |
| Renderer | Prometheus parser/linter | `promtool check metrics` and `prometheus_client.parser` accept output; no duplicate/conflicting metadata. |
| Renderer | Shared-source parity | Only true aliases (running/waiting/generated/latency/finish) agree; full and computed prompt totals intentionally differ on hits. |
| Route | `/metrics` | 200, existing content type/CORS, both namespaces, deterministic order, no scrape-time counter mutation. |
| Raw provider | Template bypass | Exact raw prompt reaches `UserInput(prompt:)`; no system instruction, messages, chat template, or role tokens. |
| DTO/controller | Prompt array | Every array shape returns stable pre-header 400 and no provider admission. |
| Protocol | Non-stream success | Chat and text-completion shapes, finish reasons, IDs/model, exact usage. |
| Protocol | Chat SSE | Delta payloads, one finish, optional usage-only, one `[DONE]` in exact order. |
| Protocol | Legacy SSE | Text payloads/no role/no delta, one finish, optional usage-only, one `[DONE]`. |
| Protocol | SSE post-header error | One OpenAI error event, no finish/usage/text substitution/`[DONE]`; pinned GuideLLM counts failure. |
| Protocol | Usage disabled | Suppresses only usage event and preserves successful finish/`[DONE]`. |
| Discovery | Determinism | Consecutive responses byte-stable; loaded generative model first; tail sorted. |
| Runtime matrix | Capability/route gating | MLX and DwarfStar expose raw route only with exact contract; Foundation/proxy do not. |
| Runtime matrix | Lifecycle paths | MLX batch/serial/MTP/EAGLE3 and DwarfStar/DSpARK each balance events and exact usage. |

Likely tests:

- New `Tests/MacLocalAPITests/InferenceTelemetryCollectorTests.swift`.
- New `Tests/MacLocalAPITests/MetricsControllerTests.swift` and pinned fixture directory.
- New `Tests/MacLocalAPITests/RawTextGenerationContractTests.swift`.
- New `Tests/MacLocalAPITests/LegacyCompletionsControllerTests.swift`.
- New `Tests/MacLocalAPITests/ModelDiscoveryDeterminismTests.swift`.
- Extend `StreamingUsageChunkTests.swift` and `MLXChatCompletionsControllerStreamingTests.swift`.
- Add DwarfStar provider conformance tests; retain Foundation telemetry tests only to prove it remains unqualified/route-gated.

All SwiftPM builds/tests run through `Scripts/swiftpm-reliable.sh` per repository instructions.

## Live qualification matrix

Run every applicable row separately for MLX batch, MLX serial, MTP, EAGLE3, DwarfStar, and DSpARK. A runtime is listed as qualified only when all required rows pass without estimated usage.

| Scenario | Traffic/profile | Pass criteria/artifacts |
| --- | --- | --- |
| Idle metrics | Scrape before traffic | Pinned parser/linter passes; canonical keys and AFM metrics present; finite zeros. |
| Full/computed prefix | Repeat a known-token long prefix | Full/computed/query/hit/miss deltas match tokenizer/cache evidence exactly. |
| Logical KV | Queue, admit, decode, cancel, complete | Gauge follows active positions/capacity, excludes waiting/retained radix, remains `[0,1]`. |
| Concurrent metrics | More streams than capacity | Running reaches capacity, waiting becomes nonzero, lifecycle balances, throughput nonzero. |
| Speculation | Each supported speculative mode | Exact draft/accepted arithmetic and canonical acceptance-rate key; non-spec remains zero. |
| Cancellation/failure | Queued/active cancel and forced provider error | Sole terminal owner, bounded failure count, no preemption increment for cancel. |
| Unmodified Playground | Point pinned checkout at server base URL | Hardcoded `/metrics` populates required panels without patch/config fork. |
| GuideLLM synchronous | Chat/raw, stream false/true | Zero protocol errors, exact usage and finish semantics. |
| GuideLLM throughput | Chat/raw throughput profile | Zero errors and nonzero request/token throughput. |
| GuideLLM concurrent | At least two streams | JSON/CSV/HTML include nonzero TTFT, ITL/TPOT, E2E, rates, input/output counts. |
| GuideLLM constant | Below/above capacity | Below-capacity zero errors; overload bounded and represented as failure, never hang/success text. |
| GuideLLM Poisson | Seeded arrival profile | No parser errors; nonzero latency/throughput artifacts. |
| GuideLLM sweep | Small rate/concurrency sweep | Every point completes and model discovery does not drift. |
| SSE forced error | Both endpoint parsers | Pinned GuideLLM counts post-header error as failure; no successful terminal. |
| Runtime gating | Foundation and proxy start | No raw route/GuideLLM accuracy claim; existing chat behavior remains available. |

The harness captures server config/version, runtime/model, upstream commits, exact commands, raw `/metrics`, GuideLLM JSON/CSV/HTML, and a pass/fail manifest. Generated artifacts are ignored and not committed.

## Compatibility risks and mitigations

- Full and computed prompt tokens are deliberately separate; golden cache-hit tests prevent accidental re-aliasing.
- Canonical pinned names are fixture-controlled. AFM additions use `afm:*` and are documented as non-upstream.
- Provider events and server rejection events have disjoint ownership; collector tokens enforce one terminal.
- Raw completions bypass chat templates by contract and test, not by controller convention.
- Chat and text SSE have separate payload/state tests; failures cannot become normal text.
- Logical KV occupancy is an approximation with explicit numerator/denominator and excludes retained cache state.
- Foundation remains unqualified until it has native raw generation and exact usage; estimates are never emitted on a qualified path.
- External GuideLLM/Playground revisions are pinned and recorded to contain CLI/parser drift.
- Existing AFM metric names, meanings, buckets, and Grafana behavior are preserved; upstream bucket updates are additive.

## Architecture review verdict and resolution trace

Durable gate: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW.md`, dated 2026-08-17. Verdict for issue 192: **REQUEST CHANGES**. Implementation remains blocked until this revision is re-gated.

| Reviewer requirement | Resolution in this revision |
| --- | --- |
| 1. Correct full vs computed prompt tokens | Added distinct full and computed counters, corrected `vllm:prompt_tokens_total`, and specified exact `F/H/C` cache-hit assertions. |
| 2. Exact speculative key and pinned contract | Replaced the wrong key with `vllm:spec_decode_acceptance_rate`; added pinned name/HELP/TYPE/label/bucket fixtures and classified misses/rejected/failure metrics as AFM additions. |
| 3. Resolve cross-runtime collector ownership | Selected AFMKitCore protocols plus AFMKitServices collector/snapshot, injected into providers and AFMServer; MLX singleton retained only as compatibility facade. |
| 4. Add event ownership table | Defined acceptance, sole owners for every event, nonblocking observer constraints, cancellation behavior, and one-terminal enforcement. |
| 5. Define true raw-prompt contract | Added provider-neutral `AFMRawTextGenerating` using direct raw `UserInput(prompt:)`, exact usage, shared stream/non-stream provider events, and capability-gated route registration. |
| 6. Do not claim Foundation accuracy | Added exact runtime matrix: MLX and DwarfStar contracts are required; Foundation and proxies are explicitly unqualified and receive no silent raw route. |
| 7. Unambiguous prompt arrays | Chose stable rejection of every array shape with HTTP 400/code `unsupported_prompt_array`, before admission/SSE. |
| 8. Separate SSE state machines | Defined distinct chat `delta` and legacy `text` machines, terminal/usage/`[DONE]` ordering, and one post-header error event with no success terminal. |
| 9. Precise KV utilization | Adopted active logical positions divided by `contextWindow * maxConcurrent`, atomic snapshot, no waiting/retained radix contribution, and full lifecycle tests. |
| 10. Exclude release cleanup | Explicitly prohibits release workflow/package cleanup under issue 192. |
| 11. Retain and extend tests | Retained parser, AFM golden, discovery, usage, SSE, concurrency, GuideLLM, and Playground tests; added prompt split, registry, raw bypass, KV, ownership, and runtime-gating coverage. |

No architectural questions from the first checkpoint remain open: the gate supplied the KV definition, approved the 10-second window and additive `/metrics`, and this revision selects cross-runtime ownership and exact runtime/route behavior. Implementation waits for reviewer approval.
