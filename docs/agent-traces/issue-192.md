# Issue 192 Phase A: vLLM Metrics and GuideLLM Interoperability Plan

Status: planning checkpoint only. No feature code is included in this phase. Implementation must wait for reviewer approval.

Issue: <https://github.com/scouzi1966/maclocal-api/issues/192>

## Scope and acceptance target

Issue 192 has two coupled compatibility goals:

1. Expose metrics that an unmodified vLLM Playground can scrape while retaining every existing `afm:*` metric and its meaning.
2. Make both OpenAI completion endpoints usable by GuideLLM in streaming and non-streaming modes, with deterministic model discovery, correct terminal events, and trustworthy usage.

The implementation should not claim general vLLM/OpenAI conformance. It should qualify the exact metric families and GuideLLM profiles in this plan against pinned upstream revisions:

- GuideLLM `97b3077c05a367599112fd7080082c2d32c14b7e` (inspected 2026-08-17).
- vLLM Playground `76276229092455f9ef66748731e4a615f4d80720` (inspected 2026-08-17).
- vLLM `9633933dd81228fbcae07969f20881ad0b7cb766` (inspected 2026-08-17).

## Current-state evidence

### Metrics and telemetry ownership

- `MetricsController` registers only `GET /metrics`, returns Prometheus text 0.0.4, and renders one `StatsAggregator` snapshot (`Sources/AFMServer/Controllers/MetricsController.swift:22`, `Sources/AFMServer/Controllers/MetricsController.swift:26`, `Sources/AFMServer/Controllers/MetricsController.swift:64`).
- The current exposition contains only `afm:*` series. Running/waiting/cache gauges are at `Sources/AFMServer/Controllers/MetricsController.swift:82`; token/request/cache counters at line 128; finish-reason counters at line 160; latency and token histograms at line 173; process metadata at line 252.
- `StatsAggregator` is a process singleton owned by `AFMKitMLX`; AFMServer is deliberately the HTTP/presentation owner (`Sources/AFMKitMLX/Models/StatsAggregator.swift:18`, `Sources/AFMKitMLX/Models/StatsAggregator.swift:23`, `Sources/AFMKitMLX/Models/StatsAggregator.swift:37`). Keep that ownership split.
- Its current counters cover prompt/generated tokens, accepted/completed requests, request-level radix hit/miss events, and `requestSuccessByReason` (`Sources/AFMKitMLX/Models/StatsAggregator.swift:126`). It has no failure-status, preemption, token-level prefix-cache, speculative, per-token ITL, or rolling-throughput state.
- `observeRequest` derives E2E, queue, inference, prefill, TTFT, and one average TPOT observation per request (`Sources/AFMKitMLX/Models/StatsAggregator.swift:304`, `Sources/AFMKitMLX/Models/StatsAggregator.swift:340`). It cannot reconstruct an ITL distribution after completion.
- The histogram provenance is an older vLLM snapshot (`Sources/AFMKitMLX/Models/StatsAggregator.swift:47`). Current upstream has additional upper buckets. Those can be appended without removing or reinterpreting existing AFM buckets.
- `BatchScheduler` owns live running/waiting state and registers readers at `Sources/AFMKitMLX/Models/BatchScheduler.swift:590`; it starts accepted work at line 654 and records completion at lines 1979-1992.
- Batched prefix caching adds only uncached suffix tokens to `promptTokensTotal` and increments hit/miss once per request (`Sources/AFMKitMLX/Models/BatchScheduler.swift:1359`). This is not vLLM's token-level prefix-query/hit semantics.
- Serial MLX paths update the same singleton (`Sources/AFMKitMLX/Models/MLXModelService.swift:2100`, `Sources/AFMKitMLX/Models/MLXModelService.swift:2433`, `Sources/AFMKitMLX/Models/MLXModelService.swift:3821`). Several speculative fast paths terminate counters without recording the complete token/latency observation, for example `Sources/AFMKitMLX/Models/MLXModelService.swift:2177` and line 2254.
- `afm:gpu_cache_usage_perc` is Metal working-set pressure, including weights and intermediates, not KV occupancy (`Sources/AFMServer/Controllers/MetricsController.swift:103`). It must not be relabeled as `vllm:kv_cache_usage_perc`.
- DwarfStar already tracks accepted speculative tokens internally (`Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift:647`) but cannot import `AFMKitMLX`: `AFMKitDwarfStar` and `AFMKitMLX` are sibling targets depending on `AFMKitCore` (`Package.swift:166`, `Package.swift:188`).
- No existing test directly exercises `StatsAggregator`, `MetricsController`, Prometheus parsing, or AFM/vLLM parity.

### Routes and OpenAI protocol behavior

- Both runtime controllers register only `POST /v1/chat/completions`; no `/v1/completions` route or legacy completion DTO exists (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:90`, `Sources/AFMServer/Controllers/ChatCompletionsController.swift:40`, `Sources/AFMOpenAICompat/OpenAIRequest.swift:3`).
- Chat requests accept both `max_tokens` and `max_completion_tokens`; current precedence is `max_tokens` then `max_completion_tokens` (`Sources/AFMOpenAICompat/OpenAIRequest.swift:8`, `Sources/AFMOpenAICompat/OpenAIRequest.swift:66`). Unknown GuideLLM extensions such as `ignore_eos` and `continuous_usage_stats` are ignored by Codable.
- Streaming usage defaults to enabled when `stream_options` is absent (`Sources/AFMOpenAICompat/OpenAIRequest.swift:60`) and that non-OpenAI default is intentionally locked by `Tests/MacLocalAPITests/AgentFriendlyTier1Tests.swift:52`. Do not change it incidentally.
- MLX emits a terminal finish chunk, optionally an empty-choice usage chunk, then `[DONE]` (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1453`). MLX normally receives real token counts, but falls back to estimation when final runtime counts are unavailable (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1279`).
- Streaming failures are currently converted to normal text content followed by `[DONE]` (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1514`; Foundation equivalent at `Sources/AFMServer/Controllers/ChatCompletionsController.swift:490`). GuideLLM can therefore count a failed generation as successful.
- MLX derives `tool_calls`, `stop`, and `length` terminal reasons in `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:2140`. Foundation completion and prompt token usage is heuristic (`Sources/AFMServer/Controllers/ChatCompletionsController.swift:192`, `Sources/AFMServer/Controllers/ChatCompletionsController.swift:304`).
- Existing streaming tests check selected finish reasons and `[DONE]`, but not the complete ordering/usage/error contract (`Tests/MacLocalAPITests/MLXChatCompletionsControllerStreamingTests.swift:46`; `Tests/MacLocalAPITests/StreamingUsageChunkTests.swift:97`).

### Model discovery

- `/v1/models` is defined inline in `Server` (`Sources/AFMServer/Server.swift:321`). The loaded MLX model is first, which matches GuideLLM's first-model fallback, but its `created` value changes on every request (`Sources/AFMServer/Server.swift:333`).
- Foundation also uses the current timestamp and appends backend-discovery results without an explicit sort (`Sources/AFMServer/Server.swift:356`, `Sources/AFMServer/Server.swift:369`). Full responses are therefore not deterministic.
- Existing coverage checks embedding-model presence, not repeat-response equality or first-model suitability (`Tests/MacLocalAPITests/EmbeddingsControllerTests.swift:231`).

### External clients and harnesses

- GuideLLM maps its `completions` and `chat_completions` request formats to `/v1/completions` and `/v1/chat/completions` respectively (`src/guidellm/backends/openai/http.py:45` in the pinned checkout). If no model is supplied, it selects the first ID from `/v1/models` (`http.py:322`).
- GuideLLM streaming sends `stream_options.include_usage=true`, `continuous_usage_stats=true`, `stop=null`, and `ignore_eos=true`. Legacy completions send `max_tokens`; chat sends `max_completion_tokens` (`src/guidellm/backends/openai/request_handlers.py:485`, line 1010). Its stream parser expects token-bearing SSE chunks, a usage object, and exact `[DONE]` termination (`request_handlers.py:620`, line 1153).
- The pinned GuideLLM documents synchronous, throughput, concurrent, constant-rate, Poisson, and sweep profiles plus JSON/CSV/HTML outputs (`docs/getting-started/benchmark.md:105`, line 295).
- vLLM Playground always scrapes `<base-url>/metrics` (`vllm_playground/app.py:622`) and drops metric names not prefixed `vllm:` (`app.py:313`). It has no independent metrics-path setting in the inspected revision.
- Its registry expects, among others, running/waiting, KV usage, prefix counters/rate, preemptions, prompt/generation throughput, speculative counts/rate, E2E, TTFT, ITL, and TPOT (`vllm_playground/static/js/modules/metrics-registry.js`).
- The existing local qualification script covers model discovery, chat streaming/non-streaming, and optional `vllm bench serve`, but not legacy completions, GuideLLM, or Prometheus parsing (`Scripts/feature-codex-optimize-api/test-openai-compat-evals.py:1`, line 120, line 302).
- `Scripts/test-assertions.sh` has reusable HTTP/SSE helpers (`Scripts/test-assertions.sh:110`) and streaming parity coverage around line 4349, but no metrics compatibility section.
- The release workflow currently attempts to package absent root scripts `test-streaming.sh` and `test-metrics.sh` (`.github/workflows/release.yml:62`). The new qualification harness must not depend on those stale references.

## Metrics compatibility decision

### Chosen representation: additive dual exposition on `/metrics`

Keep all current `afm:*` HELP/TYPE/sample families and append `vllm:*` compatibility families to the same Prometheus response.

This is backward-compatible for Prometheus consumers because adding families does not alter existing AFM series. It also satisfies the unmodified Playground, which hardcodes `/metrics` and ignores `afm:*` names.

Content negotiation is unsafe here: Playground sends no vLLM-specific media profile, Prometheus commonly sends broad `Accept` values, caches/proxies would require correct `Vary`, and a scrape could silently receive the wrong namespace. A separate endpoint is safer than negotiation in isolation because its representation is explicit and cache-stable, but it does not meet the no-fork Playground requirement unless Playground first gains a metrics-path option. Therefore neither mechanism is the primary design. If reviewers require isolation later, add `/metrics/vllm` as an alias in addition to, not instead of, the dual `/metrics` response.

AFM is the source of truth. The vLLM renderer consumes the same immutable snapshot; it does not maintain a second metric registry. Tests must assert AFM/vLLM parity for aliased values.

## Proposed metric mapping and semantics

| vLLM family | AFM source / new observation | Type and policy |
| --- | --- | --- |
| `vllm:num_requests_running` | existing live running reader | Gauge; exact alias of `afm:num_requests_running`. |
| `vllm:num_requests_waiting` | existing live waiting reader | Gauge; exact alias of `afm:num_requests_waiting`. |
| `vllm:num_preemptions_total` | new explicit preemption counter | Counter; emit zero until a scheduler actually evicts/requeues KV state. Cancellation is not preemption. |
| `vllm:request_success_total{finished_reason}` | existing terminal-reason counter | Counter; exact value parity with AFM, including bounded `stop`, `length`, `tool_calls`, `abort`, `error`, `unknown`. Preserve the awkward upstream name rather than rewriting AFM history. |
| `vllm:request_failures_total{status}` | new failure classification | Counter; bounded `client_error`, `capacity`, `cancelled`, `inference`, `internal`. No HTTP code or error text labels. |
| `vllm:prompt_tokens_total` | existing processed-prefill counter | Counter; tokens actually processed after reusable prefix, matching the present AFM HELP text. |
| `vllm:generation_tokens_total` | existing generated-token counter | Counter; exact AFM alias. |
| `vllm:request_prompt_tokens` | existing full prompt-size histogram | Histogram; one observation per accepted request. |
| `vllm:request_generation_tokens` | existing output-size histogram | Histogram; one observation per completed request, including zero-token failures where appropriate. |
| `vllm:request_throughput_requests_per_s` | new rolling completion events | Gauge over a documented 10-second monotonic window. |
| `vllm:avg_prompt_throughput_toks_per_s` | new rolling processed-token events | Gauge over the same window; use the exact legacy name consumed by Playground. |
| `vllm:avg_generation_throughput_toks_per_s` | new rolling generated-token events | Gauge over the same window. |
| `vllm:time_to_first_token_seconds` | existing TTFT histogram | Histogram; arrival/queue timestamp to first generated token. |
| `vllm:request_time_per_output_token_seconds` | existing per-request TPOT | Histogram; decode duration divided by generated intervals (`tokens - 1`). |
| `vllm:inter_token_latency_seconds` | new token timestamp observations | Histogram; one observation per adjacent output-token pair, not an alias of TPOT. |
| `vllm:e2e_request_latency_seconds` | existing E2E histogram | Histogram; accepted/queued to terminal completion. |
| `vllm:kv_cache_usage_perc` | new logical KV occupancy reader | Gauge in `[0,1]`; never alias Metal working-set pressure. Denominator requires reviewer decision below. |
| `vllm:prefix_cache_queries_total` | new eligible prompt-token counter | Counter; queried tokens, matching current vLLM semantics. |
| `vllm:prefix_cache_hits_total` | new reused prompt-token counter | Counter; cached tokens. |
| `vllm:prefix_cache_misses_total` | derived queried minus hit tokens | Counter; compatibility extension requested by issue 192. |
| `vllm:prefix_cache_hit_rate` | hits / queries | Gauge; zero when queries are zero. Existing AFM request-event hit/miss counters remain unchanged. |
| `vllm:spec_decode_num_draft_tokens_total` | new provider observation | Counter; all proposed speculative tokens. |
| `vllm:spec_decode_num_accepted_tokens_total` | new provider observation | Counter; accepted draft tokens. |
| `vllm:spec_decode_num_rejected_tokens_total` | derived draft minus accepted | Counter. |
| `vllm:spec_decode_num_drafts_total` | new provider observation | Counter; number of draft rounds, for Playground's existing panel. |
| `vllm:spec_decode_draft_acceptance_rate` | accepted / draft tokens | Gauge; zero when draft count is zero. |

Update vLLM histogram buckets to the pinned upstream values by appending new upper bounds only. Existing AFM bucket boundaries and all existing AFM names remain present, so historical dashboard queries continue to work.

### Cardinality policy

- Every vLLM family uses only `model_name` and fixed `engine="0"`, matching current upstream's bounded labels. `model_name` is process configuration, not request input.
- Only the two enumerated labels above are added: `finished_reason` and `status`. Unknown runtime values are normalized to `unknown`; arbitrary strings are never emitted.
- Do not label by request ID, API key, user, prompt, endpoint, sampling parameters, HTTP path/code, cache key, tool name, or exception type/message.
- Runtime/provider identity, if needed, is one bounded info gauge such as `afm:runtime_info{model_name,runtime="mlx|foundation|dwarfstar"} 1`; do not add it to every time series.
- Emit stable zero samples for expected bounded series so HELP/TYPE and Playground panels do not appear/disappear with traffic.

## Implementation sequence and likely files

1. **Add provider-neutral telemetry observations without moving `StatsAggregator`.**
   - Add a small callback/protocol and value types under `Sources/AFMKitCore/Telemetry/InferenceTelemetry.swift` for request lifecycle, token timestamps, prefix token counts, KV occupancy, and speculative rounds.
   - Make `StatsAggregator` the AFMKitMLX implementation/consumer in `Sources/AFMKitMLX/Models/StatsAggregator.swift`.
   - Inject the observer through model/runtime construction rather than importing AFMKitMLX from DwarfStar. Likely wiring files are `Sources/AFMCLI/main.swift`, `Sources/AFMKitMLX/AFMMLXRuntime.swift`, and `Sources/AFMKitMLX/AFMMLXProvider.swift`.

2. **Close MLX/DwarfStar telemetry gaps.**
   - Instrument `Sources/AFMKitMLX/Models/BatchScheduler.swift` and `Sources/AFMKitMLX/Models/MLXModelService.swift` once per lifecycle event, including speculative fast paths and streaming failures.
   - Emit DSpARK rounds/draft/accepted totals from `Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift` through the neutral observer.
   - Expose MTP/EAGLE3 totals from AFM-owned patch sources `Scripts/patches/Qwen3_5MoE.swift`, `Scripts/patches/DeepseekV4.swift`, and `Scripts/patches/Gemma4Eagle3.swift`, then update the existing patch application/checksum machinery. Do not edit vendored dependency sources directly.
   - Add a real KV logical-occupancy reader beside the current prefix-cache and Metal readers in `Sources/AFMKitMLX/Models/MLXModelService.swift` and the batched scheduler.

3. **Render additive compatibility metrics.**
   - Refactor `Sources/AFMServer/Controllers/MetricsController.swift` into AFM-native and vLLM compatibility render sections over one snapshot.
   - Preserve the current content type, CORS behavior, AFM HELP/TYPE strings, and AFM samples. Add deterministic HELP/TYPE ordering for all vLLM families.
   - Update `Scripts/grafana/README.md` and `Scripts/grafana/UPSTREAM.md` with the pinned upstream revision and the fact that direct vLLM scraping no longer needs a prefix rewrite.

4. **Implement legacy completions as a protocol adapter.**
   - Add `CompletionRequest`, completion response/chunk DTOs, and explicit coding keys under `Sources/AFMOpenAICompat/` (likely new `CompletionRequest.swift` and `CompletionResponse.swift`). Support `prompt` string and string-array forms required by GuideLLM, `max_tokens`, `stream`, `stream_options.include_usage`, stop, temperature, and model. Reject unsupported batched prompt shapes clearly rather than returning a malformed partial result.
   - Add `Sources/AFMServer/Controllers/CompletionsController.swift`. Route both Foundation and model-backed servers through the existing serving abstraction, but map wire output to `text_completion` objects and `choices[].text` rather than exposing chat objects.
   - Register POST/OPTIONS `/v1/completions` alongside both current chat controllers and document it in `Sources/AFMServer/Controllers/OpenAPIController.swift` and the startup route list in `Sources/AFMServer/Server.swift`.
   - Keep the existing chat `max_tokens`/`max_completion_tokens` precedence and default streaming-usage behavior unless a dedicated compatibility test proves GuideLLM requires a change.

5. **Make terminal and usage semantics machine-detectable.**
   - For successful streaming: role/content chunks, exactly one finish-reason chunk, optional final usage-only chunk when requested, then exactly one `[DONE]`.
   - For non-streaming: return actual prompt/completion/total counts from runtime telemetry and the terminal reason in the normal JSON response.
   - Before response headers, use the existing OpenAI error envelope and non-2xx status. After SSE starts, emit an OpenAI-shaped SSE error event and close without a success finish or `[DONE]`; verify GuideLLM treats it as a failed request. Do not convert failures to assistant text.
   - MLX fallback estimates must be visible as a documented limitation or eliminated for qualified paths. Foundation must not be advertised as exact until native `FoundationSessionUsageTelemetry` is wired into `ChatCompletionsController`.

6. **Stabilize model discovery.**
   - Extract `/v1/models` construction from `Sources/AFMServer/Server.swift` into a testable helper.
   - Use a stable created epoch from model metadata/process load time, sort discovered backends and embeddings by stable keys, and always keep the loaded generative model first. Keep IDs unchanged.

7. **Add repeatable qualification and documentation.**
   - Add `Scripts/test-vllm-guidellm-compat.py` for model discovery, both endpoints, stream/non-stream usage and finish checks, Prometheus parsing, AFM/vLLM equality, Playground-required family checks, and GuideLLM artifact validation.
   - Add focused shell integration to `Scripts/test-assertions.sh`; keep heavyweight live GuideLLM/model tests opt-in via explicit environment variables.
   - Add `docs/guidellm.md` with pinned-version install, server startup, synchronous/throughput/concurrent/constant/Poisson/sweep commands, output paths, parser/linter commands, supported request fields, and runtime-specific limitations.
   - Repair or remove the stale release-package references to absent `test-streaming.sh`/`test-metrics.sh` when adding the new script; do not expand this issue into a broader release refactor.

## Automated test matrix

| Layer | Test | Expected assertion |
| --- | --- | --- |
| Aggregator unit | Counter/histogram lifecycle | Accepted/completed/error/cancel paths increment once; reset preserves readers/metadata; no double counting. |
| Aggregator unit | Rolling throughput | Deterministic injected clock validates empty, partial-window, expiry, and concurrent completion behavior. |
| Aggregator unit | Prefix semantics | Request-event AFM counters stay unchanged; vLLM query/hit/miss token counters and rate are exact. |
| Aggregator unit | Speculative semantics | accepted <= draft; rejected is derived; zero denominator gives zero rate; all provider modes map identically. |
| Aggregator unit | TTFT/TPOT/ITL | Known timestamps produce distinct expected distributions; zero/one-token outputs do not fabricate intervals. |
| Renderer unit | AFM preservation | Golden list of every pre-issue AFM HELP/TYPE/family remains present and values are unchanged. |
| Renderer unit | vLLM schema | Stable HELP then TYPE then samples; counters/histograms follow Prometheus suffix rules; escaping and finite values are valid. |
| Renderer unit | Alias parity | Running/waiting, tokens, finish reasons, E2E and TTFT match AFM source values exactly. |
| Parser test | Prometheus validation | Pipe rendered output through `promtool check metrics` and Python `prometheus_client.parser`; reject duplicate/conflicting TYPE metadata. |
| Route test | `/metrics` | 200, existing content type/CORS, both namespaces, deterministic family order. |
| DTO unit | GuideLLM payload decoding | Chat and legacy payloads accept the pinned client's extra fields; both max-token field variants work as specified. |
| Protocol unit | Non-stream responses | Chat and text-completion object shapes, finish reasons, IDs/model, and exact usage are correct. |
| Protocol unit | SSE ordering | Both endpoints produce content, one finish chunk, requested usage-only chunk, and one `[DONE]` in that order. |
| Protocol unit | Usage disabled | `include_usage=false` suppresses only the usage chunk, preserving finish and `[DONE]`. |
| Protocol unit | Error semantics | Pre-header errors are non-2xx OpenAI JSON; post-header errors are detectable SSE errors and never assistant text/success finish. |
| Route unit | `/v1/models` | Two consecutive responses are byte-stable; loaded generative model is first; discovered/embedding tail is sorted. |
| Scheduler/service | Lifecycle paths | Batch, serial, MTP, EAGLE3, DSpARK, cancellation, and thrown errors each balance started/completed and observations. |

Likely test files:

- New `Tests/MacLocalAPITests/StatsAggregatorCompatibilityTests.swift`.
- New `Tests/MacLocalAPITests/MetricsControllerTests.swift`.
- New `Tests/MacLocalAPITests/LegacyCompletionsControllerTests.swift`.
- New `Tests/MacLocalAPITests/ModelDiscoveryDeterminismTests.swift`.
- Extend `Tests/MacLocalAPITests/StreamingUsageChunkTests.swift`.
- Extend `Tests/MacLocalAPITests/MLXChatCompletionsControllerStreamingTests.swift`.
- Extend `Tests/MacLocalAPITests/FoundationSessionUsageTelemetryTests.swift` if Foundation is included in the qualified set.

All SwiftPM builds/tests must run through `Scripts/swiftpm-reliable.sh` per repository instructions.

## Live qualification matrix

Run with a small deterministic MLX model first, then repeat applicable rows with prefix caching, each speculative backend, concurrency > 1, and Foundation if approved.

| Live scenario | Traffic/profile | Pass criteria/artifact |
| --- | --- | --- |
| Metrics idle | Scrape before requests | Prometheus parser/linter pass; all stable families exist; no NaN/Inf; AFM metrics retained. |
| Metrics single request | One fixed prompt/output | AFM/vLLM aliases agree; request/token/latency counts become nonzero; running/waiting return to zero. |
| Metrics concurrency | More streams than `--concurrent` | Running reaches capacity, waiting becomes nonzero, completed balances accepted, throughput nonzero. |
| Prefix cache | Same long prefix twice | AFM event counters preserve behavior; vLLM queried/hit token counts and rate agree with service logs/known lengths. |
| Speculative | MTP, EAGLE3, DSpARK separately | draft/accepted/rejected arithmetic holds; acceptance rate in `[0,1]`; non-spec mode remains zero. |
| Cancellation/failure | Cancel queued and active requests; force inference failure | abort/error/status counters increment once; GuideLLM reports errors rather than successful text. |
| Playground | Point pinned unmodified Playground at server base URL | Dashboard discovers `/metrics` and populates running/waiting, throughput, latency, cache, and applicable speculative panels without source changes. |
| GuideLLM synchronous | Both endpoints, stream false/true | Zero protocol errors; exact usage; correct finish and termination. |
| GuideLLM throughput | Throughput profile | Zero request errors and nonzero request/token rate metrics. |
| GuideLLM concurrent | At least 2 concurrent streams | JSON, CSV, and HTML artifacts are produced and contain nonzero TTFT, ITL/TPOT, E2E, request rate, input/output token counts. |
| GuideLLM constant | Constant-rate below and above capacity | Below-capacity errors zero; overload is bounded and reported as failures, not hangs. |
| GuideLLM Poisson | Seeded arrival profile | Completes without parser errors; artifacts contain nonzero latency/throughput. |
| GuideLLM sweep | Small concurrency/rate sweep | Every point completes, outputs are ordered/stable, no model rediscovery drift. |
| Repeatability | Repeat `/v1/models` and fixed benchmark | Model response is byte-stable; metric families/labels stay constant across scrapes. |

The qualification script should capture server version/config, model ID, runtime, upstream GuideLLM version/commit, commands, raw `/metrics`, GuideLLM JSON/CSV/HTML, and a concise pass/fail manifest. Generated benchmark artifacts remain ignored and are not committed.

## Compatibility risks and mitigations

- **Metric-name compatibility:** Playground is tied to legacy/current vLLM names. Emit the exact names it consumes plus current upstream names where they differ; lock the list in tests.
- **Semantic false equivalence:** Metal pressure is not KV usage, request cache hits are not token hits, and per-request TPOT is not ITL. Keep separate observations and HELP text.
- **Double counting:** Controller, service, and scheduler all see overlapping lifecycle stages. Define one owner per event and test every exit path.
- **Streaming false success:** Current text-wrapped errors are accepted by benchmark clients. Change only with explicit SSE error tests and document the post-header behavior.
- **Foundation accuracy:** Estimated token counts do not meet the issue's “accurate usage” requirement. Qualify MLX first or wire native telemetry before including Foundation.
- **Prompt semantics:** Implementing `/v1/completions` by blindly wrapping prompts in chat messages may add template tokens and alter raw-completion behavior. The adapter must either use a raw prompt-capable serving path or document/test the template boundary.
- **Bucket evolution:** Append upstream buckets; never delete old AFM buckets or change their counts.
- **Cardinality:** Model/runtime values must be configuration-derived and bounded; sanitize reason/status to enums rather than arbitrary strings.
- **External CLI drift:** Pin GuideLLM for qualification and record its version; keep its invocation isolated in one script.
- **Release harness drift:** The current workflow references missing scripts. Verify packaging separately and avoid claiming release coverage until the new harness is actually present and run.

## Unresolved architectural questions for reviewer approval

1. **KV denominator:** Should `vllm:kv_cache_usage_perc` mean logical resident tokens divided by `contextWindow * maxConcurrent`, allocated KV bytes divided by an explicit budget, or another runtime-native capacity? The existing Metal gauge cannot be reused. Recommendation: logical token-slot occupancy because it is portable and deterministic, with runtime-specific readers.
2. **Foundation qualification:** Is GuideLLM interoperability required for Foundation in this issue, or may the acceptance claim be MLX/DwarfStar only until native Foundation usage replaces estimates? Recommendation: do not claim Foundation accuracy with heuristic counts.
3. **Legacy completion semantics:** Must `/v1/completions` be true raw-prompt generation, or is a documented prompt-to-chat adapter acceptable for GuideLLM? Recommendation: use a raw prompt path; template wrapping changes both output and usage.
4. **Streaming error wire format:** GuideLLM's exact behavior for an SSE `error` object after HTTP 200 must be confirmed live. If it does not classify that as failure, the safe alternative is to terminate the stream without `[DONE]` and count a client-visible transport failure.
5. **Telemetry bridge breadth:** Approve the small `AFMKitCore` observer contract so DwarfStar can report to the existing MLX-owned aggregator without a target cycle. The alternative is to limit speculative metrics to MLX, which would leave DSpARK incomplete.
6. **Compatibility endpoint fallback:** Approve additive AFM+vLLM output on `/metrics`. A separate `/metrics/vllm` may be added for humans/tools, but cannot be the only compatibility surface while Playground hardcodes `/metrics`.
7. **Throughput window:** Approve a fixed, documented 10-second rolling window with an injected monotonic clock. If strict upstream parity is required instead, define the exact vLLM logging interval semantics before implementation.

Implementation remains blocked on reviewer approval of this plan and the questions above.
