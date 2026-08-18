# Issue 192 Phase A: vLLM Metrics and GuideLLM Interoperability Plan

Status: architecture approved at `343d02acf482620a76a81fd29a87488693a92531`; production implementation is in progress.

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
- Chat accepts `max_tokens` and `max_completion_tokens`, preferring the former (`Sources/AFMOpenAICompat/OpenAIRequest.swift:8`, line 66). At this checkpoint GuideLLM extensions such as `ignore_eos` and `continuous_usage_stats` are ignored by Codable; issue 192 makes their accepted semantics explicit below.
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

AFM and vLLM rendering consume one immutable provider-neutral snapshot. Built-in/server composition uses only push-based collector updates, so its scrape path is one atomic state copy with no callback invocation or scrape-time mutation. The deprecated legacy facade has an explicitly separate non-atomic callback-sampling behavior described below; those callbacks never participate in the server renderer's snapshot source.

## Selected telemetry architecture

This plan selects **cross-runtime qualification** rather than keeping the concrete collector in AFMKitMLX.

- `AFMKitCore` owns only provider-neutral cross-target contracts: immutable `AFMInferenceMetricsSnapshot`/`AFMHistogramSnapshot` values, `AFMInferenceMetricsSnapshotSource`, `Sendable` provider observation protocols, bounded provider finish/failure enums, generation-admission contracts, provider-neutral generation options, and raw-generator type erasure. Proposed snapshot declarations live in `Sources/AFMKitCore/Telemetry/InferenceMetricsSnapshot.swift`. Core has no mutable collector state, rolling-window implementation, Prometheus names, Vapor/OpenAI types, server rejection/connection protocol, transport status enum, or provider dependency.
- `AFMKitServices` owns the process collector's mutable state, locks, rolling-window/clock implementation, one-terminal enforcement, server-ingress mutation interface, legacy callback adapter, and conversion of one atomic state read into the Core-owned immutable snapshot. Proposed files are `Sources/AFMKitServices/Telemetry/InferenceTelemetryCollector.swift`, `IngressTelemetryRecording.swift`, and `LegacyInferenceMetricsCompatibilityAdapter.swift`. Services does not define a second public snapshot type.
- AFMKitMLX and AFMKitDwarfStar receive an `any AFMInferenceTelemetryObserving` during construction. They never import AFMKitServices; the CLI/server composition root injects the collector through the AFMKitCore protocol.
- AFMKitServices owns `AFMIngressTelemetryRecording`, `AFMIngressRejectionReason`, and `AFMIngressConnectionToken`. The protocol has only `recordRejection(_:)`, `connectionOpened()`, and idempotent `connectionClosed(_:)`; it cannot allocate a provider request token or mutate provider accepted/terminal totals. AFMServer depends explicitly on AFMKitServices and owns an `AFMServerTelemetryAdapter` that maps HTTP decode/authentication/validation and provider queue-capacity errors into the Services enum. No declaration whose name or cases encode server/HTTP policy is added to AFMKitCore.
- AFMServer receives read-only `any AFMInferenceMetricsSnapshotSource` from Core for rendering and the Services-owned ingress recorder through its server adapter for bounded pre-admission rejections and active-connection lifecycle. AFMServer owns HTTP classification; AFMKitServices remains the sole mutable state owner.
- `Sources/AFMKitMLX/Models/StatsAggregator.swift` remains a concrete deprecated forwarding facade with its existing public name, `shared`, nested types, and method signatures. It is not a type alias and owns no authoritative metric state. The explicit compatibility bridge is defined below.
- `Sources/AFMCLI/main.swift` creates exactly one collector per server process, constructs the AFMServer ingress adapter and the legacy-facade compatibility bridge, and injects the collector's provider observer and snapshot-source views into provider/runtime and Server. `Package.swift` adds an explicit `AFMServer -> AFMKitServices` dependency while preserving `AFMKitServices -> AFMKitCore` and provider-target `-> AFMKitCore` direction.
- Foundation is not adapted to the provider event contract in issue 192 and is not part of the qualified matrix below.

Provider observation and Services ingress recording are synchronous, nonblocking, `Sendable`, cancellation-safe, and limited to a short lock/atomic update. They must never await, call provider code, invoke legacy gauge callbacks, or re-enter a scheduler actor. Provider event values include an opaque collector-issued request token so terminal deduplication does not require request IDs as Prometheus labels. Server rejection calls cannot allocate or receive a provider token.

`AFMIngressRejectionReason` is the closed Services enum `decode`, `authentication`, `validation`, `capacity`; provider failures (`cancelled`, `inference`, `internal`) enter through the Core provider observer instead. The opaque Services connection token makes close idempotent and carries no request/model label data. The Core snapshot exposes immutable renderer-neutral named-count entries without importing the Services enum. Services constructs only its closed keys, and AFMServer renders only the closed Services/provider enum cases; unknown externally supplied keys are ignored with a diagnostic, so the public generic value does not create unbounded Prometheus cardinality.

### Immutable snapshot boundary

`AFMInferenceMetricsSnapshotSource` is implementable without a reverse dependency because its method returns the concrete Core-owned `AFMInferenceMetricsSnapshot`:

```swift
public protocol AFMInferenceMetricsSnapshotSource: Sendable {
    func metricsSnapshot() -> AFMInferenceMetricsSnapshot
}
```

The snapshot contains immutable scalar counters/gauges/metadata, bounded renderer-neutral reason/status entries, and Core-owned `AFMHistogramSnapshot` values (`buckets`, cumulative counts, sum, count). It contains no Services collector type or enum, lock, closure/gauge reader, Prometheus name, HTTP concept, or renderer policy. `InferenceTelemetryCollector` in AFMKitServices conforms by locking/copying its push-based state once and returning this value. AFMServer depends on the Core protocol/value for rendering and separately on Services for ingress writes. Provider targets depend on Core only and receive provider write/admission interfaces, not the Services implementation.

### Public compatibility and migration contract

AFMKit is distributed as SwiftPM source and does not enable library evolution. Issue 192 guarantees source compatibility after consumers rebuild; it does not promise ABI compatibility for already-compiled modules, protocol witness tables, or hot-swapped binaries. Existing public names, exact pre-issue protocol requirements, and practical old initializer overloads remain available for the full current major version.

#### `StatsAggregator` facade

- Keep `public final class StatsAggregator`, `public static let shared`, `GaugeReader`, `FractionReader`, `Buckets`, nested `Histogram`, `RequestObservation`, nested `Snapshot`, and every current public mutation/registration/observation/snapshot method in `Sources/AFMKitMLX/Models/StatsAggregator.swift` with the same signatures and accessibility.
- Mark the concrete class deprecated with a message directing new code to injected AFMKitCore telemetry protocols and `InferenceTelemetryCollector` from AFMKitServices. Do not replace it with a type alias: qualified names such as `StatsAggregator.Snapshot` and `StatsAggregator.Histogram` must continue to compile.
- Add an AFMKitMLX-local `StatsAggregatorCompatibilityTarget` used only by the facade; no legacy MLX callback or server contract is added to Core. AFMKitServices owns `LegacyInferenceMetricsCompatibilityAdapter`, which forwards legacy counters/histograms/metadata into the one `InferenceTelemetryCollector` and stores only legacy gauge-reader closures. A thin bridge under `Sources/AFMKit/Compatibility/StatsAggregatorServicesCompatibilityTarget.swift` can import both AFMKitMLX and AFMKitServices and conform to the facade-local protocol without reversing either target dependency.
- The concrete MLX facade stores only a thread-safe reference to `any StatsAggregatorCompatibilityTarget`. The composition root binds the bridge before constructing providers or Server. Every legacy counter/observation/reset/metadata method forwards one-for-one to the authoritative Services collector; `snapshot()` maps the adapter's Core snapshot into the existing nested `StatsAggregator.Snapshot`; nested `Histogram` remains a source-compatible local value wrapper over `AFMHistogramSnapshot` semantics.
- Legacy gauge callbacks are compatibility-only and explicitly non-atomic. On an outer `StatsAggregator.snapshot()`, the Services adapter copies callback references while holding only its callback lock, releases every lock, invokes each currently registered reader exactly once, obtains one atomic push-based collector snapshot, and overlays the sampled legacy gauge values only in the facade snapshot returned to that caller. After external callbacks return, the adapter updates a compatibility-only `legacyBatchSizePeak = max(previousPeak, sampledRunning)` under its own lock so the deprecated facade preserves its old peak behavior; `reset()` clears that compatibility peak. Callback values and this compatibility peak never mutate the collector and never appear in AFMServer's `/metrics` snapshot. No collector/facade lock is held while external code runs.
- Re-entrant `StatsAggregator.snapshot()` from inside a gauge callback is detected by the bridge. The nested call skips all callback sampling and returns the collector's push-based base snapshot, preventing recursion and deadlock; the outer call continues and applies its samples. Re-registration racing a snapshot affects the next snapshot, not the copied callback set. Built-in MLX, DwarfStar, AFMServer middleware, and MetricsController are forbidden by tests from registering or invoking this legacy callback path.
- Provide an explicit `StatsAggregator.installCompatibilityTarget(_:)` accepting the AFMKitMLX-local protocol. It is idempotent for the same target and rejects replacement after the first forwarded mutation or snapshot, preventing split state. Shipped CLI composition installs the AFMKit bridge backed by the same Services collector used by new protocols, but built-in metric production remains push-based.
- When no target is installed, the deprecated facade uses an AFMKitMLX-local stateless no-op target and returns a documented zero snapshot. It does not create fallback counters or a second registry. Existing source still builds; standalone consumers that require legacy metrics add the `AFMKit` compatibility product, create one `InferenceTelemetryCollector` plus bridge, install it, and inject the collector's Core protocol views. This is the explicit behavioral migration.
- Existing AFMKitMLX provider code is migrated off `StatsAggregator.shared`; the facade is only for external-source compatibility. It imports AFMKitCore, never AFMKitServices, so AFMKitMLX remains independent of Services.

#### Scheduling and chat-serving protocols

- Preserve the existing `AFMMLXRequestScheduling` protocol exactly: `maxConcurrent`, `tryReserveSlot()`, `waitForSlot(timeout:)`, and `releaseSlot()` remain its only requirements. Preserve every current requirement and default overload of `AFMMLXOpenAIChatGenerating` and `AFMMLXOpenAIChatServing`.
- Add a new refined `AFMMLXGenerationAdmitting: AFMMLXRequestScheduling` protocol with required `var generationAdmitter: AnyAFMGenerationAdmitter { get }`. The type eraser and admission value/lease types are Core-owned. Existing external conformers acquire no witness-table or source requirement.
- Built-in MLX and erased DwarfStar services conform to the refined protocol and are forbidden by tests from using the legacy polling path. Controllers capability-cast the existing scheduling existential to `any AFMMLXGenerationAdmitting` and call its provider-owned admitter for qualified runtimes.
- If an external conformer does not implement the refined capability, controllers preserve current behavior through a deprecated `LegacyAFMMLXAdmissionAdapter` that calls its existing `waitForSlot(timeout:)`/`releaseSlot()`. That fallback preserves service behavior and source-rebuild compatibility but is explicitly not vLLM waiting/latency-qualified because it cannot report provider queue admission.
- Do not add admission requirements directly to `AFMMLXOpenAIChatGenerating` or raw generation methods to `AFMMLXOpenAIChatServing`. Raw generation remains an optional `AnyAFMRawTextGenerator` capability, so old conformers do not acquire new requirements.
- Preserve every existing public initializer declaration unchanged where telemetry/admission composition touches it. Add a new overload carrying the new dependency and have the old overload forward to a documented no-op/default composition. Consumers rebuild from source, but exact old source signatures and symbols are retained where practical; no initializer is replaced merely by appending a defaulted argument.

Compile-level compatibility coverage imports public modules without `@testable`, defines external conformers implementing only the pre-issue protocol requirements, exercises every retained `StatsAggregator` nested type/method, and constructs services through the exact old public initializers. A small local-package fixture also builds against the issue branch to catch cross-module qualified-name and capability-cast failures. The compatibility documentation states that SwiftPM consumers must rebuild and that no binary-module ABI guarantee is made.

### Event ownership and one-terminal rule

A request becomes **accepted** when the provider atomically inserts it into its bounded runnable/waiting admission queue and allocates its collector token. It does not wait for a compute slot first. Queue insertion increments accepted and waiting in one provider-owned operation, so slot-wait time is visible in waiting, E2E, queue latency, and TTFT. A provider that refuses insertion because the admission queue itself is full returns a typed rejection with no token; AFMServer maps and records that pre-admission capacity rejection. Every accepted request gets exactly one provider-owned terminal event, including timeout or cancellation while waiting.

| Event/state | Sole owner | Recording rule |
| --- | --- | --- |
| HTTP body decode, auth, endpoint/model/client-field validation failure | AFMServer maps through `AFMServerTelemetryAdapter` to Services ingress recorder | Record one bounded rejection status; no provider request token and no accepted/terminal mutation. |
| Admission-queue capacity rejection | Provider decides; AFMServer maps through its adapter to Services | Provider returns typed rejection before insertion and without token. Server records `capacity`; accepted/terminal remain unchanged. |
| Provider queue admission | Runtime/provider | Atomically insert into waiting queue, allocate token, emit `accepted`, and start accepted latency exactly once. |
| Slot wait | Runtime scheduler | Accepted request remains waiting. Success moves waiting to running; timeout/cancel removes waiting and emits the sole provider terminal (`abort`) for its token. |
| Waiting/running gauges | Runtime scheduler | Publish an atomic snapshot from the admission queue and active leases. Server-side polling/reservation is removed. |
| Full prompt token count | Runtime/provider tokenizer | Record exact model-input token IDs once per accepted request, including reusable cached prefix. |
| Computed prompt token count | Runtime/provider prefill | Record only prompt token positions actually computed after prefix reuse. |
| Output tokens and monotonic timestamps | Runtime/provider generation loop | Record actual generated token events; server text chunk boundaries are irrelevant. |
| Prefix queried/hit token counts | Runtime/provider cache owner | Record eligible queried tokens and reused tokens once, at cache resolution. |
| Logical KV positions | Runtime scheduler | Publish atomic active-position sum; waiting and retained radix entries contribute zero. |
| Speculative draft round/draft/accepted token counts | Runtime speculative implementation | Record one observation per draft round; collector derives rejected totals and rate. |
| Successful/failed runtime terminal | Runtime/provider | Atomically claim the request token's terminal state once, with bounded finish/failure reason and exact usage. |
| Client disconnect/cancel after acceptance | AFMServer requests; provider owns result | Server signals cancellation. Provider removes runtime/KV state and emits the sole `cancelled` terminal. Server must not emit a second terminal metric. |
| Wire HTTP status, JSON errors, SSE framing | AFMServer | Map provider result/error to wire state only; never increment accepted-runtime terminal counters. |
| Active HTTP connection open/close | AFMServer middleware through `AFMServerTelemetryAdapter` to Services | Migrate current active/peak writes from `StatsAggregator.shared`; use a Services connection token/idempotent close so every counted open closes once. Existing route exclusions remain unchanged. |

The shared Core admission contract is used by chat and raw generation. AFMServer performs decode/auth/field validation, then calls the provider admission operation once; `MLXChatCompletionsController` and `AFMKitMLXChatServingAdapter` no longer poll or reserve slots. Admission asynchronously waits inside the provider after queue insertion and returns a running lease on success. The lease carries the telemetry token and has provider-owned idempotent release/terminal cleanup. Pre-insertion rejection returns no lease/token. Cancellation races are resolved by the provider's single terminal compare-and-set.

The collector rejects duplicate terminal events for the same telemetry token and tests assert balanced accepted/terminal counts for every exit path. Services-ingress/AFMServer-adapter tests separately prove that rejection and connection writes cannot allocate provider tokens or mutate accepted/terminal totals.

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
| `vllm:request_success_total{finished_reason}` | sole provider terminal | Pinned family with exactly `stop`, `length`, `abort`, `error`, `repetition`; all five series are pre-created. |
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
| `afm:request_failures_total{status}` | server rejection plus provider failure classification | AFM issue-192 addition, not represented as upstream vLLM. Closed union: `decode`, `authentication`, `validation`, `capacity`, `cancelled`, `inference`, `internal`. |
| `afm:prefix_cache_missed_tokens_total` | queries minus hits | AFM issue-192 addition, not upstream vLLM. Existing request-event radix counters remain unchanged. |
| `afm:spec_decode_num_rejected_tokens_total` | draft minus accepted | AFM issue-192 addition, not upstream vLLM. |
| `afm:request_throughput_requests_per_s` | rolling terminal events | AFM issue-192 addition, not upstream vLLM. |

Create pinned fixtures such as `Tests/MacLocalAPITests/Fixtures/vllm-9633933-metrics.prom` and `Tests/MacLocalAPITests/Fixtures/vllm-playground-7627622-registry.json`. Renderer tests lock canonical sample names, HELP/TYPE names, label sets, bucket boundaries, and the exact five `finished_reason` values to those fixtures. Prometheus counters retain `_total` on rendered samples. AFM additions are tested separately and never described as upstream families.

Append only the upper histogram bounds added by pinned upstream. Do not remove old AFM buckets or change their cumulative values.

### Cardinality policy

- Canonical vLLM families use only process-derived `model_name` and fixed `engine="0"`, matching pinned upstream.
- Canonical vLLM `finished_reason` is a closed five-value enum: `stop`, `length`, `abort`, `error`, `repetition`. It has no `unknown` or `tool_calls` series. AFM rejection `status` remains a separate closed enum whose unexpected internal value may collapse to `unknown`.
- Do not label by request ID/token, API key, user, prompt, endpoint, sampling values, HTTP code/path, cache key, tool name, or error type/message.
- Runtime identity, if exposed, is one bounded AFM info gauge, not a label copied onto every series.
- Expected bounded series render stable zero samples so schema does not change with traffic.

### Engine finish labels versus OpenAI wire finish reasons

Provider terminal observations store a pinned engine reason separately from the OpenAI response reason. AFMServer may refine the wire reason after parsing generated content, but it cannot mutate the canonical metric label.

| Provider/runtime outcome | `vllm:request_success` label | OpenAI wire behavior |
| --- | --- | --- |
| Natural stop or stop sequence | `stop` | `finish_reason: "stop"`. |
| Valid tool call parsed from a normally stopped generation | `stop` | Preserve `finish_reason: "tool_calls"`; `tool_calls` is never a vLLM label. |
| Maximum output/context limit | `length` | `finish_reason: "length"`. |
| Repetition guard terminates generation | `repetition` | Map to wire `finish_reason: "stop"` because OpenAI has no repetition reason; retain provider diagnostic separately. |
| Accepted request cancelled/aborted, including slot-wait timeout/cancel | `abort` | Error/cancellation path; no successful finish event. |
| Accepted request fails in provider/runtime | `error` | OpenAI error JSON/SSE path; no successful finish event. |

Existing AFM-native finish-reason output remains unchanged, including any `tool_calls` wire-oriented AFM series. Only the vLLM compatibility renderer applies the canonical mapping. Tests drive all five provider outcomes, assert exactly five pre-created vLLM label values, and independently assert that tool-call responses still use the OpenAI wire value `tool_calls`.

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

- `AFMRawTextGenerationRequest`: one `prompt: String`, selected model, maximum output tokens, stop strings, supported sampling values, seed, and provider-neutral `ignoreEndOfSequence`. It contains no messages, roles, instructions, or HTTP/SSE fields.
- `AFMRawTextGenerationEvent`: text/token delta with provider monotonic timestamp, followed by exactly one terminal result containing bounded finish reason and exact `promptTokens`, `completionTokens`, and `totalTokens`; or one typed provider failure.
- `AFMRawTextGenerating`: capability plus a generation method returning the provider event stream. The same stream is collected for non-streaming responses, preventing separate usage/finish implementations.

Provider requirements:

1. Tokenize the raw prompt and construct the runtime equivalent of `UserInput(prompt:)` directly.
2. Do not insert system/default instructions, synthesize a user/assistant turn, invoke a chat template, or use `buildUserInput`'s message path.
3. Count the exact model input token IDs as full prompt usage even when a prefix is cached; report computed prompt tokens separately to telemetry.
4. Count actual generated token IDs, preserve stop/length semantics, and emit one provider terminal event.
5. When `ignoreEndOfSequence` is true, exclude model EOS token IDs as generation stop candidates until an explicit caller stop sequence, context limit, cancellation/failure, or maximum output-token limit ends the request. Do not emit an EOS token as user-visible text or count a discarded EOS candidate as an output token. Explicit stop strings and structured/tool completion remain authoritative.
6. Advertise the capability only when these semantics and exact usage are implemented.

### Capability-preserving model erasure

Use a Core-owned type eraser rather than attempting a runtime cast after model erasure:

- Add `AnyAFMRawTextGenerator` in `AFMKitCore`. It captures the raw admission/generation closures of an `AFMRawTextGenerating` provider without importing server or wire types.
- Extend `AnyAFMModel` in `Sources/AFMKitCore/AFMProviderRegistry.swift` with `public let rawTextGenerator: AnyAFMRawTextGenerator?`. Its generic initializer captures the optional conformance before storing the base model closures; non-conforming models store `nil`.
- Provider-registry model creation returns an `AnyAFMModel` whose optional raw generator survives registry lookup and erasure.
- The DwarfStar CLI path constructs `AFMDwarfStarModel`, erases it once, and passes that same `AnyAFMModel` to `Server`; `Server` reads `afmModel.rawTextGenerator` and injects it into `CompletionsController`. It never attempts to recover the concrete DwarfStar model.
- MLX composition uses the same erased capability when a model travels through `AnyAFMModel`; direct MLX service composition wraps the same provider contract in `AnyAFMRawTextGenerator` before server registration.

AFMOpenAICompat owns `CompletionRequest`, response, chunk, usage, and error wire DTOs. AFMServer owns DTO validation and mapping provider events to OpenAI JSON/SSE. `/v1/completions` POST/OPTIONS is registered only when the composed `AnyAFMModel.rawTextGenerator` or direct `AnyAFMRawTextGenerator` is non-`nil`; unsupported runtimes do not silently route through chat.

Composition tests cover: direct conforming/non-conforming erasure; provider-registry construction through `AnyAFMModel`; DwarfStar CLI server-composition helper from concrete model through erasure to registered route; and DSpARK using the same retained capability. These are in addition to direct provider conformance tests.

### Prompt-array behavior

The request DTO decodes `prompt` as either string or array so it can return a stable protocol error. Issue 192 supports only one prompt string, matching pinned GuideLLM. Any array, including a one-element array, is rejected before provider admission with:

- HTTP `400`.
- `type: "invalid_request_error"`.
- `code: "unsupported_prompt_array"`.
- `param: "prompt"` when the existing error DTO supports it; otherwise include `prompt` in the stable message and add `param` as part of the DTO work.

No partial choices, usage, accepted telemetry, or SSE headers are emitted for an array.

### Pinned GuideLLM extension semantics

Issue 192 implements both compatibility extensions sent by pinned GuideLLM rather than relying on unknown-field decoding:

- Add `ignoreEOS: Bool?` (`ignore_eos`) to chat and legacy completion DTOs. AFMServer maps `true` to Core `AFMGenerationOptions.ignoreEndOfSequence` for chat and `AFMRawTextGenerationRequest.ignoreEndOfSequence` for raw completions; absent/false preserves existing EOS behavior. The Core option is provider-neutral because it describes a generation stopping policy, not a GuideLLM or OpenAI transport field.
- Add `continuousUsageStats: Bool?` to `StreamOptions`. The field is accepted for both endpoints, but it does not request per-token usage chunks in issue 192. Exact provider terminal usage remains authoritative and is emitted once in the final usage-only event when `include_usage` is true. This is sufficient for the pinned GuideLLM handlers, which retain the most recent usage object; no intermediate estimate is emitted.
- `stop: null` means no caller stop strings. It does not cancel model EOS by itself; only `ignore_eos: true` changes EOS handling.
- MLX batch/serial/MTP/EAGLE3 and DwarfStar/DSpARK must implement EOS exclusion in AFM-owned provider/scheduler code. If an implementation cannot exclude EOS for a sampling mode without changing an upstream dependency, that runtime/mode is not GuideLLM-qualified and is route/documentation-gated until the AFM-owned implementation exists. No upstream repository is modified.

Deterministic DTO/provider tests cover absent/false/true `ignore_eos`, `continuous_usage_stats` with usage enabled/disabled, explicit stop strings while EOS is ignored, maximum-token length termination, and exact token accounting. Live fixed-output GuideLLM rows require the requested output-token count unless an explicit stop string or context bound ends generation; early EOS is a qualification failure.

## Exact runtime qualification matrix

| Runtime/path | vLLM metrics | Chat GuideLLM | Raw `/v1/completions` | Usage claim | Issue-192 result |
| --- | --- | --- | --- | --- | --- |
| MLX batch and serial | Required | Required, stream and non-stream | Required through `AFMRawTextGenerating` | Exact provider token IDs and `ignore_eos` | Fully qualified baseline. |
| MLX MTP/EAGLE3 | Required including spec metrics | Required | Required through same raw contract | Exact provider token IDs and `ignore_eos` | Qualified after the same matrix passes with each mode. |
| DwarfStar and DSpARK | Required through neutral observer, including DSpARK spec metrics | Required | Required through `AFMRawTextGenerating` | Exact provider token IDs and `ignore_eos` | Cross-runtime qualification is required; do not register raw route or claim success until contract passes. |
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

1. Add only provider-neutral Core API: immutable snapshot/histogram values, concrete snapshot-source protocol, provider observation protocol, five-value canonical finish enum, shared admission lease/type eraser, provider-neutral `ignoreEndOfSequence`, raw-prompt contract, and `AnyAFMRawTextGenerator` under `Sources/AFMKitCore/Telemetry/` and `Sources/AFMKitCore/Providers/`; extend `AnyAFMModel` in `AFMProviderRegistry.swift` to retain the optional raw capability. Preserve the exact old `AFMGenerationOptions` initializer and add an overload that accepts `ignoreEndOfSequence`.
2. Immediately extract and review the new AFMKitCore symbol graph, intentionally update `docs/api-baselines/AFMKitCore.symbols.json` in the same implementation commit, and require `./Scripts/check-afmkit-core-api.sh` to pass. Add a dedicated PR/push workflow such as `.github/workflows/afmkit-core-api.yml` that runs this script on macOS; this gate is mandatory CI, not an optional release check. Do not modify `.github/workflows/release.yml`.
3. Add the mutable collector/clock/rolling-window implementation plus Services-owned ingress protocol/token/enum and legacy callback adapter under `Sources/AFMKitServices/Telemetry/`. The collector conforms to Core provider observation and snapshot-source protocols; the ingress recorder mutates the same collector without provider tokens; the legacy adapter owns callback references outside collector state. Update `Package.swift` while preserving `AFMKitServices -> AFMKitCore`, provider-target `-> AFMKitCore`, and adding explicit `AFMServer -> AFMKitServices` direction.
4. Rewrite `Sources/AFMKitMLX/Models/StatsAggregator.swift` as the concrete deprecated forwarding facade while retaining its full public surface and adding only its local compatibility-target protocol. Add `Sources/AFMKit/Compatibility/StatsAggregatorServicesCompatibilityTarget.swift` as the cross-product bridge. Add the new refined `AFMMLXGenerationAdmitting` protocol without changing `AFMMLXRequestScheduling` or chat-serving requirements.
5. Instrument `BatchScheduler.swift`, `MLXModelService.swift`, `AFMMLXRuntime.swift`, and `AFMMLXProvider.swift` through injected Core protocols, implement push-based live gauges and `ignoreEndOfSequence`, and move built-in slot wait/reservation from `MLXChatCompletionsController.swift` into provider admission. Preserve exact old public initializers and add overloads for injected dependencies.
6. Instrument `Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift` and its model/runtime construction; replace pre-generation polling in `AFMKitMLXChatServingAdapter.swift` with the same provider-owned admission contract; implement push-based gauges and EOS exclusion in AFM-owned scheduler/bridge code. Expose MTP/EAGLE3/DSpARK observations from AFM-owned sources. Patch only `Scripts/patches/Qwen3_5MoE.swift`, `DeepseekV4.swift`, or `Gemma4Eagle3.swift` if direct implementation proves a callback unavailable; never edit vendor or upstream dependency sources directly.
7. Construct one collector in `Sources/AFMCLI/main.swift`; construct/install the AFMKit legacy-facade bridge before runtime start, inject the Core provider observer into runtimes, construct the AFMServer-to-Services ingress adapter, and inject the Core snapshot source into Server. Add a testable DwarfStar server-composition helper that preserves `AnyAFMModel.rawTextGenerator` through erasure. Built-in/server composition must not register a legacy gauge reader.
8. Refactor `Sources/AFMServer/Controllers/MetricsController.swift` into unchanged AFM-native rendering plus pinned vLLM rendering over the collector's atomic Core snapshot. Add `AFMServerTelemetryAdapter`, inject it into request-validation paths and `ActiveConnectionsMiddleware`, and migrate all active/peak writes away from `StatsAggregator.shared`. Keep ingress rejection writes disjoint from provider accepted/terminal state. Update `Scripts/grafana/README.md` and `Scripts/grafana/UPSTREAM.md`.
9. Add raw completion DTOs under `Sources/AFMOpenAICompat/CompletionRequest.swift` and `CompletionResponse.swift`; extend chat/stream option DTOs with `ignore_eos` and `continuous_usage_stats`; add `Sources/AFMServer/Controllers/CompletionsController.swift`; register routes from retained `AnyAFMRawTextGenerator` capability and update `OpenAPIController.swift` and Server route output.
10. Correct chat and raw SSE framing/error/final-usage behavior in `MLXChatCompletionsController.swift`, `ChatCompletionsController.swift` only where existing unqualified behavior must not be shared, and common response helpers/DTOs.
11. Extract deterministic model-list construction from `Sources/AFMServer/Server.swift` into a testable helper.
12. Add `Scripts/test-vllm-guidellm-compat.py`, focused deterministic coverage in `Scripts/test-assertions.sh`, and `docs/guidellm.md`. Document source-rebuild compatibility, the accepted-but-final-only `continuous_usage_stats` behavior, provider-neutral `ignore_eos`, and any runtime/mode withheld from qualification. Heavy model/GuideLLM runs remain opt-in.

Release packaging/workflow cleanup is explicitly out of scope. Do not modify `.github/workflows/release.yml` or repair/remove its stale script references under issue 192.

## Automated test matrix

| Layer | Test | Required assertion |
| --- | --- | --- |
| Collector | Lifecycle/terminal dedupe | Accepted/terminal balances; duplicate terminal ignored/reported; cancellation race is exactly once. |
| Collector | Nonblocking observer | Concurrent event calls do not await or re-enter a scheduler; deterministic stress test has no lost counts. |
| Target boundary | Core snapshot contract | AFMKitCore snapshot/source compile without Services; Services conforms; MLX/DwarfStar import no Services module; AFMServer renders using only the Core value/protocol. No Core declaration contains `Server`, HTTP/OpenAI DTOs, ingress rejection cases, connection lifecycle, legacy callbacks, or Services types. |
| Services/AFMServer adapter | Writable ownership | Decode/auth/validation/capacity mapping increments only bounded Services ingress state; no provider token, accepted, or terminal mutation. Active connection open/close is balanced and peak remains correct. |
| Collector | Full/computed/cache split | On cache hit `F/H/C`, vLLM full `+F`, AFM computed and throughput `+C`, query `+F`, hit `+H`, miss `+C`. |
| Collector | Rolling throughput | Injected monotonic clock covers empty, partial 10-second window, expiry, and concurrency. |
| Collector | Speculation | Draft/accepted/derived rejected arithmetic and exact `vllm:spec_decode_acceptance_rate`; zero denominator is zero. |
| Collector | TTFT/TPOT/ITL | Known token timestamps produce distinct values; zero/one-token outputs fabricate no intervals. |
| Collector | KV logical occupancy | Admission/decode/completion/cancel bounds and retained-prefix exclusion. |
| Renderer | AFM preservation golden | Every pre-issue AFM HELP/TYPE/sample family and meaning remains unchanged. |
| Renderer | Pinned vLLM fixture | Exact canonical names, `_total` samples, HELP/TYPE, labels, buckets, and exactly `stop|length|abort|error|repetition` finish series match pinned fixtures. |
| Renderer | Playground registry | Every pinned registry key exists, especially `vllm:spec_decode_acceptance_rate`; AFM additions are not classified as upstream. |
| Renderer | Prometheus parser/linter | `promtool check metrics` and `prometheus_client.parser` accept output; no duplicate/conflicting metadata. |
| Renderer | Shared-source parity | Only true aliases (running/waiting/generated/latency/finish) agree; full and computed prompt totals intentionally differ on hits. |
| Route | `/metrics` | 200, existing content type/CORS, both namespaces, deterministic order, no scrape-time counter mutation. |
| Raw provider | Template bypass | Exact raw prompt reaches `UserInput(prompt:)`; no system instruction, messages, chat template, or role tokens. |
| Raw composition | Type erasure | Direct `AnyAFMModel`, provider registry, and DwarfStar CLI composition retain conforming raw capability and omit it for non-conforming models. |
| DTO/controller | Prompt array | Every array shape returns stable pre-header 400 and no provider admission. |
| DTO/provider | GuideLLM extensions | `continuous_usage_stats` is accepted with one final exact usage event; `ignore_eos` absent/false/true maps correctly for chat/raw, explicit stops remain active, and true reaches each qualified provider. |
| Provider | Ignore EOS | MLX batch/serial/MTP/EAGLE3 and DwarfStar/DSpARK exclude EOS without counting/emitting it, continue to the requested maximum, and terminate as `length`; unsupported runtime/modes remain unqualified. |
| Protocol | Non-stream success | Chat and text-completion shapes, finish reasons, IDs/model, exact usage. |
| Protocol | Chat SSE | Delta payloads, one finish, optional usage-only, one `[DONE]` in exact order. |
| Protocol | Legacy SSE | Text payloads/no role/no delta, one finish, optional usage-only, one `[DONE]`. |
| Protocol | SSE post-header error | One OpenAI error event, no finish/usage/text substitution/`[DONE]`; pinned GuideLLM counts failure. |
| Protocol/metrics | Finish mapping | Wire `tool_calls` records canonical vLLM `stop`; repetition records vLLM `repetition` and wire `stop`; all five canonical labels are pre-created. |
| Protocol | Usage disabled | Suppresses only usage event and preserves successful finish/`[DONE]`. |
| Discovery | Determinism | Consecutive responses byte-stable; loaded generative model first; tail sorted. |
| Runtime matrix | Capability/route gating | MLX and DwarfStar expose raw route only with exact contract; Foundation/proxy do not. |
| Runtime matrix | Lifecycle paths | MLX batch/serial/MTP/EAGLE3 and DwarfStar/DSpARK each balance events and exact usage. |
| Admission | MLX wait paths | Deterministic clock/slot tests cover wait success, queue-full rejection, timeout, and cancellation; waiting is observable and each admitted request has one accepted/terminal pair. |
| Admission | DwarfStar wait paths | The erased-provider adapter passes the same success/rejection/timeout/cancel matrix with no server polling or duplicate terminal. |
| Public API compile | Legacy protocol conformers | External conformers implementing only pre-issue `AFMMLXRequestScheduling`, `AFMMLXOpenAIChatGenerating`, and `AFMMLXOpenAIChatServing` requirements compile unchanged; lack of refined `AFMMLXGenerationAdmitting` conformance selects the legacy adapter. |
| Public API compile | `StatsAggregator` surface | `shared`, nested types, bucket constants, every current mutation/registration/observation method, and `snapshot()` compile from an external module. |
| Compatibility behavior | Facade binding/parity | Installed Services collector receives each legacy non-gauge call once; nested snapshot conversion matches Core values; same-target reinstall is harmless, replacement after first use is rejected, unbound facade is zero/no-op. |
| Compatibility behavior | Legacy gauge sampling | Each copied reader runs exactly once per outer facade snapshot with no locks held; samples and persistent compatibility-only batch peak overlay only that facade result; reset clears the compatibility peak; re-entrant snapshot skips callbacks and cannot deadlock; callback registration is absent from built-in/server composition and `/metrics`. |
| Compatibility behavior | Built-in admission enforcement | Built-in MLX/DwarfStar conform to refined admission and cannot fall back to legacy slot polling; an external old conformer preserves old behavior but is marked unqualified. |
| Public API baseline | Core symbol graph | The intentional additive diff is reviewed and committed; `./Scripts/check-afmkit-core-api.sh` passes locally and in mandatory PR/push CI. |
| Compatibility scope | Source rebuild | The external package fixture rebuilds against the branch using exact old initializers/protocol requirements; docs state that precompiled binary-module ABI compatibility is not supported. |

Likely tests:

- New `Tests/MacLocalAPITests/InferenceTelemetryCollectorTests.swift`.
- New `Tests/MacLocalAPITests/MetricsControllerTests.swift` and pinned fixture directory.
- New `Tests/MacLocalAPITests/RawTextGenerationContractTests.swift`.
- New `Tests/MacLocalAPITests/AnyAFMModelRawGenerationTests.swift` plus provider-registry and CLI composition coverage.
- New `Tests/MacLocalAPITests/LegacyCompletionsControllerTests.swift`.
- New `Tests/MacLocalAPITests/GenerationAdmissionTelemetryTests.swift` for MLX and DwarfStar slot-wait handoffs.
- New `Tests/MacLocalAPITests/LegacyMetricsCompatibilityAdapterTests.swift` for callback invocation, overlay isolation, re-entry, and built-in-path prohibition.
- New `Tests/MacLocalAPITests/PublicAPICompatibilityTests.swift`, importing public modules without `@testable`.
- New local package fixture `Tests/CompatibilityFixtures/Issue192LegacyClient/` compiled through `Scripts/swiftpm-reliable.sh` to exercise qualified nested types, exact old initializers, old protocol witnesses, and refined-capability fallback.
- New `Tests/MacLocalAPITests/ModelDiscoveryDeterminismTests.swift`.
- Extend `StreamingUsageChunkTests.swift` and `MLXChatCompletionsControllerStreamingTests.swift`.
- Add DwarfStar provider conformance tests; retain Foundation telemetry tests only to prove it remains unqualified/route-gated.

Feature SwiftPM builds/tests run through `Scripts/swiftpm-reliable.sh` per repository instructions. The public API gate runs through its dedicated `Scripts/check-afmkit-core-api.sh` locally and in the new mandatory macOS CI workflow because that script owns symbol extraction and normalization.

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
| GuideLLM fixed output | Chat/raw with target output tokens | `ignore_eos` reaches provider; generation reaches requested maximum unless explicit stop/context bound intervenes; final exact usage agrees and finish is `length`. |
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
- Canonical engine finish labels and OpenAI wire reasons are separate; `tool_calls` remains wire-only and maps to engine `stop`.
- Provider events use Core protocols; writable ingress rejection/connection events stay in Services behind an AFMServer adapter. Only provider admission can allocate a telemetry token, and Core exposes no server/transport mutation contract.
- Built-in/server metrics are push-based and atomically snapshotted. Legacy live-reader callbacks are sampled only by the deprecated facade adapter, outside locks, with explicit non-atomic overlay and re-entry behavior.
- Compatibility is source-rebuild compatibility for SwiftPM consumers, not binary-module ABI compatibility. Existing protocols remain unchanged and old initializer overloads remain callable.
- Slot waiting is provider-owned from queue insertion through lease cleanup, so waiting/latency metrics include successful, timed-out, and cancelled waits without controller polling.
- Raw completions bypass chat templates by contract and survive `AnyAFMModel` erasure through a Core-owned optional type eraser.
- Chat and text SSE have separate payload/state tests; failures cannot become normal text.
- Logical KV occupancy is an approximation with explicit numerator/denominator and excludes retained cache state.
- Foundation remains unqualified until it has native raw generation and exact usage; estimates are never emitted on a qualified path.
- External GuideLLM/Playground revisions are pinned and recorded to contain CLI/parser drift.
- Pinned GuideLLM `continuous_usage_stats` is accepted with final exact usage authoritative; `ignore_eos` is a provider-neutral stopping option and fixed-output qualification fails if EOS ends a run early.
- Existing AFM metric names, meanings, buckets, and Grafana behavior are preserved; upstream bucket updates are additive.
- Every intentional Core API addition updates `docs/api-baselines/AFMKitCore.symbols.json`; the dedicated check is mandatory locally and in PR/push CI.

## Architecture review verdict and resolution trace

Durable gate: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW.md`, dated 2026-08-17. Verdict for issue 192: **REQUEST CHANGES**. Implementation remains blocked until this revision is re-gated.

| Reviewer requirement | Resolution in this revision |
| --- | --- |
| 1. Correct full vs computed prompt tokens | Added distinct full and computed counters, corrected `vllm:prompt_tokens_total`, and specified exact `F/H/C` cache-hit assertions. |
| 2. Exact speculative key and pinned contract | Replaced the wrong key with `vllm:spec_decode_acceptance_rate`; added pinned name/HELP/TYPE/label/bucket fixtures and classified misses/rejected/failure metrics as AFM additions. |
| 3. Resolve cross-runtime collector ownership | Selected AFMKitCore protocols/immutable snapshot plus AFMKitServices mutable collector, injected into providers and AFMServer; MLX singleton retained only as compatibility facade. |
| 4. Add event ownership table | Defined acceptance, sole owners for every event, nonblocking observer constraints, cancellation behavior, and one-terminal enforcement. |
| 5. Define true raw-prompt contract | Added provider-neutral `AFMRawTextGenerating` using direct raw `UserInput(prompt:)`, exact usage, shared stream/non-stream provider events, and capability-gated route registration. |
| 6. Do not claim Foundation accuracy | Added exact runtime matrix: MLX and DwarfStar contracts are required; Foundation and proxies are explicitly unqualified and receive no silent raw route. |
| 7. Unambiguous prompt arrays | Chose stable rejection of every array shape with HTTP 400/code `unsupported_prompt_array`, before admission/SSE. |
| 8. Separate SSE state machines | Defined distinct chat `delta` and legacy `text` machines, terminal/usage/`[DONE]` ordering, and one post-header error event with no success terminal. |
| 9. Precise KV utilization | Adopted active logical positions divided by `contextWindow * maxConcurrent`, atomic snapshot, no waiting/retained radix contribution, and full lifecycle tests. |
| 10. Exclude release cleanup | Explicitly prohibits release workflow/package cleanup under issue 192. |
| 11. Retain and extend tests | Retained parser, AFM golden, discovery, usage, SSE, concurrency, GuideLLM, and Playground tests; added prompt split, registry, raw bypass, KV, ownership, and runtime-gating coverage. |

At the first-checkpoint revision, the gate-supplied KV definition, approved 10-second window/additive `/metrics`, cross-runtime ownership, and runtime/route behavior were recorded. Later gate sections below supersede the ownership and compatibility details that required further review.

## Architecture review 2 verdict and resolution trace

Second durable gate: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW_2.md`, dated 2026-08-17. Verdict for issue 192: **REQUEST CHANGES**. Feature implementation remains blocked until a third architecture gate approves this plan.

| Gate-2 finding | Resolution in this revision |
| --- | --- |
| 1. Pinned finish labels and wire mapping | Canonical vLLM labels are exactly `stop`, `length`, `abort`, `error`, `repetition`, all pre-created. Wire `tool_calls` maps to engine `stop` without changing the response; engine `repetition` maps to wire `stop`. Fixtures assert the exact label set and counter samples. |
| 2. Raw capability lost through `AnyAFMModel` | Added the planned Core-owned `AnyAFMRawTextGenerator`, retained as an optional property during `AnyAFMModel` initialization. Server route composition consumes that retained capability. Direct erasure, provider registry, DwarfStar CLI, and DSpARK composition tests are specified. |
| 3. AFMServer cannot write through a snapshot source | Added Core-owned writable `AFMServerTelemetryObserving`, implemented by the same Services collector and injected separately from the snapshot source. It owns bounded rejection and active-connection writes; tests prove rejection writes allocate no provider token and do not alter accepted/terminal counts. |
| 4. Pre-provider slot wait conflicts with acceptance metrics | Moved slot wait/reservation into provider-owned queue admission for MLX and the DwarfStar adapter. Queue insertion atomically creates accepted/waiting state; success returns a running lease, while timeout/cancel emits one provider terminal. Pre-insertion queue-full rejection has no token. Deterministic tests cover all handoffs. |

This was the Gate-2 checkpoint. The Gate-3 verdict below supersedes its approval condition.

## Architecture review 3 verdict and resolution trace

Third durable gate: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW_3_ISSUE_192.md`, dated 2026-08-17. Verdict for issue 192: **REQUEST CHANGES**. No production implementation may begin until a new independent architecture gate approves this revision.

| Gate-3 blocker | Resolution in this revision |
| --- | --- |
| 1. Snapshot ownership violates Core/Services dependency direction | Moved the complete immutable `AFMInferenceMetricsSnapshot`/`AFMHistogramSnapshot` value contract and concrete `AFMInferenceMetricsSnapshotSource` return type to AFMKitCore. AFMKitServices owns only mutable collector state and constructs the Core value. AFMServer reads the Core contract; MLX/DwarfStar depend only on Core. Added package-boundary compile tests. |
| 2. Public compatibility/migration is conditional | Chose a non-breaking source-compatible migration. `StatsAggregator` remains a concrete deprecated AFMKitMLX facade with every existing public nested type and method, forwarding through a Core legacy sink to the single Services collector. Existing scheduling/chat requirements and initializers remain; a new optional admitter requirement defaults to `nil`, preserving external conformers through a documented unqualified legacy adapter. Added external-module and local-package compile fixtures plus facade binding/parity tests. |

At the gate-3 checkpoint, target ownership and public migration became normative rather than conditional. The gate-4 decisions below supersede its Core-owned legacy sink and defaulted requirement on the existing scheduling protocol. Implementation remains blocked pending a new independent approval.

## Architecture review 4 verdict and resolution trace

Fourth durable gate: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW_4_ISSUE_192.md`, dated 2026-08-17. Verdict for issue 192: **REQUEST CHANGES**. No production implementation may begin until a new independent architecture gate approves this revised checkpoint.

| Gate-4 finding | Resolution in this revision |
| --- | --- |
| 1. Server/transport responsibilities remain in AFMKitCore | Core now owns only provider-neutral snapshot, observation, admission, finish/failure, generation-option, and raw-generation contracts. Services owns ingress rejection/connection types and mutable writes; AFMServer owns HTTP classification in an adapter and depends explicitly on Services. Boundary tests forbid server/HTTP/legacy callback declarations in Core. |
| 2. Legacy callbacks conflict with atomic snapshots | Built-in/server composition is push-only and `/metrics` reads one atomic collector snapshot. A Services-owned compatibility adapter stores legacy readers, invokes one copied set exactly once outside all locks, overlays only the deprecated facade result, and handles re-entrant snapshots by returning the callback-free base snapshot. Tests cover invocation, overlay isolation, races, re-entry, and built-in-path prohibition. |
| 3. Binary-facing compatibility is undecided | AFMKit is explicitly source-distributed SwiftPM with source compatibility after rebuild and no ABI promise. Existing scheduling/chat protocols remain unchanged through a new refined admission protocol; exact old public initializer declarations are preserved with new overloads rather than replaced by trailing parameters. External fixtures rebuild against those old surfaces. |
| 4. Core symbol baseline gate is absent | The implementation sequence now requires intentional update/review of `docs/api-baselines/AFMKitCore.symbols.json`, local `Scripts/check-afmkit-core-api.sh`, and a new mandatory macOS PR/push CI workflow. Release workflow cleanup remains out of scope. |
| 5. GuideLLM extension semantics are not normative | `continuous_usage_stats` is decoded and accepted while one final exact usage event remains authoritative. `ignore_eos` maps to a new provider-neutral Core stopping option for chat and raw generation and is required across every qualified runtime/mode; deterministic and live fixed-output tests fail qualification on early EOS. |

This revision resolves all gate-4 approval conditions at planning level. Implementation remains blocked until an independent fifth architecture review records **APPROVED**.

## Independent architecture review 5 verdict

Date: 2026-08-17

Reviewed branch: `codex/issue-192-vllm-guidellm`

Reviewed commit: `981af0ad2654de14573858a0347e80476ed9b0e9`

Gate under review: `/Volumes/edata/dev/git/CODEX/agent-traces/maclocal-api-191-192/ARCHITECTURE_REVIEW_4_ISSUE_192.md`

Verdict: **APPROVED**

No blocking architecture findings remain from gate 4. This is a plan approval only; no production or dependency implementation was part of this review.

| Gate-4 requirement | Independent verification and disposition |
| --- | --- |
| Server/HTTP telemetry stays outside AFMKitCore | **Resolved.** The normative ownership section limits Core to immutable snapshot values plus provider observation, admission, finish/failure, generation-option, and raw-generation contracts (`issue-192.md:68-79`). Writable decode/authentication/validation/capacity and connection lifecycle are Services-owned `AFMIngressTelemetryRecording` concerns; AFMServer retains HTTP classification in `AFMServerTelemetryAdapter` and gains an explicit Services dependency. The Core snapshot carries only immutable renderer-neutral values, not Services enums, HTTP status, OpenAI/Vapor DTOs, or writable ingress tokens (`issue-192.md:81-91`). The package-boundary matrix expressly rejects Server, HTTP/OpenAI, ingress, connection, legacy-callback, and Services declarations in Core (`issue-192.md:345-346`). This matches the accepted dependency direction in `Package.swift:105-138`, where Core is dependency-free and Services depends on Core. |
| Atomic collector snapshot versus legacy callbacks | **Resolved.** The plan no longer claims arbitrary callbacks are part of an atomic snapshot. Built-in providers and `/metrics` are push-only and consume one atomic collector copy (`issue-192.md:62`). The deprecated facade has a deliberately non-atomic compatibility overlay: it copies callback references under the callback lock, releases all locks, invokes each copied reader once, then overlays those values on one atomic base snapshot without mutating the collector (`issue-192.md:99-107`). Re-entry returns the callback-free base snapshot, registration races have explicit next-snapshot behavior, and the built-in/server path cannot register or invoke callbacks. The automated matrix covers invocation count, lock freedom, re-entry, compatibility-only peak/reset behavior, overlay isolation, and absence from `/metrics` (`issue-192.md:376-378`). This is a coherent selection of the explicit non-atomic compatibility option allowed by gate 4 while preserving atomic server scrapes. |
| Source versus binary compatibility | **Resolved.** The plan explicitly supports SwiftPM source compatibility after a rebuild and disclaims ABI, precompiled-module, witness-table, and hot-swap compatibility (`issue-192.md:93-96`). It keeps `StatsAggregator` concrete with its existing nested names and full public surface, leaves the existing scheduling/chat protocol requirements unchanged, introduces a separate refined `AFMMLXGenerationAdmitting` capability, and preserves old initializer declarations through overloads (`issue-192.md:97-118`). External-module and local-package fixtures exercise old protocol conformers, qualified nested names, old initializers, and capability fallback (`issue-192.md:374-380`, `391-397`). Because no AFMKitMLX binary-stability claim remains, no AFMKitMLX ABI baseline is required by the gate-4 alternative. |
| Mandatory AFMKitCore API baseline | **Resolved in the implementation contract.** The sequence requires extraction and intentional review/update of `docs/api-baselines/AFMKitCore.symbols.json` in the Core API implementation checkpoint, requires `Scripts/check-afmkit-core-api.sh` locally, and adds a dedicated mandatory macOS PR/push workflow rather than an optional release check (`issue-192.md:324-326`, `379`, `397`, `440`). The script and baseline already exist and are the repository's documented additive-API mechanism (`docs/afmkit-public-api.md:88-128`). Independent execution at the reviewed commit built AFMKitCore but found pre-existing baseline drift for `AFMDownloadProgressUserInfo` introduced by `8692bf4` after baseline refresh `4be517f`. Therefore the first implementation checkpoint must intentionally review and reconcile that existing drift, keep it distinguishable from issue-192 additions, and establish a green baseline before the new CI gate can be claimed as passing. This is implementation evidence, not an unresolved ownership decision. |
| GuideLLM `ignore_eos` and `continuous_usage_stats` semantics | **Resolved.** At pinned GuideLLM `97b3077c05a367599112fd7080082c2d32c14b7e`, text and chat handlers send `stream_options.include_usage=true` plus `continuous_usage_stats=true`, and fixed-output requests send `ignore_eos=true` with `stop=null` (`request_handlers.py:497-510`, `1032-1048`). Their streaming parsers retain the most recent non-empty usage object (`request_handlers.py:566-605`, `1153-1204`), so accepting `continuous_usage_stats` while emitting one final exact usage-only event is compatible and avoids fabricated estimates. The plan now decodes both wire fields, maps `ignore_eos` to a provider-neutral Core stopping option for chat and raw generation, preserves explicit stops, requires EOS exclusion and exact usage in every qualified runtime/mode, and withholds qualification where that cannot be implemented (`issue-192.md:274-283`, `287-295`). Deterministic DTO/provider tests plus the live fixed-output row make early EOS a qualification failure (`issue-192.md:361-362`, `410-414`). |

### Review evidence and implementation gate

- Branch/commit and origin were exact and the worktree was clean before this review.
- The gate-4 document, full revised plan, package target graph, Core public-API policy, current `StatsAggregator` callback behavior, current scheduling/chat protocols, OpenAI request DTOs, and pinned GuideLLM request/stream handlers were inspected.
- `./Scripts/check-afmkit-core-api.sh`: **failed at the reviewed commit** because the checked-in baseline predates existing `AFMDownloadProgressUserInfo` public symbols. Issue 192 must not absorb that drift without explicit review; the approved plan's mandatory baseline checkpoint remains a merge/qualification gate.
- No feature tests or live model workloads were run because this checkpoint reviews architecture and the branch contains plan changes only.
- Production implementation may begin under this approved plan. Each runtime remains unqualified until its focused deterministic and live rows pass, and the Core API/CI gate must be green before merge.

## Implementation trace

### 2026-08-17: Core and Services ownership checkpoint

- Added only provider-neutral immutable telemetry snapshots, bounded finish/failure observations, provider-owned admission leases/type erasure, raw-prompt generation contracts, and `ignoreEndOfSequence` to AFMKitCore. The exact pre-issue `AFMGenerationOptions` initializer remains present and initializes the new policy to `false`; a distinct overload opts into it.
- Extended `AnyAFMModel` to capture `AFMRawTextGenerating` before erasure. Non-conforming models retain `nil` and cannot accidentally acquire a raw completion route.
- Added one mutable `InferenceTelemetryCollector` in AFMKitServices. Provider observations, ingress rejection/connection recording, rolling windows, terminal deduplication, and atomic Core snapshot construction share one lock-owned state. No writable ingress or connection declarations were added to Core.
- Added `LegacyInferenceMetricsCompatibilityAdapter` in Services. It copies callbacks under its callback lock, invokes them exactly once after releasing locks, overlays only its returned Core snapshot, persists a compatibility-only peak, and returns the push-only collector snapshot on same-adapter re-entry.
- Added an external path-dependent SwiftPM fixture that imports AFMKitCore and AFMKitServices as a consumer. Its two lifecycle/ingress ownership tests pass without compiling MLX or mutating dependency checkouts.
- The first root filtered test attempt stopped before test compilation because the untouched `mlx-swift` checkout does not contain the repository's required `MLXFast.deepseekV4SymmetricQ8Matvec` patch. The implementation did not apply or edit that upstream checkout. Main-suite tests remain committed for the repository's normal patched CI/build environment.
- Intentionally reconciled the pre-existing `AFMDownloadProgressUserInfo` baseline drift identified by architecture review 5 together with the reviewed issue-192 additive Core API. `./Scripts/check-afmkit-core-api.sh` now passes. Added a dedicated macOS PR/push workflow; `.github/workflows/release.yml` remains untouched.

### 2026-08-17: Legacy facade and provider admission checkpoint

- Replaced `StatsAggregator` internals with a deprecated concrete AFMKitMLX facade while retaining its existing public nested types, type aliases, methods, defaults, and qualified names for SwiftPM source rebuilds. A new AFMKit-owned composition target forwards the facade to the Services-owned legacy adapter; AFMKitMLX still has no Services dependency.
- The facade accepts one compatibility target before first use. Reinstalling the same object is idempotent, replacing a used binding is rejected, and an unbound facade is a deterministic no-op. Legacy callback sampling remains deliberately non-atomic and compatibility-only; built-in provider and server metrics will use the push observer and atomic collector snapshot instead.
- Preserved arbitrary legacy finish-reason labels in compatibility supplemental counts while retaining the bounded five-value provider finish contract for qualified built-in telemetry.
- Added a refined `AFMMLXGenerationAdmitting` capability without changing the existing `AFMMLXRequestScheduling` requirements. The legacy adapter reports accepted/start/timeout/cancellation observations around the old scheduler API and returns an idempotent Core lease.
- Added provider-neutral request TaskLocals for the telemetry token, accepted timestamp, and `ignoreEndOfSequence`, plus explicit reservation-release transfer for scheduler-owned streaming lifetimes. These additions were intentionally incorporated into `docs/api-baselines/AFMKitCore.symbols.json`; `./Scripts/check-afmkit-core-api.sh` passes.
- The external Core/Services SwiftPM fixture passes both tests. The rewritten facade and AFMKit composition bridge typecheck independently against the built Core/Services modules; full AFMKitMLX compilation remains blocked before these sources by the untouched MLX dependency mismatch recorded above.

### 2026-08-17: MLX provider telemetry and GuideLLM request checkpoint

- Replaced remaining MLX serial and batch inference mutations of the deprecated `StatsAggregator` facade with injected `AFMInferenceTelemetryObserving` lifecycle events. Admission, running/waiting state, prefix-cache evidence, output-token timing, terminal ownership, failures, and bounded finish reasons now flow through the provider-neutral Core contract.
- Added observer-aware MLX factory/model/runtime/service construction while retaining the exact legacy initializers with no-op telemetry defaults for SwiftPM source compatibility. `MLXModelService` now implements the refined admission capability, and request TaskLocals carry the provider admission token and `ignoreEndOfSequence` policy into every qualified serial/batch path.
- Decoded GuideLLM's `ignore_eos` and `stream_options.continuous_usage_stats` extensions. The MLX controller maps `ignore_eos` to the Core generation policy; final exact usage remains authoritative.
- Validation at this checkpoint: 15 focused Release tests passed (`InferenceTelemetryCollectorTests`, `LegacyInferenceMetricsCompatibilityAdapterTests`, `RawTextGenerationContractTests`, and `AFMMLXRuntimeEventsTests`); the external `Issue192TelemetryClient` fixture passed 3/3 tests; and `./Scripts/check-afmkit-core-api.sh` passed.
- The supported patch workflow staged the repository's MLX dependency patches for compilation. The resulting `vendor/mlx-swift-lm` worktree state is intentionally not part of this checkpoint.
- Remaining integration after this checkpoint: create one host-owned collector before runtime construction, inject it into the MLX/DwarfStar provider and AFMServer, migrate server connection/ingress writes and `/metrics` rendering off the deprecated facade, add the raw `/v1/completions` route, and execute the pinned Playground/GuideLLM API qualification matrix.

### 2026-08-17: Shared server collector and vLLM renderer checkpoint

- Added an AFMServer-owned adapter that exposes the immutable Core snapshot source and Services ingress recorder without moving HTTP classification into Core. `Server` now receives that adapter, uses it for middleware connection accounting, configures the process model/capacity, and injects the same snapshot source into `/metrics`.
- Updated the CLI composition roots to construct one `InferenceTelemetryCollector` per MLX or DwarfStar server process. The MLX runtime and server share that collector, and the deprecated `StatsAggregator` facade is installed only as a source-compatibility bridge. DwarfStar currently shares server ingress/snapshot state but still requires provider lifecycle event integration before its engine-level metrics can be qualified.
- Replaced the `/metrics` renderer's direct `StatsAggregator` dependency with one immutable `AFMInferenceMetricsSnapshot`. Existing `afm:*` families retain their established computed-prefill semantics, while pinned `vllm:*` families expose full prompt totals/histograms, rolling throughput, logical KV occupancy, prefix-cache hit rate, speculation accounting, the exact five bounded finish labels, and canonical latency histograms.
- Preserved request-level legacy radix hit/miss counters separately from token-level vLLM prefix query/hit counters. Added deterministic renderer tests for the full/computed prompt split, label escaping, zero-series precreation, cache/speculation arithmetic, rejection/connection metrics, histogram values, and the complete pinned vLLM family registry.
- Validation: `Scripts/swiftpm-reliable.sh build -c release --target AFMCLI` passed; 10 focused Release tests passed (`MetricsControllerTests`, `InferenceTelemetryCollectorTests`, and `LegacyInferenceMetricsCompatibilityAdapterTests`); the external `Issue192TelemetryClient` fixture passed 3/3 in Release; and `Scripts/check-afmkit-core-api.sh` passed. Existing dependency exclude and unrelated Swift concurrency warnings remain unchanged.
- Remaining after this checkpoint: migrate the six controller-owned active-connection writes off `StatsAggregator`, classify pre-provider ingress rejections, integrate DwarfStar provider lifecycle telemetry, add the capability-gated raw `/v1/completions` route and separate text SSE state machine, then run the pinned Playground/GuideLLM live qualification matrix.

### 2026-08-17: Controller ingress and connection migration checkpoint

- Removed direct `StatsAggregator` writes from both Foundation Models and MLX chat controllers. Each controller now receives the process-owned `AFMServerTelemetryAdapter`; default standalone adapters preserve existing source-level controller construction in focused tests and downstream rebuilds.
- The active-connection middleware now observes `/v1/chat/completions` while Vapor parses the request and constructs its response, so non-streaming requests are no longer invisible. Streaming response bodies acquire a separate idempotent adapter token for their asynchronous lifetime after response construction. `/metrics`, health, documentation, and existing batch-stream exclusions remain unchanged.
- Classified empty-message, model-selection, `top_logprobs`, malformed-body, and provider-capacity failures into the bounded Services-owned ingress rejection reasons. No provider request token is allocated for these pre-provider failures.
- Validation: 25 focused Release tests passed (`MetricsControllerTests`, `InferenceTelemetryCollectorTests`, and `MLXChatCompletionsControllerStreamingTests`), including concurrent streaming isolation, structured output, tool-call serialization, rejection handling, and middleware route classification. `git diff --check` passed and no direct `StatsAggregator` references remain in AFMServer; the two AFMCLI references install only the deliberate deprecated compatibility target.
- Remaining after this checkpoint: authenticate-rejection wiring if an authentication middleware is present, DwarfStar provider lifecycle telemetry, capability-gated raw `/v1/completions` with a separate text SSE serializer, and the pinned live Playground/GuideLLM qualification matrix.
