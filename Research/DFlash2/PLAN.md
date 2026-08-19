# DFlash 2 Implementation Plan

Status: implementation and compile/unit checkpoint complete; live model matrix pending coordination, 2026-08-19

## Objective

Add opt-in DFlash 2 speculative decoding to maclocal-api while preserving the
existing autoregressive, MTP, EAGLE3, DSpARK, Qwen 3.8, and Muse Glimmer paths.
The implementation must detect and validate checkpoints from metadata, expose
deterministic fallback/error behavior, and report provider-neutral speculative
telemetry without putting DFlash 2 model or kernel details in AFMKitCore.

## Implementation Status

Implemented and pushed:

- strict `DFlash2DraftModel` metadata/config and target-shape validation;
- Qwen 3.8 and Muse Glimmer target hidden-state/cache adapters;
- one-pass draft, selector, block-local two-tap dynamic convolutions, greedy
  target verification, rollback/replay, cancellation, and monotonic phase timing;
- local/Hub drafter resolution with existing download progress;
- `--dflash2`, `--dflash2-block`, and `--dflash2-required` startup controls;
- provider-neutral AFMKit startup/request contracts and OpenAPI schema;
- serial streaming/non-streaming service integration and neutral metrics;
- reproducible vendor overlay/drift checks and focused losslessness tests.

Deliberately deferred behind deterministic fallback/error policy: sampling,
explicit string stops, tools/grammar/logprobs, speculative prefix snapshots, and
concurrent/batched DFlash 2. No heavy live model run or local speed claim has
been made.

## Current Inventory

| Concern | Current path | Relevant behavior |
| --- | --- | --- |
| Stable provider contracts | `Sources/AFMKitCore/AFMCoreTypes.swift` | Already has `.speculativeDecoding`, request/response metadata, `.metadata` events, usage, cancellation finish reason |
| AFMKit facade | `Sources/AFMKit/AFMEngine.swift` | `EngineConfig` has MTP and EAGLE3-specific fields; maps into MLX provider configuration |
| MLX configuration/lifecycle | `Sources/AFMKitMLX/AFMMLXRuntime.swift` | Applies provider configuration to `MLXModelService`; owns model load and scheduler startup |
| Speculative policy | `Sources/AFMKitMLX/AFMMLXSpeculativeDecoding.swift` | Modes are off/auto/MTP/EAGLE3; fast path is greedy, text-only, no reasoning/modifiers/stops |
| Runtime bridge | `Sources/AFMKitMLX/AFMMLXRuntimeAdapter.swift` | Runtime enum and execution bridge cover MTP and EAGLE3 only |
| Main MLX service | `Sources/AFMKitMLX/Models/MLXModelService.swift` | Loads MTP sidecars; installs EAGLE3; has separate streaming/non-streaming speculative paths; batch scheduler uses AR |
| Model resolution/download | `MLXModelService.ensureLoaded`, `downloadModel`, `MLXCacheResolver` | Hub download progress/stages and resumable cache resolution already exist; MTP has an auxiliary-repository resolver |
| CLI | `Sources/AFMCLI/main.swift` | `--mtp`, `--mtp-model`, `--eagle3`; DSpark is restricted to DwarfStar; startup config flows through `AFMMLXRuntimeConfiguration` |
| OpenAI request | `Sources/AFMOpenAICompat/OpenAIRequest.swift` | Supports sampling, reasoning, tools, constraints, and template kwargs; no speculative request object yet |
| HTTP execution | `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift` and `AFMLocalClient.swift` | Both streaming and non-streaming call the shared service |
| Prefix cache | `MLXModelService` and `BatchScheduler` | Serial radix cache and batch cache are AR-oriented; speculative paths currently bypass cache reuse |
| Batch/concurrency | `BatchScheduler.swift` | Concurrent mode routes through batched AR; existing MTP/EAGLE3 fast paths are serial only |
| Cancellation | service speculative stream and normal generation tasks | SSE termination cancels the task; token callbacks observe cancellation |
| Metrics | `Sources/AFMKitMLX/Models/StatsAggregator.swift` | Request/token/timing/cache metrics exist; no neutral speculative counters or phase timings |
| DSpARK | `Sources/AFMKitDwarfStar/*`, `CDwarfStar`, DS4 vendor | Different GGUF/fixed-schedule runtime with its own support model and scheduler; not a reusable DFlash runtime |
| Vendor workflow | `Scripts/patches`, `Scripts/apply-mlx-patches.sh`, `Scripts/check-mlx-source-selection.sh` | Reproducible source overlays are applied to `vendor/mlx-swift-lm`; URL consumers use a pinned pre-patched fork |

There is no existing DFlash runtime in this checkout. DFlash 2 therefore
requires a new MLX runtime primitive. It can reuse the general auxiliary-model
resolution, request orchestration, cancellation, output parsing, and telemetry
contracts, but not the MTP, EAGLE3, or DSpARK draft/verify implementation.

## Implementation Sequence

### 1. Contract and policy layer

- Add a metadata-driven DFlash draft descriptor/parser in AFMKitMLX.
- Recognize `architectures: ["DFlash2DraftModel"]` exactly for DFlash 2.
- Preserve legacy DFlash recognition as a separate descriptor version.
- Validate required config fields and tensor-shape expectations before model load.
- Match target and drafter by architecture/config dimensions, tokenizer IDs,
  vocabulary, target layer count, hidden size, context/rope contract, and target
  feature taps. Repository names are diagnostics only.
- Keep DFlash 2 opt-in. Required mode fails closed; preferred mode may fall back
  before emission; disabled mode never downloads or activates it.

### 2. Stable AFMKit boundary

- Use the existing `AFMModelCapabilities.speculativeDecoding` capability.
- Add provider-neutral configuration and telemetry types only; do not mention
  selectors, dynamic convolution, mask IDs, or DFlash tensor layouts in
  AFMKitCore.
- Keep concrete checkpoint resolution and model loading in AFMKitMLX.
- Keep CLI flags, HTTP decoding, download policy, startup diagnostics, and
  metrics export in maclocal-api layers.
- See `ARCHITECTURE.md` for proposed API deltas and compatibility strategy.

### 3. MLX runtime

- Add the DFlash 2 draft model, target feature capture adapter, one-pass draft,
  candidate selector, block-local two-tap dynamic convolution, lossless
  verification, rollback, and cancellation to the supported MLX vendor patch
  set or AFMKitMLX as ownership dictates.
- Use the checkpoint's selector/convolution/block metadata; do not hard-code
  Qwen/Muse repository names.
- Implement Qwen 3.8 and Muse target adapters from target architecture metadata.
- Return a neutral per-cycle record: proposed draft tokens, accepted draft
  tokens, emitted tokens (including verifier bonus), draft/verify/commit time.
- Do not silently fall back after any output has been emitted.

### 4. Orchestration

- Add `--dflash2 <repo-or-path>`, `--dflash2-block`, and
  `--dflash2-required`. Keep the default off.
- Resolve explicit local paths and Hub repositories with normal progress/stage
  reporting. Do not infer a drafter repository from the target name.
- Add OpenAI-compatible extension controls under a structured
  `speculative_decoding` object. Request-level disable is allowed; enabling or
  switching auxiliary checkpoints after startup is rejected deterministically.
- Reject conflicting startup modes (MTP/EAGLE3/DFlash2, DwarfStar/DSpark,
  incompatible concurrency settings) with actionable diagnostics.
- Route unsupported request features according to an explicit policy: required
  DFlash 2 errors before generation; preferred mode falls back before generation
  and emits a reason.

### 5. Feature integration

- Streaming and non-streaming must share one DFlash 2 token loop and summary.
- Reasoning and tool parsing remain downstream of token generation, but DFlash 2
  eligibility must be tested with each mode rather than inheriting the current
  MTP/EAGLE3 blanket fallback.
- Prefix caching is disabled for DFlash 2 until a snapshot includes all target
  and draft state; advertise a neutral fallback reason rather than restoring an
  AR-only cache.
- Concurrent/batch requests use the existing AR scheduler until a row-aligned
  DFlash 2 batch primitive is validated. Explicit required mode must reject this
  conflict rather than quietly use AR.

### 6. Verification and checkpoints

- Unit tests: descriptor parsing, compatibility, tensor contract, conflicts,
  request decoding, fallback policy, telemetry aggregation.
- Contract tests: AFMKit metadata/events, CLI help/config flow, OpenAI JSON,
  streaming/non-streaming equivalence, cancellation.
- Vendor tests: clean application, idempotent `--check`, expected upstream hash,
  compile fixture, and drift failure.
- Integration tests: tiny synthetic draft/target fixtures where practical.
- Compile and focused unit tests first. Do not run long Qwen/Muse inference until
  coordination confirms the AFMKit usability study is not loading Qwen.
- Run the live matrix in `TEST_MATRIX.md` only after that coordination point.
- Commit/push separate checkpoints for research, runtime/orchestration, vendor
  patch, and tests. Never force-push.

## Exit Criteria

- DFlash 2 is off by default and cannot activate from a repository name alone.
- Released Qwen and Muse draft configs pass strict compatibility validation;
  intentionally mismatched fixtures fail before allocating model weights.
- Existing MTP, EAGLE3, DSpARK, normal MLX, Qwen 3.8, and Muse tests pass.
- Streaming and non-streaming produce target-equivalent output under the
  documented greedy methodology. Seeded sampling remains gated until rejection
  sampling is ported and distribution-tested.
- Telemetry reports counts and timings without a speedup claim.
- Live performance claims, if any, compare the same target checkpoint,
  quantization, prompt set, generation parameters, and concurrency on the same
  machine, with warmups and raw evidence retained.
