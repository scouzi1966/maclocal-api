# DFlash 2 Architecture

## Verified Algorithm Contract

DFlash 2 remains a one-pass block-diffusion drafter. All draft positions are
predicted in parallel. It adds two independent mechanisms:

1. A selector keeps the top candidates at each position, scores all adjacent
   predecessor/successor pairs in parallel using a context-gated rank-256
   bilinear score, then walks the precomputed scores to choose a path. Sampling
   uses rejection sampling to preserve the target distribution.
2. A block-local, stateless, dynamic depthwise convolution with two taps mixes
   each position with its predecessor. The first draft position reads the last
   verified token. A convolution is placed before and after attention and MLP
   sublayers in every draft layer.

The article reports the selector and convolution together add about 1.3% to a
matched five-layer DFlash draft/verify cycle and improve mean acceptance length
by 21% in its Qwen3.5-4B experiment. Those are upstream measurements, not
maclocal-api or Apple-Silicon performance claims.

## Released Checkpoint Contracts

Both released repositories contain only `config.json`, one BF16
`model.safetensors`, a card, and an image. They are auxiliary models, not
standalone language models.

| Field | Qwen 3.8 drafter | Muse Glimmer drafter |
| --- | ---: | ---: |
| `architectures[0]` | `DFlash2DraftModel` | `DFlash2DraftModel` |
| `model_type` | `qwen3` | `qwen3` |
| hidden size | 5120 | 6656 |
| draft layers | 5 | 5 |
| target layers | 64 | 52 |
| target feature layers | 5, 19, 33, 47, 61 | 1, 13, 25, 37, 49 |
| vocabulary | 248320 | 202048 |
| mask token | 248070 | 201818 |
| checkpoint block size | 8 | 16 |
| selector top-k/rank | 16 / 256 | 16 / 256 |
| convolution kernel/group | 2 / 16 | 2 / 16 |
| rope theta | 10000000 | 500000 |
| sliding window | 2048 | 2048 |
| max positions | 262144 | 131072 |
| final logit soft cap | absent | 20.0 |
| output multiplier | absent | 0.19611613513818404 |
| weight bytes | 3,848,817,896 | 5,544,328,424 |
| revision inspected | `dedf8df68adf...` | `8336acb8dc9...` |

Each checkpoint has 81 tensors. Shared DFlash 2-specific tensors include:

- `candidate_selector.hidden_projection.weight`: `[256, hidden_size]`
- predecessor/successor codebooks: `[vocab_size, 256]`
- `fc.weight`: `[hidden_size, hidden_size * 5]`, fusing five target features
- for every draft layer and each attention/MLP convolution:
  - `base_kernel`: `[2, 2, hidden_size]`
  - `kernel_projection.weight`: `[hidden_size / 4, hidden_size]`

The target configs are not named the same as the draft config. Qwen 3.8's
published target is a top-level `qwen3_5` conditional-generation model with a
`qwen3_5_text` stack. Muse is a top-level `muse_glimmer` model with a
`muse_glimmer_text` stack. Compatibility must therefore compare normalized
target text metadata and dimensions, not require identical `model_type` values.

The checkpoint block size is not automatically the server's draft-token count.
The Qwen card describes block size 8 as seven drafted tokens plus a verifier
token, while one SGLang example passes 8 and vLLM passes 7. oMLX exposes a
separate runtime block setting and the article recommends 5 for its Qwen demo.
The runtime API must name and validate these quantities unambiguously.

## Runtime Primitive

DFlash 2 cannot reuse an existing draft primitive in this checkout:

- MTP chains target-specific prediction heads autoregressively and has different
  cache/rollback state.
- EAGLE3 uses an autoregressive feature drafter and Gemma-specific runtime.
- DSpARK is implemented in the DwarfStar GGUF/fixed-schedule runtime and uses a
  support model, confidence pruning, and a different scheduler.
- There is no current DFlash implementation.

It can reuse orchestration concepts: auxiliary resource resolution, model load
progress, serial generation ownership, token callbacks, cancellation, output
parsers, fallback policy, and telemetry aggregation.

The implemented vendor/AFMKitMLX primitive uses these internal interfaces:

```swift
struct AFMMLXDFlash2Configuration
struct AFMMLXSpeculativeTelemetry

protocol DFlash2Target {
    func dflash2Forward(...)
    func dflash2CaptureCache(...)
    func dflash2RestoreCache(...)
}

final class DFlash2DraftModel
final class DFlash2Generator
```

Qwen and Muse adapters are model extensions in the supported MLX vendor patch.
Selector, convolution, draft attention/MLP, tensor loading, and lossless greedy
verification stay inside that patch; orchestration stays in AFMKitMLX.

The current correctness-first generator snapshots target cache state before
verification, restores it, and replays only the committed verifier token plus
accepted draft prefix. It recomputes draft context instead of maintaining the
reference runtime's optimized draft KV cache. This establishes correctness but
is not evidence of the article's cycle-latency result.

## AFMKit Boundary

### Stable, provider-neutral concepts

- A model can support speculative decoding.
- A caller can disable, prefer, or require speculative decoding.
- A strategy has a stable identifier, but providers decide which identifiers
  they implement.
- A request may fall back before emission for a neutral reason.
- A cycle proposes and accepts tokens and has draft/verification timing.
- A completion can summarize proposal count, accepted draft tokens, emitted
  tokens, cycle count, acceptance length, and phase timing.

### AFMKitCore must not contain

- DFlash 2 config keys, architecture names, tensor names, mask IDs, target layer
  taps, selector ranks/top-k, convolution parameters, kernels, or model classes.
- Hugging Face repository IDs or download/cache policy.
- MLX arrays, caches, model protocols, or scheduling details.

### Implemented source-compatible API delta

- `AFMKit.AFMSpeculativeDecodingConfiguration`: provider-neutral startup mode,
  drafter resource, maximum draft tokens, and requirement.
- `AFMOpenAICompat.SpeculativeDecodingOptions`: the same request concepts under
  `speculative_decoding`.
- `AFMKitMLX.AFMMLXSpeculativeTelemetry`: provider-neutral counts and phase
  timing returned by the MLX bridge.
- `StatsAggregator`/Prometheus: neutral drafted, accepted, cycle, strategy, and
  phase-time counters.

No source in `AFMKitCore` was changed. The concrete DFlash descriptor, config,
runtime enum, target adapters, model, and generator remain in AFMKitMLX or the
patched MLX dependency.

### Proposed future AFMKitCore API delta

This is a proposed source-compatible addition; implementing it in a separately
versioned AFMKit release should include a public API baseline update.

```swift
public struct AFMSpeculativeStrategyID: RawRepresentable, Hashable, Codable, Sendable {
    public let rawValue: String
}

public enum AFMSpeculativeDecodingPolicy: String, Codable, Hashable, Sendable {
    case disabled
    case preferred
    case required
}

public struct AFMSpeculativeDecodingOptions: Hashable, Codable, Sendable {
    public var policy: AFMSpeculativeDecodingPolicy
    public var strategy: AFMSpeculativeStrategyID?
    public var maximumDraftTokens: Int?
}

public enum AFMSpeculativeFallbackReason: String, Codable, Hashable, Sendable {
    case disabled, unavailable, incompatibleModel, incompatibleRequest
    case concurrency, prefixCache, runtimeFailure
}

public struct AFMSpeculativeDecodingTelemetry: Hashable, Codable, Sendable {
    public var strategy: AFMSpeculativeStrategyID
    public var draftTokens: Int
    public var acceptedDraftTokens: Int
    public var emittedTokens: Int
    public var verificationCycles: Int
    public var draftDurationSeconds: Double
    public var verificationDurationSeconds: Double
    public var totalCycleDurationSeconds: Double
    public var fallbackReason: AFMSpeculativeFallbackReason?
}
```

Add `speculativeDecoding` to `AFMGenerationOptions` only when all providers can
ignore `.preferred` safely and reject unsupported `.required` consistently.
Until then, maclocal-api can transport this through `AFMRequest.metadata` and
emit telemetry through the existing `.metadata` event using versioned keys.
Adding a new `AFMGenerationEvent` enum case is deferred because exhaustive
downstream switches would be source-breaking even though the enum is not frozen.

### AFMKitMLX responsibilities

- Config parsing, compatibility reports, runtime strategy selection.
- DFlash 2 model/tensor implementation and target adapters.
- Draft/verify loop, cache state, lossless greedy verification, and cancellation
  checkpoints. Distribution-preserving sampling is deferred.
- Translation from internal cycle records to neutral AFM telemetry.

### maclocal-api server/CLI responsibilities

- Flags, OpenAI extension decoding, startup conflict diagnostics.
- Hub repository/path resolution, downloads, progress, cache visibility.
- Request policy, fallback/error mapping, streaming transport.
- Prometheus/log exposure and benchmark scripts.

## Request and Fallback Semantics

- Startup default: disabled.
- Explicit preferred startup + missing/incompatible drafter: log the diagnostic
  and keep AR available; `--dflash2-required` fails startup.
- Request `disabled`: use normal generation even when a runtime is loaded.
- Request `preferred`: use DFlash 2 if eligible; otherwise fall back before
  emitting output and report the reason.
- Request `required`: reject before generation when ineligible.
- Setup or request ineligibility before output: preferred may fall back once;
  required errors. A generator execution failure is surfaced rather than
  replayed through AR.
- Runtime failure after output: error/cancel the request; never restart with AR.
- Batch/concurrent scheduler: AR fallback for preferred, deterministic conflict
  for required, until a batch primitive is implemented.
- Prefix cache: AR fallback for preferred and conflict for required until a
  complete target+draft snapshot format exists.

## Telemetry Definitions

- `draftTokens`: candidate path tokens proposed to verification; exclude the
  already verified anchor and verifier bonus token.
- `acceptedDraftTokens`: proposed tokens accepted from the draft.
- `emittedTokens`: accepted draft tokens plus target verifier/bonus tokens that
  become output.
- `verificationCycles`: target verification passes.
- `meanAcceptedDraftLength`: `acceptedDraftTokens / verificationCycles`.
- `meanCommittedTokensPerCycle`: `emittedTokens / verificationCycles`, when an
  event consumer needs verifier-token-inclusive throughput.
- Timings use monotonic clocks and synchronize only at existing correctness
  boundaries. Instrumentation must not add per-token host synchronization.
