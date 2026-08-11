# AFMKit Public API Policy

`AFMKitCore` is the stable contract shared by Vesta, maclocal-api, and external
provider packages. It must remain free of MLX, XGrammar, Vapor, Foundation
Models, CoreAI, and vendor SDK dependencies.

The package exposes dependency-scoped products:

- `AFMKitCore`: provider contracts, registry, events, and concurrency helpers.
- `AFMOpenAICompat`: OpenAI transport DTOs without a model runtime.
- `AFMKitMLX`: MLX loading, generation, scheduling, caching, and grammar support.
- `AFMKitFoundationModels`: Apple's Foundation Models service and schema bridge.
- `AFMKitFoundationModels27`: macOS 27 adapters that expose AFMKit models through
  Apple's `LanguageModel` and `LanguageModelExecutor` protocols.
- `AFMKitServices`: Apple Vision, Speech, synthesis, and NaturalLanguage embeddings.

`AFMKit` remains the compatibility umbrella during migration. It re-exports
these products so existing consumers do not need immediate import changes.
New libraries should import only the products they use.

## macOS 27 Provider Contract

The primary extension surface on macOS 27 is Apple's own `LanguageModel` and
`LanguageModelExecutor` protocol pair. AFMKit does not replace that contract.
It supplies reusable public adapters so a Swift package can implement an
execution engine while retaining the behavior expected by
`LanguageModelSession`:

- `AFMFoundationModelsRequestAdapter` converts Apple transcripts, tool schemas,
  sampling options, reasoning levels, and structured-output schemas.
- `AFMFoundationModelsExecutorBridge` converts portable AFMKit generation events
  into Apple's response, reasoning, tool-call, usage, metadata, and completion
  channel events.
- `AFMFoundationModelsModelConfiguration` declares the small amount of model
  configuration needed by the shared request adapter.

AFMKit includes `MLXLanguageModel` and `DwarfStarLanguageModel` as concrete
examples. Applications use either one with Apple's API directly:

```swift
import AFMKitFoundationModels27
import FoundationModels

let model = DwarfStarLanguageModel(
    modelPath: "/path/to/deepseek-v4-flash.gguf"
)
let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Use the available tools if needed.")
```

A third-party provider follows the same pattern: define a `LanguageModel`,
implement its `LanguageModelExecutor`, translate the Apple generation request
with the public request adapter, and send typed events through the public
executor bridge. This preserves source-level compatibility with Apple's macOS
27 framework while allowing the inference engine to be replaced independently.

## MLX Provider

Applications register `AFMMLXProviderFactory` with an `AFMProviderRegistry`,
then construct models by provider and model ID. `AFMMLXModelDescriptor` reports
cached-model capabilities, context length, privacy boundary, and whether the
weights require network access. `AFMMLXModel` owns loading, generation,
reasoning and tool event separation, cancellation, and unload behavior.

```swift
let registry = AFMProviderRegistry()
try registry.register(AFMMLXProviderFactory())
let model = try registry.makeModel(
    providerID: AFMMLXProviderFactory.providerID,
    modelID: "mlx-community/Qwen3-4B-4bit"
)
```

## Contract Overview

Providers expose an `AFMProviderDescriptor` and create type-erased
`AnyAFMModel` values through `AFMProviderFactory`. Models accept
`AFMRequest` values and either return an `AFMModelResponse` or stream
`AFMGenerationEvent` values for response text, reasoning, tool calls,
metadata, usage, and completion.

`AFMProviderRegistry` is optional application infrastructure. A macOS 27
application may construct a concrete Apple `LanguageModel` directly and use
`LanguageModelSession` without registering it.

## Compatibility Rules

- Adding a provider must not require editing `AFMEngine`.
- Portable requests and events must not expose OpenAI transport DTOs.
- Optional products may depend on `AFMKitCore`; `AFMKitCore` must never depend
  on an optional product.
- macOS 27 providers map the portable contract to Apple's `LanguageModel` and
  `LanguageModelExecutor` APIs rather than replacing those protocols.
- New enum cases and protocol requirements require an explicit compatibility
  review.
- Removing, renaming, or changing a public declaration requires a major AFMKit
  version.
- Additive public API changes require updating the checked-in symbol graph in
  the same commit.

## Additive API Changes

### Provider-routed log probabilities

The provider-contract migration adds optional `logprobs` and `topLogprobs`
generation controls, portable `AFMTopLogProbability` and
`AFMTokenLogProbability` values, and `AFMModelResponse.tokenLogprobs`.
Existing initializers remain source-compatible because the new arguments have
defaults. Providers that do not advertise or implement log probabilities may
leave the controls and response field unset.

## API Baseline

Run:

```bash
./Scripts/check-afmkit-core-api.sh
```

The script builds only `AFMKitCore`, extracts its public symbol graph with the
selected Xcode toolchain, and compares it byte-for-byte with
`docs/api-baselines/AFMKitCore.symbols.json`.

When an API change is intentional, review the generated graph at
`.build/api-current/AFMKitCore.symbols.json`, replace the baseline, and explain
the compatibility impact in the commit or pull request.
