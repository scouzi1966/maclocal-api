# AFMKit Public API Policy

`AFMKitCore` is the stable contract shared by Vesta, maclocal-api, and external
provider packages. It must remain free of MLX, XGrammar, Vapor, Foundation
Models, CoreAI, and vendor SDK dependencies.

The package exposes dependency-scoped products:

- `AFMKitCore`: provider contracts, registry, events, and concurrency helpers.
- `AFMOpenAICompat`: OpenAI transport DTOs without a model runtime.
- `AFMKitMLX`: MLX loading, generation, scheduling, caching, and grammar support.
- `AFMKitFoundationModels`: Apple's Foundation Models service and schema bridge.
- `AFMKitServices`: Apple Vision, Speech, synthesis, and NaturalLanguage embeddings.

`AFMKit` remains the compatibility umbrella during migration. It re-exports
these products so existing consumers do not need immediate import changes.
New libraries should import only the products they use.

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
