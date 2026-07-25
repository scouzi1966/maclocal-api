# AFMKit Public API Policy

`AFMKitCore` is the stable contract shared by Vesta, maclocal-api, and external
provider packages. It must remain free of MLX, XGrammar, Vapor, Foundation
Models, CoreAI, and vendor SDK dependencies.

`AFMKit` remains the compatibility product during migration. It re-exports
`AFMKitCore`, so existing `import AFMKit` consumers can adopt the portable
types without changing imports immediately.

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
