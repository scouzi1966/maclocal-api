# mlx-swift-lm Patch Ownership

maclocal-api keeps compatibility snapshots for an older, pinned
`mlx-swift-lm` revision in `Scripts/patches/`. AFMKit owns the current MLX
provider sources and its vendored `mlx-swift-lm` tree. The patch catalog is
therefore a compatibility mechanism for the pinned consumer branch, not an
alternate source of truth for AFMKit.

`Scripts/apply-mlx-patches.sh` is the authoritative mapping between patch
files and their destination paths.

## Patched libraries

Only three `mlx-swift-lm` libraries contain directly patched source files:

| Library | Patched files |
| --- | --- |
| `MLXLLM` | `LLMModelFactory.swift`, `DeepseekV3.swift`, `GLM4MOELite.swift`, `NemotronH.swift`, `Qwen3Next.swift`, `GatedDelta.swift`, `Qwen3_5MoE.swift`, `MiniMaxM2.swift`, `GLM5MoeDsa.swift`, `KimiK25.swift`, `Qwen4Exp.swift` |
| `MLXLMCommon` | `Evaluate.swift`, `Load.swift`, `Tokenizer.swift` |
| `MLXVLM` | `Qwen3VL.swift`, `VLMModelFactory.swift`, `Qwen3_5MoEVL.swift`, `Qwen4ExpVL.swift` |

The following libraries are not directly patched:

- `BenchmarkHelpers`
- `IntegrationTestHelpers`
- `MLXCXGrammar`
- `MLXEmbedders`
- `MLXFoundationModels`
- `MLXGuidedGeneration`
- `MLXHuggingFace`
- `MLXHuggingFaceMacros`
- `MLXRerankers`

## Targets outside Libraries

The catalog also patches the root `Package.swift` to add `MLXLLM` as an
`MLXVLM` dependency and supplies two files under `Tests/MLXLMTests/`:

- `Qwen4ExpTests.swift`
- `SamplerTests.swift`

## Relationship to AFMKit

All 21 patch-catalog target paths exist in AFMKit's vendored
`mlx-swift-lm`. At maclocal-api commit `8129216` and AFMKit commit `ec3a311`,
the comparison is:

| Comparison | Files |
| --- | --- |
| Exact copies | `DeepseekV3.swift`, `GLM4MOELite.swift`, `Qwen3Next.swift`, `MiniMaxM2.swift`, `GLM5MoeDsa.swift`, `KimiK25.swift`, `Qwen4Exp.swift`, `Qwen4ExpVL.swift`, `Qwen4ExpTests.swift`, `SamplerTests.swift` |
| Present in AFMKit but different | `Package.swift`, `LLMModelFactory.swift`, `NemotronH.swift`, `Evaluate.swift`, `Load.swift`, `Tokenizer.swift`, `Qwen3VL.swift`, `VLMModelFactory.swift`, `GatedDelta.swift`, `Qwen3_5MoE.swift`, `Qwen3_5MoEVL.swift` |

The differing files are expected: the maclocal-api snapshots target the older
pinned `mlx-swift-lm` revision, while AFMKit contains newer integrated source.
Do not copy the older patch snapshots wholesale over AFMKit. Make provider
changes in AFMKit first, then update only the compatibility snapshots required
by the pinned consumer branch.
