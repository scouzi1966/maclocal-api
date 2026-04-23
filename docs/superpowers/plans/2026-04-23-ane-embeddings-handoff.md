# Handoff: ANE Embeddings (Apple NL path)

Date: 2026-04-23
Branch: `feat/embeddings`
Repo: `/Users/jesse/GitHub/maclocal-api`

## Scope

Initial PR ships **Apple NaturalLanguage contextual embeddings only**. MLX embedding support is deferred to a follow-up branch once the MLX integration details are worked out.

## Current State

Apple NL embeddings path is implemented and working:

- `afm embed` starts a dedicated embeddings server
- `GET /v1/models` lists the running model
- `POST /v1/embeddings` works for string and array-of-strings input
- `encoding_format=base64` works
- `dimensions` truncation + L2 renormalization works
- `Scripts/test-embeddings.sh` passes for the Apple path

Token-array inputs (`[1,2,3]`, `[[1,2,3],[4,5,6]]`) are decoded at the API layer but always return 400 in this PR because the Apple backend is text-only. The decoder accepts them to keep the client contract stable for when an MLX backend lands.

## Key Files

Implementation:

- `Sources/MacLocalAPI/EmbeddingsCommand.swift`
- `Sources/MacLocalAPI/Controllers/EmbeddingsController.swift`
- `Sources/MacLocalAPI/Models/EmbeddingBackend.swift`
- `Sources/MacLocalAPI/Models/EmbeddingModelRegistry.swift`
- `Sources/MacLocalAPI/Models/NLContextualEmbeddingBackend.swift`
- `Sources/MacLocalAPI/Models/OpenAIRequest.swift`
- `Sources/MacLocalAPI/Models/OpenAIResponse.swift`

Plan / spec:

- `docs/superpowers/specs/2026-04-23-ane-embeddings-design.md`
- `docs/superpowers/plans/2026-04-23-ane-embeddings.md`

Tests / scripts:

- `Tests/MacLocalAPITests/EmbeddingModelRegistryTests.swift`
- `Tests/MacLocalAPITests/EmbeddingsControllerTests.swift`
- `Scripts/test-embeddings.sh`
- `Scripts/test-embeddings-openai.py`

## Verified Commands

- `swift build`
- `swift test --filter EmbeddingModelRegistryTests` — 3/3 pass
- `swift test --filter EmbeddingsControllerTests` — 9/9 pass
- `swift run afm embed --list-models`
- `./Scripts/test-embeddings.sh --port <port>`

Previously verified manually: a live Apple server returned real embeddings for `apple-nl-contextual-en`.

## Deferred: MLX Backend

Deferred from this PR. When picking it back up:

- Re-add the `MLXEmbedders` product dependency in `Package.swift`.
- Reintroduce an MLX backend conforming to `EmbeddingBackend` (including token-ID support).
- Add an MLX case to `EmbeddingBackendKind` and MLX resolution logic to `EmbeddingModelRegistry`.
- Restore the `--backend mlx` CLI flag and MLX branch in `EmbeddingsCommand.run()`.
- Validate end-to-end against a cached `mlx-community` embedding model (e.g. `models--mlx-community--bge-*` or similar).
- Re-enable the `--with-mlx` mode in `Scripts/test-embeddings.sh`.

## Recommended Next Steps

1. Land this PR as the default local embeddings implementation.
2. Open a follow-up branch for the MLX backend once a cached `mlx-community` embedding model is available for validation.
