# Handoff: ANE Embeddings — feat/ane-embeddings

Date: 2026-04-23
Branch: `feat/ane-embeddings`
Repo: `/Users/jesse/GitHub/maclocal-api`

## Current State

Local Apple embeddings path is implemented and working:

- `afm embed` starts a dedicated embeddings server
- `GET /v1/models` works
- `POST /v1/embeddings` works for string input on Apple NaturalLanguage
- `encoding_format=base64` works
- `dimensions` truncation + renormalization works
- `Scripts/test-embeddings.sh` passes for the Apple path

Token-array request support is now added at the API layer:

- `input: [1,2,3]`
- `input: [[1,2,3],[4,5,6]]`

Decoded and routed correctly. `Tests/MacLocalAPITests/EmbeddingsControllerTests.swift` passes.

## Important Limitation

The Apple / NaturalLanguage backend only supports text input, not pre-tokenized ID arrays.

- Token-array requests against the Apple backend return **400 by design**.
- If the caller needs token-array compatibility, either:
  - change the caller to send strings when targeting Apple embeddings, or
  - finish validating the MLX embeddings path with a cached mlx-community embedding model.

The MLX backend has interface support for token IDs, but MLX model validation is still partial on this machine — there is no cached `mlx-community` embedding model available locally.

## Key Files

Implementation:

- `Sources/MacLocalAPI/EmbeddingsCommand.swift`
- `Sources/MacLocalAPI/Controllers/EmbeddingsController.swift`
- `Sources/MacLocalAPI/Models/EmbeddingBackend.swift`
- `Sources/MacLocalAPI/Models/EmbeddingModelRegistry.swift`
- `Sources/MacLocalAPI/Models/NLContextualEmbeddingBackend.swift`
- `Sources/MacLocalAPI/Models/MLXEmbedderBackend.swift`
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

Passed in this workspace:

- `swift build`
- `swift test --filter EmbeddingModelRegistryTests`
- `swift test --filter EmbeddingsControllerTests`
- `swift run afm embed --list-models`
- `./Scripts/test-embeddings.sh --port <port>`

Previously verified manually: a live Apple server returned real embeddings for `apple-nl-contextual-en`.

## Behavior Note (HTTP 400 Regression)

- Root cause: the embeddings request decoder originally only accepted strings / arrays of strings.
- API layer now supports token arrays as well.
- Apple backend still rejects token IDs because NaturalLanguage embeddings are text-only.

## MLX Status

- Interface support exists, including the token-ID path.
- Full MLX validation is incomplete on this machine — no cached `mlx-community` embedding model available.
- `Scripts/test-embeddings.sh --with-mlx` now skips cleanly if no suitable cached model exists.

## Environment Notes

- `Scripts/test-embeddings-openai.py` was added, but the `openai` Python package is not installed in this environment, so it was not executed.
- Unrelated broader test-suite issues exist elsewhere in the repo; the embeddings-targeted tests above are clean.

## Recommended Next Steps

1. Keep the Apple path as the default local embeddings implementation.
2. If the active caller needs tokenized-input compatibility:
   - switch the caller to send strings when targeting Apple embeddings, **or**
   - finish validating the MLX embeddings path with a known-good cached `mlx-community` embedding model.
3. Before merge, run a broader repo regression pass if desired. The embeddings slice itself is in a good handoff state now.
