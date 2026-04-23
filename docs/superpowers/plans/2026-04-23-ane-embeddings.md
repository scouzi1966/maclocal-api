# ANE Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI-compatible embeddings support to AFM via a new `afm embed` command and `/v1/embeddings` endpoint, with Apple `NLContextualEmbedding` as the default ANE-oriented backend and `MLXEmbedders` linked as a gated phase-1 MLX fallback.

**Architecture:** A registry-driven embeddings stack parallel to the existing chat stack. `EmbeddingsCommand` loads exactly one embedding model at startup, resolves it through `EmbeddingModelRegistry`, initializes a single `EmbeddingBackend` actor, blocks on Apple asset prefetch when needed, and exposes `/v1/embeddings` plus `/v1/models` through a dedicated `EmbeddingsController`.

**Tech Stack:** Swift, Vapor, ArgumentParser, NaturalLanguage, MLX / `MLXEmbedders`

**Spec:** `docs/superpowers/specs/2026-04-23-ane-embeddings-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Package.swift` | **Modify.** Link `MLXEmbedders` into the executable target. |
| `Sources/MacLocalAPI/EmbeddingsCommand.swift` | **New.** `afm embed` CLI surface, startup validation, model selection, and server bootstrap. |
| `Sources/MacLocalAPI/Controllers/EmbeddingsController.swift` | **New.** `POST /v1/embeddings`, `GET /v1/models`, CORS/options handling, request validation, response encoding. |
| `Sources/MacLocalAPI/Models/EmbeddingBackend.swift` | **New.** Shared embeddings protocol, result payloads, enums, normalization helpers, typed errors. |
| `Sources/MacLocalAPI/Models/EmbeddingModelRegistry.swift` | **New.** Registry and runtime resolution for Apple NL models plus gated MLX model entries. |
| `Sources/MacLocalAPI/Models/NLContextualEmbeddingBackend.swift` | **New.** Apple Natural Language backend, asset prefetch, pooling, normalization, token counting. |
| `Sources/MacLocalAPI/Models/MLXEmbedderBackend.swift` | **New.** Actor wrapper over `MLXEmbedders`, tokenizer access, embedding generation, and metadata extraction. |
| `Sources/MacLocalAPI/Models/OpenAIRequest.swift` | **Modify.** Add embeddings request types and `String | [String]` input decoding. |
| `Sources/MacLocalAPI/Models/OpenAIResponse.swift` | **Modify.** Add embeddings response types, item payloads, and base64-capable embedding encoding shapes. |
| `Sources/MacLocalAPI/main.swift` | **Modify.** Register and manually dispatch `embed` alongside existing subcommands. |
| `Sources/MacLocalAPI/Server.swift` | **Modify or avoid.** Only touch if minimal reuse is clearly cleaner than a dedicated embeddings bootstrap. |
| `Tests/MacLocalAPITests/EmbeddingsControllerTests.swift` | **New.** Endpoint contract and error-mapping coverage. |
| `Tests/MacLocalAPITests/EmbeddingModelRegistryTests.swift` | **New.** Registry coverage for Apple and MLX entries. |
| `Tests/MacLocalAPITests/NLContextualEmbeddingBackendTests.swift` | **New.** Backend shape, determinism, and loose semantic sanity checks. |
| `Scripts/test-embeddings.sh` | **New.** End-to-end smoke test for the embeddings server. |
| `Scripts/test-embeddings-openai.py` | **New.** Optional OpenAI Python SDK compatibility smoke test. |

---

## Task 1: Package Wiring and Shared API Types

**Files:**
- Modify: `Package.swift`
- Modify: `Sources/MacLocalAPI/Models/OpenAIRequest.swift`
- Modify: `Sources/MacLocalAPI/Models/OpenAIResponse.swift`
- Create: `Sources/MacLocalAPI/Models/EmbeddingBackend.swift`

- [ ] **Step 1: Link `MLXEmbedders`**

Add this product to the `MacLocalAPI` executable target dependencies in `Package.swift`:

```swift
.product(name: "MLXEmbedders", package: "mlx-swift-lm"),
```

- [ ] **Step 2: Add embeddings request types**

Extend `OpenAIRequest.swift` with:

- `EmbeddingInput` enum decoding `String` or `[String]`
- `EmbeddingsRequest`
- `EmbeddingEncodingFormat`

Required request fields:

- `input`
- `model`
- `encoding_format`
- `dimensions`

Behavior requirements:

- Reject empty string input
- Reject empty array input
- Preserve input ordering

- [ ] **Step 3: Add embeddings response types**

Extend `OpenAIResponse.swift` with:

- `EmbeddingsResponse`
- `EmbeddingDataItem`
- `EmbeddingVectorPayload` or equivalent encoding wrapper
- embeddings-specific model metadata only if needed beyond existing `ModelInfo` / `ModelDetails`

Response requirements:

- OpenAI-compatible top-level shape
- `usage.prompt_tokens`
- `usage.total_tokens`
- support for both float-array and base64 payload forms

- [ ] **Step 4: Create shared embeddings domain types**

Create `EmbeddingBackend.swift` with:

- `protocol EmbeddingBackend: Actor`
- `struct EmbedResult`
- `struct EmbeddingModelEntry`
- `enum EmbeddingBackendKind`
- `enum PoolingKind`
- `enum EmbeddingError`

Also include shared helpers for:

- L2 normalization
- truncation + renormalization
- float32-to-base64 little-endian encoding

- [ ] **Step 5: Verify build**

Run:

```bash
swift build
```

Expected:

- build succeeds
- no missing symbol errors for `MLXEmbedders`

---

## Task 2: Embedding Registry

**Files:**
- Create: `Sources/MacLocalAPI/Models/EmbeddingModelRegistry.swift`
- Read for alignment: `Sources/MacLocalAPI/Models/MLXModelRegistry.swift`
- Read for alignment: `Sources/MacLocalAPI/Models/MLXCacheResolver.swift`

- [ ] **Step 1: Create registry types and Apple defaults**

Implement a registry with static shipped entries for:

- `apple-nl-contextual-en`
- `apple-nl-contextual-multi`

Each entry should capture:

- backend kind
- native dimension
- pooling kind
- normalization behavior
- maximum input tokens
- description

If native dimension is only knowable at runtime, store a load-time resolver rather than a fake constant.

- [ ] **Step 2: Add startup-time MLX entry construction**

Support lazy MLX resolution when `afm embed --backend mlx -m <hf-id>` is used:

- resolve/cache model ID using existing MLX cache conventions
- inspect the loaded embedder config
- construct a runtime `EmbeddingModelEntry`

Do not hard-code a broad MLX embeddings catalog in phase 1.

- [ ] **Step 3: Add lookup and listing APIs**

Expose methods for:

- resolve by model ID
- list shipped models for `--list-models`
- create the single loaded model response for `/v1/models`

- [ ] **Step 4: Add registry tests**

Create `EmbeddingModelRegistryTests.swift` with coverage for:

- Apple entries resolve
- unknown model returns nil
- MLX runtime entry creation from mocked metadata

---

## Task 3: Apple NL Backend

**Files:**
- Create: `Sources/MacLocalAPI/Models/NLContextualEmbeddingBackend.swift`
- Optional test seam helpers in the same file if needed

- [ ] **Step 1: Implement backend initialization**

The backend should:

- resolve requested Apple embedding variant
- determine native dimension dynamically
- expose `modelID`, `nativeDimension`, and `maxInputTokens`

- [ ] **Step 2: Implement asset prefetch**

Before the server binds:

- check `hasAvailableAssets`
- if unavailable, call `requestEmbeddingAssets(for:)`
- block startup until success or failure
- emit visible progress logs so first-run download does not look hung

Map failures to `EmbeddingError.assetDownloadFailed`.

- [ ] **Step 3: Implement embedding generation**

For each input:

- generate token vectors using `NLContextualEmbedding`
- mean-pool token vectors
- L2-normalize pooled vectors
- capture per-input token counts

Do not assume the framework returns sentence-level pooled embeddings.

- [ ] **Step 4: Implement token limit behavior**

Support:

- token counting
- truncation when over limit
- hard failure when caller sends `X-Embedding-Truncate: false`

Surface truncation count via response header, not silent server logs only.

- [ ] **Step 5: Add backend tests**

Create `NLContextualEmbeddingBackendTests.swift` covering:

- vector length equals native dimension
- identical input is stable across repeated calls
- loose semantic sanity via cosine comparisons

Keep assertions robust across OS/runtime changes.

---

## Task 4: MLX Embedder Backend

**Files:**
- Create: `Sources/MacLocalAPI/Models/MLXEmbedderBackend.swift`
- Read for alignment: `Sources/MacLocalAPI/Models/MLXModelService.swift`
- Read for alignment: `Sources/MacLocalAPI/Models/MLXCacheResolver.swift`

- [ ] **Step 1: Wrap `MLXEmbedders` in an actor backend**

Implement an actor that:

- loads one MLX embedder model
- exposes native dimension and token limits
- serializes embed requests at the backend level

- [ ] **Step 2: Implement tokenizer-backed token counting**

Count prompt tokens per input using the model tokenizer so `usage.prompt_tokens` is accurate.

- [ ] **Step 3: Gate MLX behind explicit backend selection**

Phase-1 rule:

- Apple NL remains default
- MLX path requires `--backend mlx`

If the requested MLX model is missing or unavailable, return `503 backendUnavailable` rather than silently falling back.

- [ ] **Step 4: Add a minimal MLX smoke path**

Ensure the implementation is testable by the integration script with a small embedder model. Avoid adding download-heavy coverage to default unit tests.

---

## Task 5: Embeddings Controller

**Files:**
- Create: `Sources/MacLocalAPI/Controllers/EmbeddingsController.swift`

- [ ] **Step 1: Add routes**

Register:

- `POST /v1/embeddings`
- `GET /v1/models`
- `OPTIONS /v1/embeddings`

- [ ] **Step 2: Implement request validation**

Controller must reject:

- unknown model
- empty input
- `dimensions < 1`
- `dimensions > nativeDimension`
- malformed `encoding_format`

Map typed errors to explicit HTTP status codes from the spec.

- [ ] **Step 3: Implement response generation**

For each request:

- normalize input into `[String]`
- call backend
- apply optional truncation + renormalization for `dimensions`
- encode as float arrays or base64
- populate `usage`
- set `X-Embedding-Truncated` header when applicable

- [ ] **Step 4: Reuse existing model response types where practical**

Prefer reusing `ModelInfo`, `ModelDetails`, and `ModelsResponse` from `Server.swift` only if that does not create awkward coupling. If reuse is awkward, move those types into a shared models file as part of this task.

- [ ] **Step 5: Add controller tests**

Create `EmbeddingsControllerTests.swift` for:

- single string request
- array request
- base64 round-trip
- valid dimension truncation
- invalid dimensions
- unknown model
- empty input
- truncation warning header

---

## Task 6: Command and Server Bootstrap

**Files:**
- Create: `Sources/MacLocalAPI/EmbeddingsCommand.swift`
- Modify: `Sources/MacLocalAPI/main.swift`
- Modify: `Sources/MacLocalAPI/Server.swift` only if the dedicated bootstrap path is clearly worse

- [ ] **Step 1: Add `afm embed` command**

Command requirements:

- default model: `apple-nl-contextual-en`
- default port: `9998`
- `-m` / `--model`
- `--backend`
- `--list-models`
- hostname and verbosity flags matching house style

- [ ] **Step 2: Load backend before binding**

Startup sequence:

- resolve model
- construct backend
- prefetch Apple assets if needed
- only then start listening

The server must not advertise readiness while assets are still downloading.

- [ ] **Step 3: Choose the bootstrap approach**

Preferred decision order:

1. create a small embeddings-specific server bootstrap if it keeps chat/server code isolated
2. reuse `Server` only if the change stays narrow and low-coupling

Do not broaden chat-oriented code paths unnecessarily in phase 1.

- [ ] **Step 4: Wire command into manual dispatch**

Update `main.swift` so `embed` is handled consistently with the current custom subcommand dispatch flow.

- [ ] **Step 5: Manual verification**

Run:

```bash
swift run afm embed --list-models
swift run afm embed --port 9998
curl -s http://127.0.0.1:9998/v1/models
```

Expected:

- model list includes Apple NL defaults
- server starts only after assets are ready
- `/v1/models` reports the loaded embedding model

---

## Task 7: Integration Scripts and SDK Compatibility

**Files:**
- Create: `Scripts/test-embeddings.sh`
- Create: `Scripts/test-embeddings-openai.py`

- [ ] **Step 1: Add shell smoke test**

`Scripts/test-embeddings.sh` should:

- start `afm embed` on a free port
- wait for `/v1/models`
- issue single-string and array requests
- assert embedding length
- assert prompt token usage
- assert `dimensions=64` truncation works
- assert truncation header behavior
- stop the server cleanly

- [ ] **Step 2: Add optional MLX smoke mode**

Support:

```bash
./Scripts/test-embeddings.sh --with-mlx
```

This path should verify the gated MLX backend without making it part of the default fast suite.

- [ ] **Step 3: Add OpenAI SDK smoke script**

`Scripts/test-embeddings-openai.py` should verify:

- `client.embeddings.create()` works against `afm embed`
- no adapter layer is required

Keep this optional/documented if the environment lacks the Python dependency.

---

## Task 8: Final Validation and Cleanup

**Files:**
- All touched files above

- [ ] **Step 1: Run unit tests**

Run:

```bash
swift test
```

- [ ] **Step 2: Run integration smoke test**

Run:

```bash
./Scripts/test-embeddings.sh
```

- [ ] **Step 3: Run optional MLX smoke test**

Run only when cache/model availability is acceptable:

```bash
./Scripts/test-embeddings.sh --with-mlx
```

- [ ] **Step 4: Confirm OpenAI client compatibility**

Run when Python environment is available:

```bash
python3 Scripts/test-embeddings-openai.py
```

- [ ] **Step 5: Review scope**

Confirm phase-1 boundaries remain intact:

- no CoreML backend
- no co-hosted embeddings on `afm` or `afm mlx`
- no silent cross-backend fallback
- no vendor edits under `vendor/`

---

## Notes for Implementers

- Prefer a dedicated embeddings bootstrap over heavily generalizing `Server.swift` unless reuse is genuinely simpler.
- Keep numeric limits as named constants; do not introduce magic numbers inline.
- When dimension truncation is requested, truncate first and then L2-renormalize.
- Base64 output must encode little-endian IEEE-754 `Float32` bytes.
- Apple NL startup should be honest about first-run latency: block until assets are available.
