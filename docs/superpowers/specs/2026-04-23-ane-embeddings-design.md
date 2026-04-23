# ANE Embeddings (+ MLX Embeddings Unblock) — Design

**Status:** Draft
**Date:** 2026-04-23
**Author:** Jesse Robbins (with Claude)
**Branch target:** `feat/embeddings` (off `main`)

> **PR scope note (2026-04-23):** The initial PR implements only the Apple `NLContextualEmbedding` path (Gap 1). The MLX embeddings unblock (Gap 2, `MLXEmbedders` linkage + MLX backend + `--backend mlx` flag) is deferred to a follow-up branch. The MLX sections below remain as the design for that follow-up.

## Problem

AFM has no embeddings capability today. Two distinct gaps:

1. **No ANE path.** MLX runs on GPU only; the Apple Neural Engine is reachable only via CoreML or the high-level `NLContextualEmbedding` framework. Neither is wired up. Users who want low-power, ANE-resident embedding inference on Apple Silicon have no route through `afm`.

2. **MLX embedding models won't run even though the code exists.** `vendor/mlx-swift-lm/Libraries/Embedders/` contains a complete Swift implementation of Bert/NomicBert/Qwen3 embedders with tokenizer, pooling, and config loading. It is published by the vendor package as the `MLXEmbedders` product (`vendor/mlx-swift-lm/Package.swift:25-26`). But `Package.swift` in this repo (lines 69–71) depends only on `MLXLLM`, `MLXVLM`, `MLXLMCommon` — `MLXEmbedders` is never linked, so none of the embedder code is compiled into `afm`. There is also no `/v1/embeddings` controller, no request/response types, and no CLI subcommand.

## Goal

Ship ANE-accelerated embeddings as the primary, zero-config embedding path in `afm`, exposed as an OpenAI-compatible `/v1/embeddings` endpoint. Simultaneously unblock MLX embeddings so the existing `MLXEmbedders` library is linked, compiled, and usable — gated behind an explicit flag in phase 1 so it does not distract from the ANE work.

## Primary success criteria

- **Functional:** `afm embed` starts a server and serves `/v1/embeddings` against Apple's `NLContextualEmbedding` English model with zero configuration after a one-time asset fetch (see below). A request with `input: "hello world"` returns a correctly shaped OpenAI response with a dense vector of the model's native dimension.
- **First-run asset fetch (honest about "zero config"):** `NLContextualEmbedding` requires on-device assets that may not be present on a clean machine. On `afm embed` startup, if `hasAvailableAssets` is false, the server calls `requestEmbeddingAssets(for:)` and blocks startup (with progress logging) until assets are ready. Subsequent startups are instant. Callers never see a partially-initialized server.
- **OpenAI parity:** Single-string and array-of-strings inputs both work. `encoding_format=float` and `encoding_format=base64` both work. `dimensions` parameter supported via truncation + L2 renormalization where the backend's native dim exceeds the request. `usage.prompt_tokens` is populated.
- **ANE residency (phase 1, observational):** Running `afm embed` while watching `powermetrics`/`mactop` shows non-zero ANE utilization during requests. No formal gate — this is the recommended implementation path but Apple does not expose a hard "is this on ANE" API from NL framework.
- **MLX unblock:** `afm embed --backend mlx -m <mlx-embedder-id>` returns a correctly shaped embedding from `MLXEmbedders`. Proves the library is linked and the dispatch path works.
- **OpenAI client compatibility:** The `openai` Python SDK's `client.embeddings.create()` works against `afm embed` with no adapter.

## Non-goals

Explicitly deferred from this spec:

- **CoreML sentence-transformer backend** (e.g., `sentence-transformers/all-MiniLM-L6-v2` via converted `.mlpackage`). This is phase 2 and gets its own spec. Placeholders in the registry + backend protocol are acceptable; the implementation is not.
- **Co-hosting embeddings on chat servers** via `--embed-model` on `afm` / `afm mlx`. Phase 3.
- **Multilingual as the default.** The default registry shipped is English-only (`apple-nl-contextual-en`); the multilingual Apple variant (`apple-nl-contextual-multi`) is present in the registry but not the default.
- **Model-conversion tooling.** Not needed for phase 1 since `NLContextualEmbedding` is on-device and `MLXEmbedders` pulls from HuggingFace.
- **ANE residency benchmarks / automation.** No `gpu-profile.sh --ane` mode in this phase. Observational only.
- **Matryoshka renormalization correctness tests beyond "dimension matches."** Numerical fidelity of truncation is a phase-2 concern when we ship models that actually advertise Matryoshka support.

## Approach

**Model-registry-driven dispatch.** A single `afm embed` subcommand hosts `/v1/embeddings`. An `EmbeddingModelRegistry` (parallel to the existing `Sources/MacLocalAPI/Models/MLXModelRegistry.swift`) maps a model ID to a `{ backend, config }` entry. An `EmbeddingController` reads the `model` field on the request, looks up the registry, and dispatches to the matching `EmbeddingBackend` implementation. Adding a supported model — whether Apple-NL, CoreML, or MLX — is fundamentally a registry entry plus (when needed) a backend implementation.

Alternatives considered and rejected:

- **Backend-polymorphic protocol with hard-coded dispatch rules (no registry).** Rejected because adding models requires editing the dispatch logic; doesn't match the existing `MLXModelRegistry` pattern already in the house style.
- **Backend-per-subcommand (`afm embed apple`, `afm embed mlx`, ...).** Rejected because OpenAI clients route via the `model` string on a single endpoint, not via subcommand. Fragments UX, forces multiple running servers.

## Architecture

### Directory layout

```
Sources/MacLocalAPI/
├── EmbeddingsCommand.swift                  (new; ArgumentParser subcommand for `afm embed`)
├── Controllers/
│   └── EmbeddingsController.swift           (new; /v1/embeddings + /v1/models routing)
├── Models/
│   ├── EmbeddingModelRegistry.swift         (new; id → backend, config)
│   ├── EmbeddingBackend.swift               (new; protocol + shared types)
│   ├── NLContextualEmbeddingBackend.swift   (new; Apple Natural Language)
│   └── MLXEmbedderBackend.swift             (new; wraps MLXEmbedders.ModelContainer)
```

`main.swift` registers the new subcommand alongside `mlx`, `speech`, `vision`. No changes to existing controllers or services.

### `Package.swift` changes

Add `MLXEmbedders` to the `MacLocalAPI` target's dependencies:

```swift
.product(name: "MLXEmbedders", package: "mlx-swift-lm"),
```

This is the only Package.swift edit in phase 1.

### EmbeddingBackend protocol

```swift
protocol EmbeddingBackend: Actor {
    var modelID: String { get }
    var nativeDimension: Int { get }
    var maxInputTokens: Int { get }

    func embed(_ inputs: [String]) async throws -> EmbedResult
    // EmbedResult contains: [[Float]] vectors, per-input token counts
}
```

Backends are actors because both NL and MLX paths are serializable at the model level (single concurrent inference per loaded model is the safe baseline). The actor isolation replaces ad-hoc semaphores. If a future backend supports parallel batched inference, it can wrap the parallel work internally — the protocol remains the same.

### Model registry

```swift
struct EmbeddingModelEntry: Sendable {
    let id: String                       // OpenAI-compatible model name
    let backend: EmbeddingBackendKind    // .nlContextual, .mlx, (.coreML in phase 2)
    let nativeDimension: Int
    let supportsMatryoshka: Bool
    let pooling: PoolingKind             // .mean, .cls, .lastToken
    let normalized: Bool                 // whether output is already L2-normalized
    let maxInputTokens: Int
    let description: String
}

enum EmbeddingBackendKind: String, Sendable {
    case nlContextual
    case mlx
    // case coreML  (phase 2)
}
```

Phase-1 shipped entries:

- `apple-nl-contextual-en` → `.nlContextual`, English variant (default), `pooling: .mean`
- `apple-nl-contextual-multi` → `.nlContextual`, multilingual variant, `pooling: .mean`
- MLX entries resolved lazily via `MLXCacheResolver` when `--backend mlx -m <hf-id>` is used; entries are constructed from the loaded model's config at startup time rather than hard-coded in the registry (MLX embedders are many and evolve).

Note on Apple NL pooling: `NLContextualEmbedding` exposes only per-token vectors via `enumerateTokenVectors`; it does not return a pooled sentence-level embedding. The backend mean-pools token vectors itself (hence `pooling: .mean` on both Apple entries), then L2-normalizes. This is stated explicitly to avoid the trap of assuming the framework does it.

### CLI surface

```
afm embed                                          # defaults: model=apple-nl-contextual-en, port=9998
afm embed -m apple-nl-contextual-multi             # multilingual NL
afm embed --port 9998                              # explicit port
afm embed -m <hf-id> --backend mlx                 # MLX path (phase 1, gated)
afm embed --list-models                            # dumps registry, exits
```

Defaults:

- `--port 9998` (chosen to not collide with `afm mlx`'s conventional `9999`)
- `--model apple-nl-contextual-en`
- `--backend` inferred from model ID; `mlx` accepted as explicit override for HF MLX embedder IDs

Phase-3 co-hosting (out of scope here, called out for API compatibility): a future `--embed-model <id>` flag on `afm` and `afm mlx` reuses the same `EmbeddingsController` and backend stack.

### API surface

`POST /v1/embeddings`:

```json
{
  "input": "text"                              // or ["t1", "t2", ...]
  "model": "apple-nl-contextual-en",
  "encoding_format": "float",                  // or "base64" (optional, default "float")
  "dimensions": 384                            // optional; must be ≤ nativeDimension
}
```

Response:

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [/* floats */] }
  ],
  "model": "apple-nl-contextual-en",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

`GET /v1/models`: returns the currently loaded embedding model(s) as OpenAI-compatible model objects. In phase 1 the server loads exactly one model (the one passed to `-m`) — `/v1/models` returns that single entry. This matches the existing `afm mlx` server behavior.

### Request lifecycle

1. **Parse & validate.** Decode request; reject empty `input`, unknown `model`, `dimensions > nativeDimension`, or `dimensions < 1`.
2. **Registry lookup.** Resolve `model` → `EmbeddingModelEntry`. For MLX entries, construct the entry from the loaded model's config at startup (not per-request).
3. **Normalize input.** `String | [String]` → `[String]`. Count inputs; empty array rejected.
4. **Enforce token limit.** Tokenize (NL: framework-internal; MLX: via swift-transformers `Tokenizer`). If any input exceeds `maxInputTokens`, truncate and set response header `X-Embedding-Truncated: <count>`. Never silently drop inputs.
5. **Backend.embed(inputs).** Returns `[[Float]]` and per-input token counts.
6. **Post-process.** If `dimensions` was requested and `dimensions < nativeDimension`: truncate each vector to the first `dimensions` components and L2-renormalize. If the backend's output was not already normalized, L2-normalize unconditionally (house default; OpenAI embeddings are normalized).
7. **Encode.** `float` → JSON float arrays. `base64` → little-endian IEEE-754 float32 bytes, base64-encoded per OpenAI spec.
8. **Respond.** OpenAI-shaped payload with `usage` populated: `prompt_tokens = sum(token counts)`, `total_tokens = prompt_tokens`.

### Concurrency

Each backend is an `actor`, so concurrent incoming HTTP requests serialize at the backend level. For phase-1 workloads (single embedding server, low-to-moderate concurrency), this is both simpler and sufficient. If contention becomes a problem in phase 2+, the protocol allows a backend to implement internal batching without changing the controller.

The existing `RequestScheduler`/`RequestSlot` machinery in `Sources/MacLocalAPI/Models/` is **not** reused in phase 1 — it's built around chat-completion token streaming and adds complexity with no payoff for synchronous embed calls.

## Error handling

A typed `EmbeddingError` with explicit HTTP mapping (same pattern as the speech/vision services per CLAUDE.md — no silent 500s):

| Error | HTTP | Condition |
|-------|------|-----------|
| `modelNotFound(id)` | 404 | `model` field not in registry |
| `invalidInput` | 400 | empty input, empty string in array, malformed JSON |
| `invalidDimensions(requested, native)` | 400 | `dimensions` > native or < 1 |
| `inputTooLong` | 413 | input exceeds `maxInputTokens` **and** caller opted out of truncation via `X-Embedding-Truncate: false` request header |
| `backendUnavailable(id, reason)` | 503 | MLX model not in cache, NL framework returned nil embedder, etc. |
| `assetDownloadRequired(id)` | 503 | NL Contextual assets not available and startup prefetch hasn't completed (should be rare — startup blocks on this; surfaces only if an async refresh later invalidates cache) |
| `assetDownloadFailed(id, reason)` | 503 | NL Contextual `requestEmbeddingAssets` callback returned an error at startup |
| `tokenizationFailed(reason)` | 400 | model's tokenizer rejected input |
| `internalFailure(reason)` | 500 | unexpected path |

Default truncation behavior is **silent truncate + response header warning** (`X-Embedding-Truncated: <count>`). Callers who want hard failure pass `X-Embedding-Truncate: false`.

No automatic fallback between backends on error — if NL contextual fails, the response is an error, not a silent fall-through to MLX.

## Testing

### Unit tests (`Tests/MacLocalAPITests/`)

- `EmbeddingsControllerTests.swift` — using `XCTVapor`:
  - Single-string input → 200 + expected shape.
  - Array input → one vector per input, indices correct.
  - `encoding_format=base64` decodes back to the same floats as `float`.
  - `dimensions < nativeDimension` → truncated + L2-renormalized (unit-norm within 1e-6).
  - `dimensions > nativeDimension` → 400 `invalidDimensions`.
  - Unknown model → 404.
  - Empty input → 400.
- `NLContextualEmbeddingBackendTests.swift`:
  - Known string → vector of expected `nativeDimension`.
  - Identical input produces bit-identical vectors across two calls (determinism sanity check).
  - Vectors for "cat" and "kitten" have cosine ≥ 0.5; vectors for "cat" and "xylophone" have lower cosine than that pair (loose semantic sanity; not a precision claim).
- `EmbeddingModelRegistryTests.swift`:
  - Both Apple entries resolvable.
  - Unknown ID returns nil.
  - MLX entry construction from a mocked config.

### Integration script (`Scripts/test-embeddings.sh`)

Autonomous-agent protocol (per CLAUDE.md: inform intent, end-to-end control, end-to-end visibility). Script steps:

1. Start `afm embed` on a random free port (background).
2. Wait for `/v1/models` to respond.
3. Issue curl against `/v1/embeddings` with both single-string and array inputs.
4. Assert `data[].embedding.length == nativeDimension`, `usage.prompt_tokens > 0`, HTTP 200.
5. Issue `dimensions=64` request, assert length 64.
6. Issue a too-long input, assert `X-Embedding-Truncated` header present.
7. Kill server, report pass/fail.

### MLX gate test

A single integration test in the shell script with `--backend mlx` against a small reference MLX embedder (e.g., one of the registered `ModelConfiguration` entries in `Libraries/Embedders/Models.swift` — `bge_small` is a reasonable default given it's pre-wired). Verifies the library is linked, the model loads, and vectors have the right shape. Not run in the default suite; invoked with `./Scripts/test-embeddings.sh --with-mlx` to avoid download-on-CI surprises.

### OpenAI client smoke test

Python snippet in `Scripts/test-embeddings-openai.py` using the official `openai` SDK pointed at `afm embed`. Verifies client compat without us hand-rolling a request. Optional in the default run; documented in the PR description.

## Risk / open questions

- **`NLContextualEmbedding` dimension stability.** Apple has not publicly committed to fixed dimensions across OS versions. Phase 1 queries the model's declared dim at load and uses that; future OS updates could change it but the registry is already dynamic for this case.
- **Asset download UX.** First `afm embed` launch on a clean machine will pause to fetch NL Contextual assets. Size is documented by Apple as roughly a few hundred MB per language. Startup logs emit progress lines so the pause doesn't look like a hang. Agents invoking `afm embed` should expect first-run latency.
- **Asset invalidation.** Apple may invalidate cached assets on OS upgrade. Detection is via `hasAvailableAssets` on each load; a running server does **not** hot-reload — the operator re-runs `afm embed` after an OS update.
- **MLX `ModelContainer` is an `actor`.** The backend wrapper needs careful `await` plumbing. Not novel — existing `MLXModelService` does similar.
- **Matryoshka truncation semantics.** Truncating + renormalizing is the mechanically correct operation for Matryoshka-trained models (MRL); for non-MRL models the result is still a valid embedding but lower quality than native-dim at that truncation. Phase 1 documents this; phase 2 tightens by exposing `supportsMatryoshka` in `/v1/models` metadata.
- **Base64 endianness.** OpenAI spec specifies little-endian float32; verified in the base64 test.

## Phasing

| Phase | Scope | Spec |
|-------|-------|------|
| 1 (this spec) | `afm embed`, `NLContextualEmbeddingBackend`, `MLXEmbedderBackend` linked + gated, registry + protocol scaffolding, full OpenAI endpoint, tests. | this doc |
| 2 | `CoreMLSBERTBackend`, `.mlpackage` conversion pipeline, reference MiniLM model, expanded registry, ANE-residency observability. | future |
| 3 | `--embed-model` co-hosting flag on `afm` / `afm mlx`, shared model lifecycle. | future |
