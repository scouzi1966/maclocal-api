import Foundation

/// Resolves the embedding model entry + backend for a request.
///
/// This lets `EmbeddingsController` serve embeddings in two deployments without
/// duplicating the request logic:
/// - **Standalone** (`afm embed`): a single backend is loaded up front
///   (`PreloadedEmbeddingResolver`).
/// - **Unified main server** (`:9999`, #132): Apple NaturalLanguage backends are
///   loaded lazily on first use (`LazyAppleEmbeddingResolver`), so a chat-only
///   server pays nothing for embeddings until the first request. The lazy path
///   only ever touches Apple NL — it never triggers `MLXMetalLibrary` init.
protocol EmbeddingBackendResolver: Sendable {
    /// Resolve the backend for the requested model id (`nil` → the default model).
    /// Throws `EmbeddingError.modelNotFound` for an unknown id.
    func resolve(requestedModelID: String?) async throws -> (entry: EmbeddingModelEntry, backend: any EmbeddingBackend)

    /// Models to advertise (used by the standalone server's `/v1/models`).
    func advertisedModels() async -> [EmbeddingModelEntry]
}

/// Serves a single, already-loaded backend (the `afm embed` standalone server).
struct PreloadedEmbeddingResolver: EmbeddingBackendResolver {
    let entry: EmbeddingModelEntry
    let backend: any EmbeddingBackend

    func resolve(requestedModelID: String?) async throws -> (entry: EmbeddingModelEntry, backend: any EmbeddingBackend) {
        let requested = (requestedModelID ?? entry.id).trimmingCharacters(in: .whitespacesAndNewlines)
        guard requested == entry.id else { throw EmbeddingError.modelNotFound(requested) }
        return (entry, backend)
    }

    func advertisedModels() async -> [EmbeddingModelEntry] { [entry] }
}

/// Lazily loads + caches Apple NaturalLanguage contextual embedding backends on
/// first use. Apple-only; never initializes MLX. Used by the unified main server.
actor LazyAppleEmbeddingResolver: EmbeddingBackendResolver {
    private let registry = EmbeddingModelRegistry()
    private var cache: [String: (entry: EmbeddingModelEntry, backend: any EmbeddingBackend)] = [:]

    func resolve(requestedModelID: String?) async throws -> (entry: EmbeddingModelEntry, backend: any EmbeddingBackend) {
        let requested = (requestedModelID ?? EmbeddingModelRegistry.defaultModelID)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let shipped = registry.resolve(modelID: requested) else {
            throw EmbeddingError.modelNotFound(requested)
        }
        if let hit = cache[shipped.id] { return hit }

        let backend = try NLContextualEmbeddingBackend(modelID: shipped.id)
        try await backend.prepare()
        guard let resolved = registry.makeResolvedAppleEntry(
            modelID: shipped.id,
            nativeDimension: await backend.nativeDimension,
            maxInputTokens: await backend.maxInputTokens
        ) else {
            throw EmbeddingError.backendUnavailable(id: shipped.id, reason: "Failed to resolve Apple embedding metadata")
        }

        let pair: (entry: EmbeddingModelEntry, backend: any EmbeddingBackend) = (resolved, backend)
        cache[shipped.id] = pair
        return pair
    }

    func advertisedModels() async -> [EmbeddingModelEntry] { registry.shippedModels() }
}
