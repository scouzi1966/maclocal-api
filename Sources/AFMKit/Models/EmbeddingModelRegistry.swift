import Foundation

struct EmbeddingModelRegistry {
    static let defaultModelID = "apple-nl-contextual-en"
    static let multilingualModelID = "apple-nl-contextual-multi"

    // Apple NL metadata is finalized at backend-load time. These sentinel values
    // let the registry enumerate shipped model IDs before the backend exists.
    private static let runtimeResolvedDimension = 0
    private static let runtimeResolvedMaxInputTokens = 0

    func shippedModels() -> [EmbeddingModelEntry] {
        [
            Self.makeAppleEntry(
                id: Self.defaultModelID,
                description: "Apple Natural Language contextual embeddings (English)"
            ),
            Self.makeAppleEntry(
                id: Self.multilingualModelID,
                description: "Apple Natural Language contextual embeddings (Latin-script multilingual)"
            ),
        ]
    }

    func listModelIDs() -> [String] {
        shippedModels().map(\.id).sorted()
    }

    func resolve(modelID: String) -> EmbeddingModelEntry? {
        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModelID.isEmpty else {
            return nil
        }

        return shippedModels().first(where: { $0.id == trimmedModelID })
    }

    func makeResolvedAppleEntry(modelID: String, nativeDimension: Int, maxInputTokens: Int) -> EmbeddingModelEntry? {
        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let shippedEntry = shippedModels().first(where: { $0.id == trimmedModelID }) else {
            return nil
        }

        return EmbeddingModelEntry(
            id: shippedEntry.id,
            backend: shippedEntry.backend,
            nativeDimension: nativeDimension,
            supportsMatryoshka: shippedEntry.supportsMatryoshka,
            pooling: shippedEntry.pooling,
            normalized: shippedEntry.normalized,
            maxInputTokens: maxInputTokens,
            description: shippedEntry.description,
            createdEpoch: shippedEntry.createdEpoch
        )
    }

    // macOS 14.0 release date — NLContextualEmbedding shipped with Sonoma and
    // has been surface-compatible since then. Using the OS GA as the model's
    // "created" time gives /v1/models a stable, documented value without
    // having to track per-macOS-version differences.
    private static let macOS14ReleaseEpoch = 1_695_686_400 // 2023-09-26 UTC

    private static func makeAppleEntry(id: String, description: String) -> EmbeddingModelEntry {
        EmbeddingModelEntry(
            id: id,
            backend: .nlContextual,
            nativeDimension: runtimeResolvedDimension,
            supportsMatryoshka: false,
            pooling: .mean,
            normalized: true,
            maxInputTokens: runtimeResolvedMaxInputTokens,
            description: description,
            createdEpoch: macOS14ReleaseEpoch
        )
    }
}
