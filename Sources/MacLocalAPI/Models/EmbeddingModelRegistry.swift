import Foundation

struct EmbeddingModelRegistry {
    static let defaultModelID = "apple-nl-contextual-en"
    static let multilingualModelID = "apple-nl-contextual-multi"

    // Apple NL metadata is finalized at backend-load time. These sentinel values are
    // placeholders so the registry can resolve shipped model IDs before the backend exists.
    private static let runtimeResolvedDimension = 0
    private static let runtimeResolvedMaxInputTokens = 0
    private static let mlxFallbackMaxInputTokens = 512

    private let resolver: MLXCacheResolver

    init(resolver: MLXCacheResolver = MLXCacheResolver()) {
        self.resolver = resolver
    }

    func shippedModels() -> [EmbeddingModelEntry] {
        [
            Self.makeAppleEntry(
                id: Self.defaultModelID,
                description: "Apple Natural Language contextual embeddings (English)"
            ),
            Self.makeAppleEntry(
                id: Self.multilingualModelID,
                description: "Apple Natural Language contextual embeddings (multilingual)"
            ),
        ]
    }

    func listModelIDs() -> [String] {
        shippedModels().map(\.id).sorted()
    }

    func resolve(modelID: String, backendOverride: EmbeddingBackendKind? = nil) throws -> EmbeddingModelEntry? {
        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModelID.isEmpty else {
            return nil
        }

        if backendOverride == .mlx {
            return try makeMLXEntry(modelID: trimmedModelID)
        }

        if let appleEntry = shippedModels().first(where: { $0.id == trimmedModelID }) {
            return appleEntry
        }

        return nil
    }

    func makeResolvedAppleEntry(modelID: String, nativeDimension: Int, maxInputTokens: Int) -> EmbeddingModelEntry? {
        guard let shippedEntry = shippedModels().first(where: { $0.id == modelID }) else {
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
            description: shippedEntry.description
        )
    }

    func makeMLXEntry(modelID: String) throws -> EmbeddingModelEntry {
        let normalizedModelID = resolver.normalizedModelID(modelID)
        guard let modelDirectory = resolver.localModelDirectory(repoId: normalizedModelID) else {
            throw EmbeddingError.backendUnavailable(
                id: normalizedModelID,
                reason: "Model directory was not found in the MLX cache"
            )
        }

        let config = try loadMLXConfig(modelDirectory: modelDirectory, modelID: normalizedModelID)
        let pooling = loadPoolingMetadata(modelDirectory: modelDirectory)

        let nativeDimension = pooling.dimension ?? config.hiddenSize
        let maxInputTokens = config.maxPositionEmbeddings ?? Self.mlxFallbackMaxInputTokens
        let poolingKind = pooling.pooling ?? Self.defaultMLXPooling(for: config.modelType)

        return EmbeddingModelEntry(
            id: normalizedModelID,
            backend: .mlx,
            nativeDimension: nativeDimension,
            supportsMatryoshka: Self.supportsMatryoshka(for: normalizedModelID),
            pooling: poolingKind,
            normalized: false,
            maxInputTokens: maxInputTokens,
            description: "MLX embedding model"
        )
    }

    private func loadMLXConfig(modelDirectory: URL, modelID: String) throws -> MLXEmbeddingConfigMetadata {
        let configURL = modelDirectory.appendingPathComponent("config.json")

        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw EmbeddingError.backendUnavailable(
                id: modelID,
                reason: "Unable to read config.json: \(error.localizedDescription)"
            )
        }

        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let json else {
                throw EmbeddingError.backendUnavailable(id: modelID, reason: "config.json is not a JSON object")
            }

            guard let modelType = stringValue(in: json, key: "model_type") else {
                throw EmbeddingError.backendUnavailable(id: modelID, reason: "config.json is missing model_type")
            }

            let hiddenSize = intValue(in: json, keys: ["hidden_size", "dim"])
            guard let hiddenSize else {
                throw EmbeddingError.backendUnavailable(
                    id: modelID,
                    reason: "config.json is missing hidden_size/dim for embedding dimension inference"
                )
            }

            let maxPositionEmbeddings = intValue(
                in: json,
                keys: ["max_position_embeddings", "max_trained_positions"]
            )

            return MLXEmbeddingConfigMetadata(
                modelType: modelType,
                hiddenSize: hiddenSize,
                maxPositionEmbeddings: maxPositionEmbeddings
            )
        } catch let embeddingError as EmbeddingError {
            throw embeddingError
        } catch {
            throw EmbeddingError.backendUnavailable(
                id: modelID,
                reason: "Unable to parse config.json: \(error.localizedDescription)"
            )
        }
    }

    private func loadPoolingMetadata(modelDirectory: URL) -> MLXEmbeddingPoolingMetadata {
        let poolingURL = modelDirectory
            .appendingPathComponent("1_Pooling")
            .appendingPathComponent("config.json")

        guard
            let data = try? Data(contentsOf: poolingURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return MLXEmbeddingPoolingMetadata(dimension: nil, pooling: nil)
        }

        let dimension = intValue(in: json, keys: ["word_embedding_dimension"])
        let pooling: PoolingKind?
        if boolValue(in: json, key: "pooling_mode_cls_token") == true {
            pooling = .cls
        } else if boolValue(in: json, key: "pooling_mode_mean_tokens") == true {
            pooling = .mean
        } else if boolValue(in: json, key: "pooling_mode_lasttoken") == true {
            pooling = .lastToken
        } else {
            pooling = nil
        }

        return MLXEmbeddingPoolingMetadata(dimension: dimension, pooling: pooling)
    }

    private static func makeAppleEntry(id: String, description: String) -> EmbeddingModelEntry {
        EmbeddingModelEntry(
            id: id,
            backend: .nlContextual,
            nativeDimension: runtimeResolvedDimension,
            supportsMatryoshka: false,
            pooling: .mean,
            normalized: true,
            maxInputTokens: runtimeResolvedMaxInputTokens,
            description: description
        )
    }

    private static func defaultMLXPooling(for modelType: String) -> PoolingKind {
        switch modelType {
        case "qwen3":
            return .lastToken
        default:
            return .mean
        }
    }

    private static func supportsMatryoshka(for modelID: String) -> Bool {
        let lowercaseID = modelID.lowercased()
        return lowercaseID.contains("nomic-embed-text-v1.5") || lowercaseID.contains("matryoshka")
    }
}

private struct MLXEmbeddingConfigMetadata {
    let modelType: String
    let hiddenSize: Int
    let maxPositionEmbeddings: Int?
}

private struct MLXEmbeddingPoolingMetadata {
    let dimension: Int?
    let pooling: PoolingKind?
}

private func intValue(in json: [String: Any], keys: [String]) -> Int? {
    for key in keys {
        if let int = intValue(in: json, key: key) {
            return int
        }
    }

    return nil
}

private func intValue(in json: [String: Any], key: String) -> Int? {
    if let int = json[key] as? Int {
        return int
    }

    if let number = json[key] as? NSNumber {
        return number.intValue
    }

    if let string = json[key] as? String, let int = Int(string) {
        return int
    }

    return nil
}

private func stringValue(in json: [String: Any], key: String) -> String? {
    json[key] as? String
}

private func boolValue(in json: [String: Any], key: String) -> Bool? {
    if let bool = json[key] as? Bool {
        return bool
    }

    if let number = json[key] as? NSNumber {
        return number.boolValue
    }

    return nil
}
