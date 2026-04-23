import Foundation
import MLX
import MLXEmbedders

actor MLXEmbedderBackend: EmbeddingBackend {
    let modelID: String
    let nativeDimension: Int
    let maxInputTokens: Int

    private let modelContainer: ModelContainer

    init(modelID: String, resolver: MLXCacheResolver = MLXCacheResolver()) async throws {
        let registry = EmbeddingModelRegistry(resolver: resolver)
        let normalizedModelID = resolver.normalizedModelID(modelID)
        let configuration: ModelConfiguration
        if let modelDirectory = resolver.localModelDirectory(repoId: normalizedModelID) {
            configuration = ModelConfiguration(directory: modelDirectory)
        } else {
            configuration = ModelConfiguration(id: normalizedModelID)
        }

        self.modelContainer = try await loadModelContainer(configuration: configuration)
        self.modelID = normalizedModelID

        if let resolvedEntry = try? registry.makeMLXEntry(modelID: normalizedModelID) {
            self.nativeDimension = resolvedEntry.nativeDimension
            self.maxInputTokens = resolvedEntry.maxInputTokens
        } else {
            let inferredMetadata = await modelContainer.perform { model, tokenizer, pooler in
                let sampleTokens = tokenizer.encode(text: "hello")
                let inputTokens = MLXArray(sampleTokens).expandedDimensions(axes: [0])
                let attentionMask = MLXArray(Array(repeating: 1, count: sampleTokens.count))
                    .expandedDimensions(axes: [0])
                let outputs = model(
                    inputTokens,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: attentionMask
                )
                let pooled = pooler(outputs, mask: attentionMask, normalize: false)
                eval(pooled)
                let inferredDimension = pooled.reshaped(-1).asArray(Float.self).count
                return (dimension: inferredDimension, maxInputTokens: 512)
            }
            self.nativeDimension = inferredMetadata.dimension
            self.maxInputTokens = inferredMetadata.maxInputTokens
        }
    }

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one input is required")
        }

        return try await modelContainer.perform { model, tokenizer, pooler in
            var vectors: [[Float]] = []
            var tokenCounts: [Int] = []
            var truncatedInputCount = 0

            for input in inputs {
                guard !input.isEmpty else {
                    throw EmbeddingError.invalidInput("Input strings must not be empty")
                }

                let encodedTokens = tokenizer.encode(text: input)
                guard !encodedTokens.isEmpty else {
                    throw EmbeddingError.tokenizationFailed("Tokenizer returned no tokens")
                }

                let truncatedTokens = Array(encodedTokens.prefix(self.maxInputTokens))
                if truncatedTokens.count < encodedTokens.count {
                    truncatedInputCount += 1
                }

                tokenCounts.append(truncatedTokens.count)

                let inputTokens = MLXArray(truncatedTokens).expandedDimensions(axes: [0])
                let attentionMask = MLXArray(Array(repeating: 1, count: truncatedTokens.count))
                    .expandedDimensions(axes: [0])
                let outputs = model(
                    inputTokens,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: attentionMask
                )
                let pooled = pooler(outputs, mask: attentionMask, normalize: false)
                eval(pooled)

                var vector = pooled.reshaped(-1).asArray(Float.self)
                if vector.count > self.nativeDimension {
                    vector = Array(vector.prefix(self.nativeDimension))
                }

                guard vector.count == self.nativeDimension else {
                    throw EmbeddingError.backendUnavailable(
                        id: self.modelID,
                        reason: "MLX embedder returned dimension \(vector.count), expected \(self.nativeDimension)"
                    )
                }

                vectors.append(EmbeddingMath.l2Normalize(vector))
            }

            return EmbedResult(
                vectors: vectors,
                tokenCounts: tokenCounts,
                truncatedInputCount: truncatedInputCount
            )
        }
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one token-id input is required")
        }

        return try await modelContainer.perform { model, _, pooler in
            var vectors: [[Float]] = []
            var tokenCounts: [Int] = []
            var truncatedInputCount = 0

            for inputTokens in inputs {
                guard !inputTokens.isEmpty else {
                    throw EmbeddingError.invalidInput("Token-id inputs must not be empty")
                }

                let truncatedTokens = Array(inputTokens.prefix(self.maxInputTokens))
                if truncatedTokens.count < inputTokens.count {
                    truncatedInputCount += 1
                }

                tokenCounts.append(truncatedTokens.count)

                let mlxInputTokens = MLXArray(truncatedTokens).expandedDimensions(axes: [0])
                let attentionMask = MLXArray(Array(repeating: 1, count: truncatedTokens.count))
                    .expandedDimensions(axes: [0])
                let outputs = model(
                    mlxInputTokens,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: attentionMask
                )
                let pooled = pooler(outputs, mask: attentionMask, normalize: false)
                eval(pooled)

                var vector = pooled.reshaped(-1).asArray(Float.self)
                if vector.count > self.nativeDimension {
                    vector = Array(vector.prefix(self.nativeDimension))
                }

                guard vector.count == self.nativeDimension else {
                    throw EmbeddingError.backendUnavailable(
                        id: self.modelID,
                        reason: "MLX embedder returned dimension \(vector.count), expected \(self.nativeDimension)"
                    )
                }

                vectors.append(EmbeddingMath.l2Normalize(vector))
            }

            return EmbedResult(
                vectors: vectors,
                tokenCounts: tokenCounts,
                truncatedInputCount: truncatedInputCount
            )
        }
    }
}
