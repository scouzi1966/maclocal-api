import Foundation
import NaturalLanguage

actor NLContextualEmbeddingBackend: EmbeddingBackend {
    private static let multilingualScript = NLScript.latin

    let modelID: String
    let nativeDimension: Int
    let maxInputTokens: Int

    private let embedding: NLContextualEmbedding
    private let language: NLLanguage?
    private var isLoaded = false

    init(modelID: String) throws {
        guard let selection = Self.selection(for: modelID) else {
            throw EmbeddingError.modelNotFound(modelID)
        }

        self.modelID = modelID
        self.embedding = selection.embedding
        self.language = selection.language
        self.nativeDimension = Int(selection.embedding.dimension)
        self.maxInputTokens = Int(selection.embedding.maximumSequenceLength)
    }

    func prepare() async throws {
        try await ensureAssetsAvailable()
        try loadIfNeeded()
    }

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one input is required")
        }

        try await prepare()

        var vectors: [[Float]] = []
        var tokenCounts: [Int] = []

        for input in inputs {
            guard !input.isEmpty else {
                throw EmbeddingError.invalidInput("Input strings must not be empty")
            }

            let result: NLContextualEmbeddingResult
            do {
                result = try embedding.embeddingResult(for: input, language: language)
            } catch let embeddingError as EmbeddingError {
                throw embeddingError
            } catch {
                throw EmbeddingError.backendUnavailable(
                    id: modelID,
                    reason: error.localizedDescription
                )
            }

            tokenCounts.append(Int(result.sequenceLength))
            vectors.append(poolMeanNormalized(result: result))
        }

        return EmbedResult(vectors: vectors, tokenCounts: tokenCounts)
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one token-id input is required")
        }
        throw EmbeddingError.invalidInput("Pre-tokenized input is not supported by Apple NL embeddings")
    }

    func unload() {
        embedding.unload()
        isLoaded = false
    }

    private func ensureAssetsAvailable() async throws {
        guard !embedding.hasAvailableAssets else {
            return
        }

        print("[Embeddings] Requesting Apple NL assets for \(modelID)...")

        do {
            let result = try await embedding.requestAssets()
            switch result {
            case .available:
                print("[Embeddings] Apple NL assets ready for \(modelID).")
            case .notAvailable:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "Assets were not available for download"
                )
            case .error:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "The NaturalLanguage framework reported an asset download error"
                )
            @unknown default:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "The NaturalLanguage framework returned an unknown asset result"
                )
            }
        } catch let embeddingError as EmbeddingError {
            throw embeddingError
        } catch {
            throw EmbeddingError.assetDownloadFailed(id: modelID, reason: error.localizedDescription)
        }
    }

    private func loadIfNeeded() throws {
        guard !isLoaded else {
            return
        }

        do {
            try embedding.load()
            isLoaded = true
        } catch {
            throw EmbeddingError.backendUnavailable(id: modelID, reason: error.localizedDescription)
        }
    }

    private func poolMeanNormalized(result: NLContextualEmbeddingResult) -> [Float] {
        var sum = Array(repeating: Float.zero, count: nativeDimension)
        var tokenCount = 0
        let fullRange = result.string.startIndex..<result.string.endIndex

        result.enumerateTokenVectors(in: fullRange) { tokenVector, _ in
            guard tokenVector.count == self.nativeDimension else {
                return true
            }

            for (index, value) in tokenVector.enumerated() {
                sum[index] += Float(value)
            }

            tokenCount += 1
            return true
        }

        guard tokenCount > 0 else {
            return sum
        }

        let scale = Float(tokenCount)
        let mean = sum.map { $0 / scale }
        return EmbeddingMath.l2Normalize(mean)
    }

    private static func selection(for modelID: String) -> (embedding: NLContextualEmbedding, language: NLLanguage?)? {
        switch modelID {
        case EmbeddingModelRegistry.defaultModelID:
            guard let embedding = NLContextualEmbedding(language: NLLanguage.english) else {
                return nil
            }
            return (embedding, NLLanguage.english)
        case EmbeddingModelRegistry.multilingualModelID:
            guard let embedding = NLContextualEmbedding(script: multilingualScript) else {
                return nil
            }
            return (embedding, nil)
        default:
            return nil
        }
    }
}
