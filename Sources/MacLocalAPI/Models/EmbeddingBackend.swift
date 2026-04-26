import Foundation

/// A local embedding backend.
///
/// Conformers must guarantee that `nativeDimension` is finalized by the time
/// the backend is handed to an `EmbeddingsController` (i.e. any framework
/// loading required to know the dimension has already run during setup).
/// The controller relies on this when validating the `dimensions` request
/// parameter before calling `embed` / `embedTokenIDs`.
protocol EmbeddingBackend: Actor {
    var modelID: String { get }
    var nativeDimension: Int { get }
    var maxInputTokens: Int { get }

    func embed(_ inputs: [String]) async throws -> EmbedResult
    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult
}

struct EmbedResult: Sendable {
    let vectors: [[Float]]
    let tokenCounts: [Int]
    let truncatedInputCount: Int

    init(vectors: [[Float]], tokenCounts: [Int], truncatedInputCount: Int = 0) {
        self.vectors = vectors
        self.tokenCounts = tokenCounts
        self.truncatedInputCount = truncatedInputCount
    }
}

struct EmbeddingModelEntry: Sendable {
    let id: String
    let backend: EmbeddingBackendKind
    let nativeDimension: Int
    let supportsMatryoshka: Bool
    let pooling: PoolingKind
    let normalized: Bool
    let maxInputTokens: Int
    let description: String
    /// Stable Unix epoch for the `created` field in OpenAI's /v1/models shape.
    /// Per-model constants keep the response idempotent — the same GET returns
    /// the same value across process restarts, which is what the OpenAI spec
    /// implies for a model's intrinsic create time.
    let createdEpoch: Int
}

enum EmbeddingBackendKind: String, Sendable {
    case nlContextual
}

enum PoolingKind: String, Sendable {
    case mean
    case cls
    case lastToken
}

enum EmbeddingError: Error, Sendable {
    case modelNotFound(String)
    case invalidInput(String)
    case invalidDimensions(requested: Int, native: Int)
    case backendUnavailable(id: String, reason: String)
    case assetDownloadRequired(String)
    case assetDownloadFailed(id: String, reason: String)
    case tokenizationFailed(String)
    case internalFailure
}

extension EmbeddingError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Embedding model not found: \(id)"
        case .invalidInput(let reason):
            return "Invalid embedding input: \(reason)"
        case .invalidDimensions(let requested, let native):
            return "Invalid dimensions \(requested); expected a value between 1 and \(native)"
        case .backendUnavailable(let id, let reason):
            return "Embedding backend unavailable for \(id): \(reason)"
        case .assetDownloadRequired(let id):
            return "Embedding assets are required for \(id)"
        case .assetDownloadFailed(let id, let reason):
            return "Embedding asset download failed for \(id): \(reason)"
        case .tokenizationFailed(let reason):
            return "Embedding tokenization failed: \(reason)"
        case .internalFailure:
            return "Internal embedding failure"
        }
    }
}

enum EmbeddingMath {
    static let zeroThreshold: Float = 1e-12

    static func l2Normalize(_ vector: [Float]) -> [Float] {
        let sumSquares = vector.reduce(Float.zero) { partialResult, value in
            partialResult + (value * value)
        }

        guard sumSquares > zeroThreshold else {
            return vector
        }

        let norm = Foundation.sqrt(sumSquares)
        return vector.map { $0 / norm }
    }

    static func truncateAndNormalize(_ vector: [Float], dimensions: Int) -> [Float] {
        precondition(dimensions > 0, "truncateAndNormalize requires dimensions > 0")

        guard dimensions < vector.count else {
            return l2Normalize(vector)
        }

        return l2Normalize(Array(vector.prefix(dimensions)))
    }
}

enum EmbeddingEncoding {
    static func base64LittleEndian(from vector: [Float]) -> String {
        var data = Data(capacity: vector.count * MemoryLayout<Float>.size)

        for value in vector {
            var littleEndianValue = value.bitPattern.littleEndian
            withUnsafeBytes(of: &littleEndianValue) { rawBuffer in
                data.append(contentsOf: rawBuffer)
            }
        }

        return data.base64EncodedString()
    }
}
