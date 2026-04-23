import Vapor
import Foundation

struct EmbeddingsController: RouteCollection {
    private let modelEntry: EmbeddingModelEntry
    private let backend: any EmbeddingBackend

    init(modelEntry: EmbeddingModelEntry, backend: any EmbeddingBackend) {
        self.modelEntry = modelEntry
        self.backend = backend
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "embeddings", body: .collect(maxSize: "1mb"), use: createEmbeddings)
        v1.on(.OPTIONS, "embeddings", use: handleOptions)
        v1.get("models", use: listModels)
    }

    private func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        applyCORSHeaders(to: response)
        return response
    }

    private func listModels(req: Request) async throws -> Response {
        let model = EmbeddingModelInfo(
            id: modelEntry.id,
            ownedBy: "apple"
        )
        let response = EmbeddingModelsResponse(data: [model])
        return try jsonResponse(for: response)
    }

    private func createEmbeddings(req: Request) async throws -> Response {
        do {
            let request = try req.content.decode(EmbeddingsRequest.self)
            let requestedModelID = request.model ?? modelEntry.id
            guard requestedModelID == modelEntry.id else {
                throw EmbeddingError.modelNotFound(requestedModelID)
            }

            if request.input.isEmpty {
                throw EmbeddingError.invalidInput("Input must not be empty")
            }

            let nativeDimension = await backend.nativeDimension

            if let dimensions = request.dimensions {
                guard dimensions > 0, dimensions <= nativeDimension else {
                    throw EmbeddingError.invalidDimensions(requested: dimensions, native: nativeDimension)
                }
            }

            let embedResult: EmbedResult
            if request.input.isTokenized {
                embedResult = try await backend.embedTokenIDs(request.input.tokenIDArrays)
            } else {
                embedResult = try await backend.embed(request.input.strings)
            }
            let targetDimensions = request.dimensions ?? nativeDimension
            let encodingFormat = request.resolvedEncodingFormat

            let data = embedResult.vectors.enumerated().map { index, vector in
                let outputVector = targetDimensions < vector.count
                    ? EmbeddingMath.truncateAndNormalize(vector, dimensions: targetDimensions)
                    : EmbeddingMath.l2Normalize(vector)
                let payload: EmbeddingVectorPayload
                switch encodingFormat {
                case .float:
                    payload = .float(outputVector)
                case .base64:
                    payload = .base64(EmbeddingEncoding.base64LittleEndian(from: outputVector))
                }
                return EmbeddingDataItem(index: index, embedding: payload)
            }

            let response = EmbeddingsResponse(
                model: modelEntry.id,
                data: data,
                promptTokens: embedResult.tokenCounts.reduce(0, +)
            )
            let httpResponse = try jsonResponse(for: response)
            if embedResult.truncatedInputCount > 0 {
                httpResponse.headers.replaceOrAdd(
                    name: "X-Embedding-Truncated",
                    value: "\(embedResult.truncatedInputCount)"
                )
            }
            return httpResponse
        } catch let embeddingError as EmbeddingError {
            return try errorResponse(for: embeddingError)
        } catch {
            return try errorResponse(for: .internalFailure(error.localizedDescription))
        }
    }

    private func jsonResponse<T: Content>(for payload: T) throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        applyCORSHeaders(to: response)
        try response.content.encode(payload)
        return response
    }

    private func errorResponse(for error: EmbeddingError) throws -> Response {
        let response = Response(status: Self.httpStatus(for: error))
        response.headers.add(name: .contentType, value: "application/json")
        applyCORSHeaders(to: response)
        try response.content.encode(OpenAIError(message: error.localizedDescription, type: "embedding_error"))
        return response
    }

    private func applyCORSHeaders(to response: Response) {
        response.headers.replaceOrAdd(name: .accessControlAllowOrigin, value: "*")
        response.headers.replaceOrAdd(name: .accessControlAllowMethods, value: "POST, GET, OPTIONS")
        response.headers.replaceOrAdd(name: .accessControlAllowHeaders, value: "Content-Type, Authorization, X-Embedding-Truncate")
    }

    static func httpStatus(for error: EmbeddingError) -> HTTPStatus {
        switch error {
        case .modelNotFound:
            return .notFound
        case .invalidInput, .invalidDimensions, .tokenizationFailed:
            return .badRequest
        case .inputTooLong:
            return .payloadTooLarge
        case .backendUnavailable, .assetDownloadRequired, .assetDownloadFailed:
            return .serviceUnavailable
        case .internalFailure:
            return .internalServerError
        }
    }
}

private struct EmbeddingModelsResponse: Content {
    let object: String
    let data: [EmbeddingModelInfo]

    init(data: [EmbeddingModelInfo]) {
        self.object = "list"
        self.data = data
    }
}

private struct EmbeddingModelInfo: Content {
    let id: String
    let object: String
    let created: Int
    let ownedBy: String

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }

    init(id: String, ownedBy: String) {
        self.id = id
        self.object = "model"
        self.created = Int(Date().timeIntervalSince1970)
        self.ownedBy = ownedBy
    }
}
