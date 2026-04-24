import Vapor
import Foundation

struct EmbeddingsController: RouteCollection {
    private static let maxRequestBodySize: ByteCount = "1mb"

    private let modelEntry: EmbeddingModelEntry
    private let backend: any EmbeddingBackend

    init(modelEntry: EmbeddingModelEntry, backend: any EmbeddingBackend) {
        self.modelEntry = modelEntry
        self.backend = backend
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "embeddings", body: .collect(maxSize: Self.maxRequestBodySize), use: createEmbeddings)
        v1.on(.OPTIONS, "embeddings", use: handleOptions)
        v1.get("models", use: listModels)
        v1.on(.OPTIONS, "models", use: handleOptions)
    }

    private func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        applyCORSHeaders(to: response, for: req)
        return response
    }

    private func listModels(req: Request) async throws -> Response {
        // Advertise only the model this server actually loaded. Advertising the
        // full shipped-model list would cause 404s when clients discovered and
        // then requested an ID the running backend can't serve.
        let model = EmbeddingModelInfo(id: modelEntry.id, ownedBy: "apple")
        let response = EmbeddingModelsResponse(data: [model])
        return try jsonResponse(for: response, request: req)
    }

    private func createEmbeddings(req: Request) async throws -> Response {
        do {
            let request = try req.content.decode(EmbeddingsRequest.self)
            let requestedModelID = (request.model ?? modelEntry.id)
                .trimmingCharacters(in: .whitespacesAndNewlines)
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

            for (index, vector) in embedResult.vectors.enumerated() where vector.count != nativeDimension {
                req.logger.error(
                    "Embeddings backend dimension drift: expected \(nativeDimension), got \(vector.count) at index \(index)"
                )
                throw EmbeddingError.internalFailure
            }

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
            let httpResponse = try jsonResponse(for: response, request: req)
            if embedResult.truncatedInputCount > 0 {
                httpResponse.headers.replaceOrAdd(
                    name: "X-Embedding-Truncated",
                    value: "\(embedResult.truncatedInputCount)"
                )
            }
            return httpResponse
        } catch let embeddingError as EmbeddingError {
            return try errorResponse(for: embeddingError, request: req)
        } catch let decodingError as DecodingError {
            return try errorResponse(for: .invalidInput(Self.describeDecodingError(decodingError)), request: req)
        } catch let abortError as AbortError where abortError.status.code >= 400 && abortError.status.code < 500 {
            return try abortErrorResponse(for: abortError, request: req)
        } catch {
            req.logger.error("Embeddings request failed: \(String(reflecting: error))")
            return try errorResponse(for: .internalFailure, request: req)
        }
    }

    private func jsonResponse<T: Content>(for payload: T, request: Request) throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        applyCORSHeaders(to: response, for: request)
        try response.content.encode(payload)
        return response
    }

    private func errorResponse(for error: EmbeddingError, request: Request) throws -> Response {
        let response = Response(status: Self.httpStatus(for: error))
        response.headers.add(name: .contentType, value: "application/json")
        applyCORSHeaders(to: response, for: request)
        try response.content.encode(OpenAIError(message: error.localizedDescription, type: "embedding_error"))
        return response
    }

    private func abortErrorResponse(for abortError: AbortError, request: Request) throws -> Response {
        let response = Response(status: abortError.status)
        response.headers.add(name: .contentType, value: "application/json")
        applyCORSHeaders(to: response, for: request)
        try response.content.encode(OpenAIError(message: abortError.reason, type: "embedding_error"))
        return response
    }

    private static let defaultAllowHeaders = "Content-Type, Authorization, OpenAI-Organization, OpenAI-Project"

    private func applyCORSHeaders(to response: Response, for request: Request) {
        response.headers.replaceOrAdd(name: .accessControlAllowOrigin, value: "*")
        response.headers.replaceOrAdd(name: .accessControlAllowMethods, value: "POST, GET, OPTIONS")
        // Reflect the browser's preflight request-headers list (which can include
        // SDK-specific headers like x-stainless-*) and fall back to a default set
        // that covers the common OpenAI-compatible clients.
        let requested = request.headers.first(name: "Access-Control-Request-Headers")
        let allowHeaders = requested.flatMap { $0.isEmpty ? nil : $0 } ?? Self.defaultAllowHeaders
        response.headers.replaceOrAdd(name: .accessControlAllowHeaders, value: allowHeaders)
        response.headers.replaceOrAdd(name: "Access-Control-Expose-Headers", value: "X-Embedding-Truncated")
        // Intermediary caches must vary on these request headers so a preflight
        // response computed for one client's header set is not served to another.
        response.headers.replaceOrAdd(name: .vary, value: "Origin, Access-Control-Request-Headers")
    }

    private static func describeDecodingError(_ error: DecodingError) -> String {
        switch error {
        case .dataCorrupted(let ctx):
            return "Malformed request body: \(ctx.debugDescription)"
        case .keyNotFound(let key, _):
            return "Missing required field: \(key.stringValue)"
        case .typeMismatch(_, let ctx), .valueNotFound(_, let ctx):
            let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
            return path.isEmpty
                ? "Invalid field value: \(ctx.debugDescription)"
                : "Invalid value for field '\(path)': \(ctx.debugDescription)"
        @unknown default:
            return "Malformed request body"
        }
    }

    static func httpStatus(for error: EmbeddingError) -> HTTPStatus {
        switch error {
        case .modelNotFound:
            return .notFound
        case .invalidInput, .invalidDimensions, .tokenizationFailed:
            return .badRequest
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
