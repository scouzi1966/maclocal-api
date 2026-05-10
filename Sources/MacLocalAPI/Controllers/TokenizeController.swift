import Vapor
import Foundation

/// `POST /v1/tokenize` (vLLM-compatible) and `POST /v1/count_tokens`
/// (Anthropic-compatible) — agent-friendly tokenization endpoints. (T1.6)
///
/// Both accept `{ "text": "..." }` (or vLLM's `prompt`) and a model id.
/// `messages` form (chat-template tokenization) is a follow-up.
///
/// `/v1/tokenize` returns `{ tokens, count, model, max_model_len? }`.
/// `/v1/count_tokens` returns `{ input_tokens, model }` (Anthropic style).
///
/// Foundation backend has no public tokenizer — both endpoints return 422
/// with `error.code: "tokenize_unsupported"` when no MLX model is loaded.
struct TokenizeController: RouteCollection {
    private let mlxModelID: String?
    private let mlxModelService: MLXModelService?
    private let contextWindow: Int?

    init(mlxModelID: String?, mlxModelService: MLXModelService?, contextWindow: Int?) {
        self.mlxModelID = mlxModelID
        self.mlxModelService = mlxModelService
        self.contextWindow = contextWindow
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "tokenize", body: .collect(maxSize: "8mb"), use: tokenize)
        v1.on(.OPTIONS, "tokenize", use: handleOptions)
        v1.on(.POST, "count_tokens", body: .collect(maxSize: "8mb"), use: countTokens)
        v1.on(.OPTIONS, "count_tokens", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    /// vLLM-compatible: returns the full token list + count.
    func tokenize(req: Request) async throws -> Response {
        let body = try Self.decodeBody(req)
        let tokens = try await encode(body.effectiveText, requestedModel: body.model, on: req)
        let payload = TokenizeResponse(
            tokens: tokens,
            count: tokens.count,
            model: mlxModelID ?? body.model ?? "unknown",
            maxModelLen: contextWindow
        )
        return try Self.jsonResponse(payload)
    }

    /// Anthropic-compatible: returns just the count under `input_tokens`.
    func countTokens(req: Request) async throws -> Response {
        let body = try Self.decodeBody(req)
        let tokens = try await encode(body.effectiveText, requestedModel: body.model, on: req)
        let payload = CountTokensResponse(
            inputTokens: tokens.count,
            model: mlxModelID ?? body.model ?? "unknown"
        )
        return try Self.jsonResponse(payload)
    }

    // MARK: - Internals

    private func encode(_ text: String, requestedModel: String?, on req: Request) async throws -> [Int] {
        guard let service = mlxModelService else {
            throw TokenizeUnsupportedError(requestId: req.afmRequestID)
        }
        // Optional sanity check: if a specific model id was requested, warn if
        // it doesn't match the loaded one (don't fail — agents often pass aliases).
        if let requestedModel,
           let loaded = mlxModelID,
           service.normalizeModel(requestedModel) != loaded {
            req.logger.info("tokenize: requested model '\(requestedModel)' differs from loaded '\(loaded)'; tokenizing with the loaded one")
        }
        return try await service.tokenize(text: text)
    }

    private static func decodeBody(_ req: Request) throws -> TokenizeRequest {
        let reqId = req.afmRequestID
        do {
            let parsed = try req.content.decode(TokenizeRequest.self)
            guard parsed.effectiveText.isEmpty == false else {
                throw TokenizeBadRequestError(
                    message: "request must include `text` (or vLLM-style `prompt`) — `messages` form is not yet supported",
                    requestId: reqId
                )
            }
            return parsed
        } catch let err as TokenizeBadRequestError {
            throw err
        } catch {
            throw TokenizeBadRequestError(
                message: "invalid tokenize request body: \(error.localizedDescription)",
                requestId: reqId
            )
        }
    }

    private static func jsonResponse<T: Content>(_ payload: T) throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
    }
}

// MARK: - Request / response shapes

struct TokenizeRequest: Content {
    let model: String?
    /// Anthropic / OpenAI style.
    let text: String?
    /// vLLM style alias for `text`.
    let prompt: String?

    enum CodingKeys: String, CodingKey {
        case model, text, prompt
    }

    /// First non-empty of `text`, then `prompt`.
    var effectiveText: String {
        if let text, !text.isEmpty { return text }
        if let prompt, !prompt.isEmpty { return prompt }
        return ""
    }
}

struct TokenizeResponse: Content {
    let tokens: [Int]
    let count: Int
    let model: String
    /// vLLM-style context window hint, if known. Helps clients budget.
    let maxModelLen: Int?

    enum CodingKeys: String, CodingKey {
        case tokens, count, model
        case maxModelLen = "max_model_len"
    }
}

struct CountTokensResponse: Content {
    let inputTokens: Int
    let model: String

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case model
    }
}

// MARK: - Errors rendered in OpenAI shape

/// 422 — endpoint requires an MLX tokenizer. (T1.6)
struct TokenizeUnsupportedError: AbortError {
    let status: HTTPResponseStatus = .unprocessableEntity
    let reason: String = "tokenize endpoints require an MLX model — Foundation backend has no public tokenizer"
    let requestId: String

    static let errorType = "tokenize_unsupported"

    /// Vapor renders `AbortError` via its default ErrorMiddleware as
    /// `{"error": true, "reason": "..."}`. To get OpenAI shape we install a
    /// custom error middleware (see Server.swift) that intercepts these.
}

/// 400 — bad tokenize input. (T1.6)
struct TokenizeBadRequestError: AbortError {
    let status: HTTPResponseStatus = .badRequest
    let reason: String
    let requestId: String

    init(message: String, requestId: String) {
        self.reason = message
        self.requestId = requestId
    }

    static let errorType = "invalid_request_error"
}
