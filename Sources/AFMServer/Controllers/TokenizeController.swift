import Vapor
import AFMKit
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
    private let tokenizer: (any AFMTextTokenizing)?
    private let contextWindow: Int?

    init(
        mlxModelID: String?,
        tokenizer: (any AFMTextTokenizing)?,
        contextWindow: Int?
    ) {
        self.mlxModelID = mlxModelID
        self.tokenizer = tokenizer
        self.contextWindow = contextWindow
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "tokenize", body: .collect(maxSize: "8mb"), use: tokenize)
        v1.on(.OPTIONS, "tokenize", use: handleOptions)
        v1.on(.POST, "count_tokens", body: .collect(maxSize: "8mb"), use: countTokens)
        v1.on(.OPTIONS, "count_tokens", use: handleOptions)
        // Anthropic-compatible token budgeting surface.  This intentionally
        // lives beside the existing generic endpoint: clients using the
        // Messages API address this path, while OpenAI/vLLM clients use
        // /v1/count_tokens or /v1/tokenize.
        v1.on(.POST, "messages", "count_tokens", body: .collect(maxSize: "8mb"), use: countMessageTokens)
        v1.on(.OPTIONS, "messages", "count_tokens", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        // Browser agents that pass X-Request-ID / OpenAI-Request-ID need the
        // preflight to whitelist them or the actual request fails with CORS. (T1.1/T1.6)
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization, X-Request-ID, OpenAI-Request-ID")
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

    /// Anthropic's `POST /v1/messages/count_tokens` accepts a conversation
    /// rather than one flat prompt. AFM's public tokenizer capability only
    /// exposes text tokenization, so we preserve each textual block in order
    /// and tokenize the resulting transcript. This is deliberately useful for
    /// context budgeting without claiming exact model-template accounting;
    /// template-exact counting requires a new AFMKit tokenizer capability.
    func countMessageTokens(req: Request) async throws -> Response {
        let body: MessagesCountTokensRequest
        do {
            body = try req.content.decode(MessagesCountTokensRequest.self)
        } catch {
            throw TokenizeBadRequestError(
                message: "invalid messages/count_tokens request body: \(error.localizedDescription)",
                requestId: req.afmRequestID
            )
        }
        guard !body.effectiveText.isEmpty else {
            throw TokenizeBadRequestError(
                message: "request must include at least one textual message or system block",
                requestId: req.afmRequestID
            )
        }
        let tokens = try await encode(body.effectiveText, requestedModel: body.model, on: req)
        return try Self.jsonResponse(CountTokensResponse(
            inputTokens: tokens.count,
            model: mlxModelID ?? body.model ?? "unknown"
        ))
    }

    // MARK: - Internals

    private func encode(_ text: String, requestedModel: String?, on req: Request) async throws -> [Int] {
        guard let tokenizer else {
            throw TokenizeUnsupportedError(requestId: req.afmRequestID)
        }
        // Optional sanity check: if a specific model id was requested, warn if
        // it doesn't match the loaded one (don't fail — agents often pass aliases).
        if let requestedModel,
           let loaded = mlxModelID,
           requestedModel != loaded {
            req.logger.info("tokenize: requested model '\(requestedModel)' differs from loaded '\(loaded)'; tokenizing with the loaded one")
        }
        do {
            return try await tokenizer.tokenize(text: text)
        } catch AFMError.unsupportedCapability {
            throw TokenizeUnsupportedError(requestId: req.afmRequestID)
        }
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

// MARK: - Anthropic Messages token-count request

/// The subset of Anthropic Messages content needed for token budgeting. It
/// accepts the normal string shorthand and arrays of content blocks, retaining
/// text/thinking blocks and safely ignoring non-text media/tool payloads.
struct MessagesCountTokensRequest: Content {
    let model: String?
    let system: MessagesCountContent?
    let messages: [MessagesCountMessage]

    var effectiveText: String {
        let parts = (system.map { [$0.text] } ?? []) + messages.map(\.content.text)
        return parts
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }
}

struct MessagesCountMessage: Content {
    let role: String
    let content: MessagesCountContent
}

enum MessagesCountContent: Content {
    case text(String)
    case blocks([MessagesCountBlock])

    var text: String {
        switch self {
        case .text(let value): return value
        case .blocks(let blocks):
            return blocks.compactMap(\.text).joined(separator: "\n")
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .blocks(try container.decode([MessagesCountBlock].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let value): try container.encode(value)
        case .blocks(let blocks): try container.encode(blocks)
        }
    }
}

struct MessagesCountBlock: Content {
    let type: String
    let text: String?
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
