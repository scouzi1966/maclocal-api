import Vapor
import AFMKit
import Foundation

private struct MultiChatCompletionResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
    let timings: StreamTimings?
    let systemFingerprint: String?
    let afmProfile: AFMProfile?
    let afmProfileExtended: AFMProfileExtended?

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case model
        case choices
        case usage
        case timings
        case systemFingerprint = "system_fingerprint"
        case afmProfile = "afm_profile"
        case afmProfileExtended = "afm_profile_extended"
    }
}

/// Runs one real non-streaming generation per requested choice and combines
/// the resulting OpenAI envelopes. Keeping this at the HTTP boundary avoids
/// changing the immutable AFMKit provider contract while ensuring `n` is not a
/// transport-only decoration.
func generateChatChoices(
    request: Request,
    count: Int,
    handler: (Request) async throws -> Response
) async throws -> Response {
    guard let body = request.body.data else {
        throw Abort(.badRequest, reason: "Request body is required.")
    }
    let sourceData = Data(buffer: body)
    guard var object = try JSONSerialization.jsonObject(with: sourceData) as? [String: Any] else {
        throw Abort(.badRequest, reason: "Request body must be a JSON object.")
    }
    object["n"] = 1
    object["stream"] = false
    let singleChoiceData = try JSONSerialization.data(withJSONObject: object)

    var completions: [ChatCompletionResponse] = []
    var firstHTTPResponse: Response?
    completions.reserveCapacity(count)

    for _ in 0..<count {
        var headers = request.headers
        headers.replaceOrAdd(name: .contentLength, value: String(singleChoiceData.count))
        var childBody = request.byteBufferAllocator.buffer(capacity: singleChoiceData.count)
        childBody.writeBytes(singleChoiceData)
        let child = Request(
            application: request.application,
            method: request.method,
            url: request.url,
            version: request.version,
            headersNoUpdate: headers,
            collectedBody: childBody,
            remoteAddress: request.remoteAddress,
            logger: request.logger,
            byteBufferAllocator: request.byteBufferAllocator,
            on: request.eventLoop
        )
        if !request.afmRequestID.isEmpty {
            await child.storage.setWithAsyncShutdown(
                RequestIDKey.self,
                to: request.afmRequestID
            )
        }

        let response = try await handler(child)
        guard response.status.code >= 200, response.status.code < 300 else {
            return response
        }
        guard let responseBody = response.body.data else {
            throw Abort(.internalServerError, reason: "Choice generation returned an empty response.")
        }
        let completion = try JSONDecoder().decode(
            ChatCompletionResponse.self,
            from: responseBody
        )
        guard completion.choices.count == 1 else {
            throw Abort(
                .internalServerError,
                reason: "Single-choice generation returned \(completion.choices.count) choices."
            )
        }
        if firstHTTPResponse == nil {
            firstHTTPResponse = response
        }
        completions.append(completion)
    }

    guard let first = completions.first, let response = firstHTTPResponse else {
        throw Abort(.internalServerError, reason: "No completion choices were generated.")
    }

    let choices = completions.enumerated().map { index, completion in
        let choice = completion.choices[0]
        return Choice(
            index: index,
            message: choice.message,
            logprobs: choice.logprobs,
            finishReason: choice.finishReason
        )
    }
    let promptTokens = completions.reduce(0) { $0 + $1.usage.promptTokens }
    let completionTokens = completions.reduce(0) { $0 + $1.usage.completionTokens }
    let cachedTokenValues = completions.compactMap { $0.usage.promptTokensDetails?.cachedTokens }
    let completionTimes = completions.compactMap(\.usage.completionTime)
    let promptTimes = completions.compactMap(\.usage.promptTime)
    let peakMemoryValues = completions.compactMap(\.usage.peakMemoryGib)
    let usage = Usage(
        promptTokens: promptTokens,
        completionTokens: completionTokens,
        totalTokens: promptTokens + completionTokens,
        cachedTokens: cachedTokenValues.isEmpty ? nil : cachedTokenValues.reduce(0, +),
        completionTime: completionTimes.isEmpty ? nil : completionTimes.reduce(0, +),
        promptTime: promptTimes.isEmpty ? nil : promptTimes.reduce(0, +),
        peakMemoryGib: peakMemoryValues.max()
    )
    let envelope = MultiChatCompletionResponse(
        id: first.id,
        object: first.object,
        created: first.created,
        model: first.model,
        choices: choices,
        usage: usage,
        timings: first.timings,
        systemFingerprint: first.systemFingerprint,
        afmProfile: first.afmProfile,
        afmProfileExtended: first.afmProfileExtended
    )
    try response.content.encode(envelope)
    return response
}

struct ChatRateLimitConfiguration: Sendable, Equatable {
    static let defaultRequestLimit = 600
    static let defaultWindowSeconds: TimeInterval = 60

    let requestLimit: Int
    let windowSeconds: TimeInterval

    var isEnabled: Bool { requestLimit > 0 }
}

actor ChatRateLimiter {
    struct Decision: Sendable, Equatable {
        let allowed: Bool
        let limit: Int
        let remaining: Int
        let resetAfter: TimeInterval
    }

    private struct Bucket {
        var startedAt: Date
        var used: Int
    }

    private let configuration: ChatRateLimitConfiguration
    private var buckets: [String: Bucket] = [:]

    init(configuration: ChatRateLimitConfiguration) {
        self.configuration = configuration
    }

    func consume(key: String, now: Date = Date()) -> Decision {
        precondition(configuration.requestLimit > 0)
        precondition(configuration.windowSeconds > 0)

        var bucket = buckets[key] ?? Bucket(startedAt: now, used: 0)
        let elapsed = now.timeIntervalSince(bucket.startedAt)
        if elapsed >= configuration.windowSeconds || elapsed < 0 {
            bucket = Bucket(startedAt: now, used: 0)
        }

        let allowed = bucket.used < configuration.requestLimit
        if allowed {
            bucket.used += 1
        }
        buckets[key] = bucket

        return Decision(
            allowed: allowed,
            limit: configuration.requestLimit,
            remaining: max(0, configuration.requestLimit - bucket.used),
            resetAfter: max(0, configuration.windowSeconds - now.timeIntervalSince(bucket.startedAt))
        )
    }
}

struct ChatRateLimitMiddleware: AsyncMiddleware {
    private let limiter: ChatRateLimiter

    init(configuration: ChatRateLimitConfiguration) {
        self.limiter = ChatRateLimiter(configuration: configuration)
    }

    static func applies(to request: Request) -> Bool {
        guard request.method == .POST else { return false }
        return request.url.path == "/v1/chat/completions"
            || request.url.path == "/v1/responses"
            || request.url.path == "/v1/messages"
    }

    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        guard Self.applies(to: request) else {
            return try await next.respond(to: request)
        }

        let clientKey = request.remoteAddress?.ipAddress ?? "local"
        let decision = await limiter.consume(key: clientKey)
        if !decision.allowed {
            let response = Response(status: .tooManyRequests)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            response.headers.replaceOrAdd(
                name: .retryAfter,
                value: String(max(1, Int(ceil(decision.resetAfter))))
            )
            try response.content.encode(OpenAIError(
                message: "Rate limit exceeded. Retry after the current window resets.",
                type: "rate_limit_error",
                code: "rate_limit_exceeded",
                requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
            ))
            addHeaders(for: decision, to: response)
            return response
        }

        let response = try await next.respond(to: request)
        addHeaders(for: decision, to: response)
        return response
    }

    private func addHeaders(for decision: ChatRateLimiter.Decision, to response: Response) {
        response.headers.replaceOrAdd(
            name: "X-RateLimit-Limit-Requests",
            value: String(decision.limit)
        )
        response.headers.replaceOrAdd(
            name: "X-RateLimit-Remaining-Requests",
            value: String(decision.remaining)
        )
        let resetMilliseconds = max(1, Int(ceil(decision.resetAfter * 1_000)))
        response.headers.replaceOrAdd(
            name: "X-RateLimit-Reset-Requests",
            value: "\(resetMilliseconds)ms"
        )
    }
}
