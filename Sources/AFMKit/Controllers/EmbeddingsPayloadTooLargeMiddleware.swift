import Vapor

/// Converts Vapor's `.payloadTooLarge` aborts into an OpenAI-shaped error body
/// for the embeddings endpoint (so clients get a structured JSON error, not a bare 413).
public struct EmbeddingsPayloadTooLargeMiddleware: AsyncMiddleware {
    public init() {}

    public func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let abort as Abort where abort.status == .payloadTooLarge {
            let errorResponse = OpenAIError(
                message: "Embeddings request body exceeds the configured size limit.",
                type: "payload_too_large"
            )
            let response = Response(status: .payloadTooLarge)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(errorResponse)
            return response
        }
    }
}
