import Vapor
import Foundation

/// `POST /v1/chat/completions/{id}/cancel` — agent-friendly cancellation. (T1.5)
///
/// Looks up the request id in the application's `InflightRequestRegistry` and
/// fires the registered cancel closure. The closure cancels the streaming task
/// (which propagates to the BatchScheduler / MLX generator via cooperative
/// `Task.isCancelled` checks).
///
/// Response shape mirrors OpenAI's batch-cancel object:
/// ```json
/// { "id": "req_…", "object": "chat.completion.cancellation", "cancelled": true }
/// ```
///
/// Returns 200 when the id was found and cancellation was triggered, 404 when
/// the id is unknown (already completed, never existed, or expired).
struct CancelController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "chat", "completions", ":id", "cancel", use: cancel)
        v1.on(.OPTIONS, "chat", "completions", ":id", "cancel", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    func cancel(req: Request) async throws -> Response {
        guard let id = req.parameters.get("id"),
              !id.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw Abort(.badRequest, reason: "missing or empty id parameter")
        }

        let cancelled = await req.inflightRegistry.cancel(id: id)
        let payload = CancelResponse(id: id, cancelled: cancelled)
        let response = Response(status: cancelled ? .ok : .notFound)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
    }

    struct CancelResponse: Content {
        let id: String
        let object: String
        let cancelled: Bool

        init(id: String, cancelled: Bool) {
            self.id = id
            self.object = "chat.completion.cancellation"
            self.cancelled = cancelled
        }
    }
}
