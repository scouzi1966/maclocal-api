import XCTest
import Vapor
import XCTVapor
import AFMKit

@testable import AFMServer

final class ChatTransportFrontierTests: XCTestCase {
    func testSharedChoiceFanoutInvokesHandlerOncePerChoice() async throws {
        let app = try await Application.make(.testing)
        let body = ByteBuffer(string: #"{"model":"foundation","n":3,"stream":false,"messages":[{"role":"user","content":"Hi"}]}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: String(body.readableBytes))
        let request = Request(
            application: app,
            method: .POST,
            url: URI(path: "/v1/chat/completions"),
            headersNoUpdate: headers,
            collectedBody: body,
            on: app.eventLoopGroup.next()
        )
        var invocationCount = 0

        let response = try await generateChatChoices(request: request, count: 3) { _ in
            invocationCount += 1
            let generated = ChatCompletionResponse(
                model: "foundation",
                content: "choice-\(invocationCount)",
                promptTokens: 2,
                completionTokens: 1
            )
            let response = Response(status: .ok)
            try response.content.encode(generated)
            return response
        }

        XCTAssertEqual(invocationCount, 3)
        guard let responseData = response.body.data else {
            return XCTFail("missing fan-out response body")
        }
        let object = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
        let choices = object?["choices"] as? [[String: Any]]
        XCTAssertEqual(choices?.compactMap { $0["index"] as? Int }, [0, 1, 2])
        XCTAssertEqual(
            choices?.compactMap { ($0["message"] as? [String: Any])?["content"] as? String },
            ["choice-1", "choice-2", "choice-3"]
        )
        try await app.asyncShutdown()
    }

    func testFixedWindowLimiterTracksRealRemainingQuotaAndReset() async {
        let limiter = ChatRateLimiter(configuration: .init(
            requestLimit: 2,
            windowSeconds: 10
        ))
        let start = Date(timeIntervalSince1970: 1_000)

        let first = await limiter.consume(key: "client", now: start)
        XCTAssertTrue(first.allowed)
        XCTAssertEqual(first.limit, 2)
        XCTAssertEqual(first.remaining, 1)
        XCTAssertEqual(first.resetAfter, 10, accuracy: 0.001)

        let second = await limiter.consume(
            key: "client",
            now: start.addingTimeInterval(2)
        )
        XCTAssertTrue(second.allowed)
        XCTAssertEqual(second.remaining, 0)
        XCTAssertEqual(second.resetAfter, 8, accuracy: 0.001)

        let exhausted = await limiter.consume(
            key: "client",
            now: start.addingTimeInterval(3)
        )
        XCTAssertFalse(exhausted.allowed)
        XCTAssertEqual(exhausted.remaining, 0)
        XCTAssertEqual(exhausted.resetAfter, 7, accuracy: 0.001)

        let reset = await limiter.consume(
            key: "client",
            now: start.addingTimeInterval(10)
        )
        XCTAssertTrue(reset.allowed)
        XCTAssertEqual(reset.remaining, 1)
        XCTAssertEqual(reset.resetAfter, 10, accuracy: 0.001)
    }

    func testRateLimitMiddlewareReturnsAccurateHeadersAndOpenAIError() async throws {
        let app = try await Application.make(.testing)
        app.middleware.use(RequestIDMiddleware())
        app.middleware.use(ChatRateLimitMiddleware(configuration: .init(
            requestLimit: 1,
            windowSeconds: 60
        )))
        app.post("v1", "chat", "completions") { _ in
            Response(status: .ok)
        }

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions"
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(response.headers.first(name: "X-RateLimit-Limit-Requests"), "1")
            XCTAssertEqual(response.headers.first(name: "X-RateLimit-Remaining-Requests"), "0")
            XCTAssertTrue(
                response.headers.first(name: "X-RateLimit-Reset-Requests")?.hasSuffix("ms") == true
            )
        }

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions"
        ) { response async in
            XCTAssertEqual(response.status, .tooManyRequests)
            XCTAssertEqual(response.headers.first(name: "X-RateLimit-Limit-Requests"), "1")
            XCTAssertEqual(response.headers.first(name: "X-RateLimit-Remaining-Requests"), "0")
            XCTAssertNotNil(response.headers.first(name: .retryAfter))
            let object = try? JSONSerialization.jsonObject(
                with: Data(buffer: response.body)
            ) as? [String: Any]
            let error = object?["error"] as? [String: Any]
            XCTAssertEqual(error?["type"] as? String, "rate_limit_error")
            XCTAssertEqual(error?["code"] as? String, "rate_limit_exceeded")
        }
        try await app.asyncShutdown()
    }

    func testRateLimitMiddlewareCoversResponsesButNotReadOnlyRoutes() async throws {
        let app = try await Application.make(.testing)
        app.middleware.use(ChatRateLimitMiddleware(configuration: .init(
            requestLimit: 2,
            windowSeconds: 60
        )))
        app.post("v1", "responses") { _ in Response(status: .ok) }
        app.get("v1", "models") { _ in Response(status: .ok) }

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/responses") { response async in
            XCTAssertEqual(response.headers.first(name: "X-RateLimit-Remaining-Requests"), "1")
        }
        try await app.testable(method: .running(port: 0)).test(.GET, "/v1/models") { response async in
            XCTAssertNil(response.headers.first(name: "X-RateLimit-Limit-Requests"))
        }
        try await app.asyncShutdown()
    }
}
