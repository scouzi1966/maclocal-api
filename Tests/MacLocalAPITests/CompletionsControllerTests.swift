@testable import AFMServer
import AFMKit
import AFMOpenAICompat
import Foundation
import OSLog
import Vapor
import XCTVapor
import XCTest

final class CompletionsControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testNonStreamingPreservesRawPromptAndParametersAndReturnsExactUsage() async throws {
        let capture = RawRequestCapture()
        let generator = AnyAFMRawTextGenerator { request in
            capture.store(request)
            return Self.events([
                .textDelta(text: "raw answer", tokenID: 7, timestamp: 1),
                .completed(.init(
                    finishReason: .length,
                    promptTokens: 3,
                    completionTokens: 2,
                    totalTokens: 5
                )),
            ])
        }
        try CompletionsController(modelID: "test-model", generator: generator).boot(routes: app)

        let body = ByteBuffer(string: #"{"model":"test-model","prompt":"<raw>\n prompt","max_tokens":9,"temperature":0.2,"top_p":0.8,"top_k":12,"min_p":0.05,"repetition_penalty":1.1,"presence_penalty":0.3,"seed":42,"stop":["END"],"ignore_eos":true}"#)
        try await app.testable().test(.POST, "/v1/completions", headers: jsonHeaders, body: body) { response async throws in
            XCTAssertEqual(response.status, .ok)
            let decoded = try response.content.decode(CompletionResponse.self)
            XCTAssertEqual(decoded.choices.first?.text, "raw answer")
            XCTAssertEqual(decoded.choices.first?.finishReason, "length")
            XCTAssertEqual(decoded.usage?.promptTokens, 3)
            XCTAssertEqual(decoded.usage?.completionTokens, 2)
            XCTAssertEqual(decoded.usage?.totalTokens, 5)
        }

        let request = try XCTUnwrap(capture.value)
        XCTAssertEqual(request.prompt, "<raw>\n prompt")
        XCTAssertEqual(request.maximumOutputTokens, 9)
        XCTAssertEqual(request.temperature, 0.2)
        XCTAssertEqual(request.topP, 0.8)
        XCTAssertEqual(request.topK, 12)
        XCTAssertEqual(request.minP, 0.05)
        XCTAssertEqual(request.repetitionPenalty, 1.1)
        XCTAssertEqual(request.presencePenalty, 0.3)
        XCTAssertEqual(request.seed, 42)
        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertTrue(request.ignoreEndOfSequence)
    }

    func testStreamingFramesChunksFinalUsageAndDoneAndReleasesLeaseOnce() async throws {
        let releases = OSAllocatedUnfairLock(initialState: 0)
        let admitter = AnyAFMGenerationAdmitter { _ in
            AFMGenerationLease(
                telemetryToken: AFMInferenceRequestToken(rawValue: UUID()),
                release: { releases.withLock { $0 += 1 } }
            )
        }
        let generator = AnyAFMRawTextGenerator { _ in
            Self.events([
                .textDelta(text: "one", tokenID: 1, timestamp: 1),
                .textDelta(text: " two", tokenID: 2, timestamp: 2),
                .completed(.init(
                    finishReason: .stop,
                    promptTokens: 4,
                    completionTokens: 2,
                    totalTokens: 6
                )),
            ])
        }
        try CompletionsController(
            modelID: "test-model",
            generator: generator,
            generationAdmitter: admitter
        ).boot(routes: app)

        let body = ByteBuffer(string: #"{"prompt":"raw","stream":true,"stream_options":{"include_usage":true}}"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/completions",
            headers: jsonHeaders,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            let wire = response.body.string
            XCTAssertTrue(wire.contains(#""text":"one""#))
            XCTAssertTrue(wire.contains(#""text":" two""#))
            XCTAssertTrue(wire.contains(#""finish_reason":"stop""#))
            XCTAssertTrue(wire.contains(#""prompt_tokens":4"#))
            XCTAssertTrue(wire.contains("data: [DONE]\n\n"))
        }
        XCTAssertEqual(releases.withLock { $0 }, 1)
    }

    func testProviderFailureReturnsOpenAIErrorForNonStreamingRequest() async throws {
        let generator = AnyAFMRawTextGenerator { _ in
            Self.events([.failed(reason: .inference, message: "provider failed")])
        }
        try CompletionsController(modelID: "test-model", generator: generator).boot(routes: app)

        let body = ByteBuffer(string: #"{"prompt":"raw"}"#)
        try await app.testable().test(.POST, "/v1/completions", headers: jsonHeaders, body: body) { response async in
            XCTAssertEqual(response.status, .internalServerError)
            XCTAssertTrue(response.body.string.contains("provider failed"))
        }
    }

    func testValidationRejectsPromptArraysModelMismatchAndContinuousUsage() async throws {
        let generator = AnyAFMRawTextGenerator { _ in Self.events([]) }
        try CompletionsController(modelID: "test-model", generator: generator).boot(routes: app)

        let cases: [(String, HTTPStatus, String)] = [
            (#"{"prompt":["a","b"]}"#, .badRequest, "unsupported_prompt_array"),
            (#"{"model":"other","prompt":"a"}"#, .notFound, "model_not_found"),
            (#"{"prompt":"a","stream":true,"stream_options":{"continuous_usage_stats":true}}"#, .badRequest, "continuous_usage_stats"),
        ]
        for (json, status, marker) in cases {
            try await app.testable().test(
                .POST,
                "/v1/completions",
                headers: jsonHeaders,
                body: ByteBuffer(string: json)
            ) { response async in
                XCTAssertEqual(response.status, status)
                XCTAssertTrue(response.body.string.contains(marker))
            }
        }
    }

    func testMiddlewareTracksOnlyNonStreamingCompletionRequests() {
        XCTAssertTrue(ActiveConnectionsMiddleware.shouldTrackInMiddleware(
            path: "/v1/completions",
            stream: false
        ))
        XCTAssertFalse(ActiveConnectionsMiddleware.shouldTrackInMiddleware(
            path: "/v1/completions",
            stream: true
        ))
        XCTAssertTrue(ActiveConnectionsMiddleware.shouldTrackInMiddleware(
            path: "/v1/chat/completions",
            stream: nil
        ))
        XCTAssertFalse(ActiveConnectionsMiddleware.shouldTrackInMiddleware(
            path: "/v1/chat/completions",
            stream: true
        ))
        XCTAssertFalse(ActiveConnectionsMiddleware.shouldTrackInMiddleware(
            path: "/metrics",
            stream: false
        ))
    }

    private var jsonHeaders: HTTPHeaders {
        var headers = HTTPHeaders()
        headers.contentType = .json
        return headers
    }

    private static func events(
        _ values: [AFMRawTextGenerationEvent]
    ) -> AsyncStream<AFMRawTextGenerationEvent> {
        AsyncStream { continuation in
            for value in values {
                continuation.yield(value)
            }
            continuation.finish()
        }
    }
}

private final class RawRequestCapture: @unchecked Sendable {
    private let storage = OSAllocatedUnfairLock<AFMRawTextGenerationRequest?>(initialState: nil)

    var value: AFMRawTextGenerationRequest? {
        storage.withLock { $0 }
    }

    func store(_ request: AFMRawTextGenerationRequest) {
        storage.withLock { $0 = request }
    }
}
