import Foundation
import Vapor
import XCTest
import XCTVapor

@testable import AFMKitCore
@testable import AFMOpenAICompat
@testable import AFMServer

final class LegacyCompletionsControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testNonStreamingCompletionUsesRawPromptAndExactTerminalUsage() async throws {
        let spy = RawGeneratorSpy(events: [
            .textDelta(text: "raw ", tokenID: 10, timestamp: 1),
            .textDelta(text: "answer", tokenID: 11, timestamp: 2),
            .completed(.init(
                finishReason: .length,
                promptTokens: 7,
                completionTokens: 2,
                totalTokens: 9
            )),
        ])
        try register(spy)

        let body = #"""
        {
          "model":"test-model",
          "prompt":"Do not template me",
          "max_tokens":2,
          "temperature":0.2,
          "top_p":0.8,
          "top_k":4,
          "min_p":0.1,
          "repetition_penalty":1.1,
          "presence_penalty":0.3,
          "seed":42,
          "stop":["END"],
          "ignore_eos":true
        }
        """#

        try await post(body) { response in
            XCTAssertEqual(response.status, .ok)
            let json = try Self.json(response.body.string)
            XCTAssertEqual(json["object"] as? String, "text_completion")
            XCTAssertEqual(json["model"] as? String, "test-model")
            let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
            XCTAssertEqual(choices.first?["text"] as? String, "raw answer")
            XCTAssertEqual(choices.first?["finish_reason"] as? String, "length")
            let usage = try XCTUnwrap(json["usage"] as? [String: Any])
            XCTAssertEqual(usage["prompt_tokens"] as? Int, 7)
            XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
            XCTAssertEqual(usage["total_tokens"] as? Int, 9)
        }

        let request = try XCTUnwrap(spy.lastRequest)
        XCTAssertEqual(request.prompt, "Do not template me")
        XCTAssertEqual(request.modelID, "test-model")
        XCTAssertEqual(request.maximumOutputTokens, 2)
        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertEqual(request.temperature, 0.2)
        XCTAssertEqual(request.topP, 0.8)
        XCTAssertEqual(request.topK, 4)
        XCTAssertEqual(request.minP, 0.1)
        XCTAssertEqual(request.repetitionPenalty, 1.1)
        XCTAssertEqual(request.presencePenalty, 0.3)
        XCTAssertEqual(request.seed, 42)
        XCTAssertTrue(request.ignoreEndOfSequence)
    }

    func testStreamingCompletionEmitsTextFinishFinalUsageAndDoneInOrder() async throws {
        let spy = RawGeneratorSpy(events: [
            .textDelta(text: "one", tokenID: 1, timestamp: 1),
            .textDelta(text: " two", tokenID: 2, timestamp: 2),
            .completed(.init(
                finishReason: .stop,
                promptTokens: 3,
                completionTokens: 2,
                totalTokens: 5
            )),
        ])
        try register(spy)

        let body = #"""
        {
          "model":"test-model",
          "prompt":"prompt",
          "stream":true,
          "stream_options":{"include_usage":true,"continuous_usage_stats":true}
        }
        """#

        try await post(body) { response in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(response.headers.contentType?.type, "text")
            XCTAssertEqual(response.headers.contentType?.subType, "event-stream")
            let payloads = Self.ssePayloads(response.body.string)
            XCTAssertEqual(payloads.count, 5)
            XCTAssertEqual(payloads.last, "[DONE]")
            XCTAssertTrue(payloads[0].contains(#""text":"one""#))
            XCTAssertTrue(payloads[1].contains(#""text":" two""#))
            XCTAssertTrue(payloads[2].contains(#""finish_reason":"stop""#))
            XCTAssertTrue(payloads[3].contains(#""choices":[]"#))
            XCTAssertTrue(payloads[3].contains(#""total_tokens":5"#))
            XCTAssertFalse(response.body.string.contains(#""delta""#))
            XCTAssertFalse(response.body.string.contains(#""role""#))
            XCTAssertEqual(
                payloads.dropLast().filter { $0.contains(#""usage""#) }.count,
                1
            )
        }
    }

    func testContinuousUsageStatsDoesNotOverrideIncludeUsageFalse() async throws {
        let spy = RawGeneratorSpy(events: [
            .completed(.init(
                finishReason: .stop,
                promptTokens: 1,
                completionTokens: 0,
                totalTokens: 1
            )),
        ])
        try register(spy)

        let body = #"""
        {
          "prompt":"prompt",
          "stream":true,
          "stream_options":{"include_usage":false,"continuous_usage_stats":true}
        }
        """#
        try await post(body) { response in
            let payloads = Self.ssePayloads(response.body.string)
            XCTAssertEqual(payloads.count, 2)
            XCTAssertTrue(payloads[0].contains(#""finish_reason":"stop""#))
            XCTAssertEqual(payloads[1], "[DONE]")
            XCTAssertFalse(response.body.string.contains(#""usage""#))
        }
    }

    func testPromptArrayIsRejectedBeforeProviderAdmissionWithParam() async throws {
        let spy = RawGeneratorSpy(events: [])
        try register(spy)

        try await post(#"{"prompt":["one"]}"#) { response in
            XCTAssertEqual(response.status, .badRequest)
            let error = try Self.errorDetail(response.body.string)
            XCTAssertEqual(error["type"] as? String, "invalid_request_error")
            XCTAssertEqual(error["code"] as? String, "unsupported_prompt_array")
            XCTAssertEqual(error["param"] as? String, "prompt")
        }
        XCTAssertEqual(spy.callCount, 0)
    }

    func testUnsupportedOptionsAndUnknownModelAreRejectedBeforeAdmission() async throws {
        let spy = RawGeneratorSpy(events: [])
        try register(spy)

        let cases: [(String, HTTPResponseStatus, String)] = [
            (#"{"model":"other","prompt":"x"}"#, .notFound, "model"),
            (#"{"prompt":"x","echo":true}"#, .badRequest, "echo"),
            (#"{"prompt":"x","logprobs":1}"#, .badRequest, "logprobs"),
            (#"{"prompt":"x","n":2}"#, .badRequest, "n"),
            (#"{"prompt":"x","best_of":2}"#, .badRequest, "best_of"),
            (#"{"prompt":"x","stop":""}"#, .badRequest, "stop"),
        ]
        for (body, status, param) in cases {
            try await post(body) { response in
                XCTAssertEqual(response.status, status)
                let error = try Self.errorDetail(response.body.string)
                XCTAssertEqual(error["param"] as? String, param)
            }
        }
        XCTAssertEqual(spy.callCount, 0)
    }

    func testPostHeaderFailureEmitsOneErrorAndNoSuccessfulTerminal() async throws {
        let spy = RawGeneratorSpy(events: [
            .textDelta(text: "partial", tokenID: 1, timestamp: 1),
            .failed(reason: .inference, message: "forced failure"),
            .completed(.init(
                finishReason: .stop,
                promptTokens: 1,
                completionTokens: 1,
                totalTokens: 2
            )),
        ])
        try register(spy)

        try await post(#"{"prompt":"x","stream":true,"stream_options":{"include_usage":true}}"#) {
            response in
            let payloads = Self.ssePayloads(response.body.string)
            XCTAssertEqual(payloads.count, 2)
            XCTAssertTrue(payloads[0].contains(#""text":"partial""#))
            XCTAssertTrue(payloads[1].contains(#""error""#))
            XCTAssertTrue(payloads[1].contains("forced failure"))
            XCTAssertFalse(response.body.string.contains("[DONE]"))
            XCTAssertFalse(response.body.string.contains(#""finish_reason":"stop""#))
            XCTAssertFalse(response.body.string.contains(#""usage""#))
        }
    }

    func testMissingProviderTerminalEmitsStreamErrorWithoutDone() async throws {
        let spy = RawGeneratorSpy(events: [
            .textDelta(text: "partial", tokenID: 1, timestamp: 1),
        ])
        try register(spy)

        try await post(#"{"prompt":"x","stream":true}"#) { response in
            let payloads = Self.ssePayloads(response.body.string)
            XCTAssertEqual(payloads.count, 2)
            XCTAssertTrue(payloads[1].contains("missing_terminal_event"))
            XCTAssertFalse(response.body.string.contains("[DONE]"))
        }
    }

    private func register(_ spy: RawGeneratorSpy) throws {
        try CompletionsController(
            modelID: "test-model",
            generator: spy.generator
        ).boot(routes: app)
    }

    private func post(
        _ json: String,
        assertions: @escaping (XCTHTTPResponse) throws -> Void
    ) async throws {
        var headers = HTTPHeaders()
        headers.contentType = .json
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/completions",
            headers: headers,
            body: ByteBuffer(string: json)
        ) { response async in
            do {
                try assertions(response)
            } catch {
                XCTFail("assertion failed: \(error)\nbody: \(response.body.string)")
            }
        }
    }

    private static func json(_ body: String) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]
        )
    }

    private static func errorDetail(_ body: String) throws -> [String: Any] {
        try XCTUnwrap(try json(body)["error"] as? [String: Any])
    }

    private static func ssePayloads(_ body: String) -> [String] {
        body.components(separatedBy: "\n\n").compactMap { event in
            guard event.hasPrefix("data: ") else { return nil }
            return String(event.dropFirst("data: ".count))
        }
    }
}

private final class RawGeneratorSpy: @unchecked Sendable {
    private let lock = NSLock()
    private let events: [AFMRawTextGenerationEvent]
    private var requests: [AFMRawTextGenerationRequest] = []

    init(events: [AFMRawTextGenerationEvent]) {
        self.events = events
    }

    var generator: AnyAFMRawTextGenerator {
        AnyAFMRawTextGenerator { [self] request in
            lock.withLock { requests.append(request) }
            return AsyncStream { continuation in
                for event in events { continuation.yield(event) }
                continuation.finish()
            }
        }
    }

    var lastRequest: AFMRawTextGenerationRequest? {
        lock.withLock { requests.last }
    }

    var callCount: Int {
        lock.withLock { requests.count }
    }
}
