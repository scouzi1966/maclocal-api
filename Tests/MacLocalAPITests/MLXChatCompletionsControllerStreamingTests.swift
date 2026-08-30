import XCTest
import Vapor
import XCTVapor
import OSLog

@testable import AFMKit
import AFMKitMLX
@testable import AFMServer

final class MLXChatCompletionsControllerStreamingTests: XCTestCase {
// dimensions: streaming=true, execution=serial
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testStreamingStopFilterWithholdsDelimiterSplitAcrossChunks() {
        var filter = StreamingStopSequenceFilter(stopSequences: ["STOP"])

        XCTAssertEqual(filter.consume("answer ST"), "answer ")
        XCTAssertEqual(filter.consume("OP trailing"), "")
        XCTAssertTrue(filter.stopped)
        XCTAssertEqual(filter.flush(), "")
    }

    func testNonStreamingNProducesExactlyNGeneratedChoices() async throws {
        let service = FakeMLXChatService(
            generateResult: (
                modelID: "test-model",
                content: "Neon Badgers",
                promptTokens: 4,
                completionTokens: 2,
                tokenLogprobs: nil,
                toolCalls: nil,
                cachedTokens: 0,
                promptTime: 0.01,
                generateTime: 0.02,
                stoppedBySequence: false
            ),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = ByteBuffer(string: #"{"model":"test-model","n":2,"stream":false,"messages":[{"role":"user","content":"Invent a band name."}]}"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            let object = try? JSONSerialization.jsonObject(
                with: Data(buffer: response.body)
            ) as? [String: Any]
            let choices = object?["choices"] as? [[String: Any]]
            XCTAssertEqual(choices?.count, 2)
            XCTAssertEqual(choices?.compactMap { $0["index"] as? Int }, [0, 1])
            let usage = object?["usage"] as? [String: Any]
            XCTAssertEqual(usage?["prompt_tokens"] as? Int, 8)
            XCTAssertEqual(usage?["completion_tokens"] as? Int, 4)
        }
        XCTAssertEqual(service.generateCount, 2)
    }

    func testStreamingNIsRejectedExplicitly() async throws {
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = ByteBuffer(string: #"{"model":"test-model","n":2,"stream":true,"messages":[{"role":"user","content":"Hi"}]}"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            XCTAssertContains(response.body.string, "unsupported_streaming_n")
        }
        XCTAssertEqual(service.generateCount, 0)
    }

    func testStreamingStopFilterFlushesUnmatchedPartialDelimiter() {
        var filter = StreamingStopSequenceFilter(stopSequences: ["STOP"])

        XCTAssertEqual(filter.consume("answer ST"), "answer ")
        XCTAssertEqual(filter.flush(), "ST")
        XCTAssertFalse(filter.stopped)
    }

    func testStreamingStopFilterHandlesUnicodeAndOverlappingStops() {
        var filter = StreamingStopSequenceFilter(stopSequences: ["🛑", "END"])

        XCTAssertEqual(filter.consume("one E"), "one ")
        XCTAssertEqual(filter.consume("ND two 🛑 three"), "")
        XCTAssertTrue(filter.stopped)
    }

    func testStreamingControllerNeverEmitsSplitStopDelimiter() async throws {
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "answer ST"),
                AFMServerStreamChunk(text: "OP trailing", stoppedBySequence: true),
                AFMServerStreamChunk(text: " ignored", promptTokens: 4, completionTokens: 5),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true, stopJSON: #"["STOP"]"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""content":"answer ""#)
            XCTAssertFalse(res.body.string.contains("STOP"))
            XCTAssertFalse(res.body.string.contains("trailing"))
            XCTAssertFalse(res.body.string.contains("ignored"))
            XCTAssertContains(res.body.string, #""finish_reason":"stop""#)
        }
    }

    func testProviderStopFlagSuppressesTrailingChunksAndToolCalls() async throws {
        let lateTool = ResponseToolCall(
            index: 0,
            id: "late",
            type: "function",
            function: .init(name: "get_weather", arguments: #"{"location":"late"}"#)
        )
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "visible", stoppedBySequence: true),
                AFMServerStreamChunk(
                    text: " trailing",
                    toolCalls: [lateTool],
                    promptTokens: 4,
                    completionTokens: 2,
                    promptTime: 0.1,
                    generateTime: 0.2
                ),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true, stopJSON: #"["STOP"]"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""content":"visible""#)
            XCTAssertFalse(res.body.string.contains("trailing"))
            XCTAssertFalse(res.body.string.contains("late"))
            XCTAssertFalse(res.body.string.contains("tool_calls"))
            XCTAssertContains(res.body.string, #""completion_tokens":2"#)
            XCTAssertContains(res.body.string, #""finish_reason":"stop""#)
        }
    }

    func testSplitStopAcrossThinkTagContentPreservesReasoning() async throws {
        let service = FakeMLXChatService(
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "<think>STOP in reasoning</think>visible ST"),
                AFMServerStreamChunk(text: "OP hidden", promptTokens: 4, completionTokens: 5),
            ])
        )
        try MLXChatCompletionsController(modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil)
            .boot(routes: app)

        let body = try requestBody(stream: true, stopJSON: #"["STOP"]"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertContains(res.body.string, "STOP in reasoning")
            XCTAssertContains(res.body.string, #""content":"visible ""#)
            XCTAssertFalse(res.body.string.contains("hidden"))
        }
    }

    func testSplitStopAcrossHarmonyFinalChannelPreservesReasoning() async throws {
        let service = FakeMLXChatService(
            responseChannelFormat: .harmony,
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "<|channel|>analysis<|message|>STOP in reasoning<|end|><|channel|>final<|message|>visible ST"),
                AFMServerStreamChunk(text: "OP hidden<|return|>", promptTokens: 4, completionTokens: 5),
            ])
        )
        try MLXChatCompletionsController(modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil)
            .boot(routes: app)

        let body = try requestBody(stream: true, stopJSON: #"["STOP"]"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertContains(res.body.string, "STOP in reasoning")
            XCTAssertContains(res.body.string, "visible ")
            XCTAssertFalse(res.body.string.contains("hidden"))
        }
    }

    func testSplitStopAcrossMuseFinalChannelPreservesReasoning() async throws {
        let service = FakeMLXChatService(
            responseChannelFormat: .muse,
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "to=self<|message|>STOP in reasoning<|eom|>to=user<|message|>visible ST"),
                AFMServerStreamChunk(text: "OP hidden<|return|>", promptTokens: 4, completionTokens: 5),
            ])
        )
        try MLXChatCompletionsController(modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil)
            .boot(routes: app)

        let body = try requestBody(stream: true, stopJSON: #"["STOP"]"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertContains(res.body.string, "STOP in reasoning")
            XCTAssertContains(res.body.string, "visible ")
            XCTAssertFalse(res.body.string.contains("hidden"))
        }
    }

    func testDeferredStructuredOutputAppliesSplitStopBeforeEmission() async throws {
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: #"{"answer":"visible ST"#),
                AFMServerStreamChunk(text: #"OP hidden"}"#, promptTokens: 4, completionTokens: 5),
            ])
        )
        try MLXChatCompletionsController(modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil)
            .boot(routes: app)

        let body = try requestBody(
            stream: true,
            responseFormatJSON: #"{"type":"json_object"}"#,
            stopJSON: #"["STOP"]"#
        )
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertFalse(res.body.string.contains("STOP"))
            XCTAssertFalse(res.body.string.contains("hidden"))
            XCTAssertContains(res.body.string, #""finish_reason":"stop""#)
        }
    }

    func testStreamingControllerSerializesProviderToolCallsIntoSSEToolCalls() async throws {
        let toolCall = ResponseToolCall(
            index: 0,
            id: "call_weather",
            type: "function",
            function: ResponseToolCallFunction(
                name: "get_weather",
                arguments: #"{"location":"Berlin"}"#
            )
        )
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCalls: [toolCall]),
                AFMServerStreamChunk(text: "", promptTokens: 14, completionTokens: 3, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.headers.contentType, .init(type: "text", subType: "event-stream"))
            XCTAssertContains(res.body.string, "\"tool_calls\"")
            XCTAssertContains(res.body.string, "\"get_weather\"")
            XCTAssertContains(res.body.string, "\\\"location\\\":\\\"Berlin\\\"")
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
            XCTAssertContains(res.body.string, "data: [DONE]")
        }
    }

    func testStreamingControllerSerializesDeepseekProviderToolCallWithoutLeakingDSML() async throws {
        let toolCall = ResponseToolCall(
            index: 0,
            id: "call_deepseek_weather",
            type: "function",
            function: ResponseToolCallFunction(
                name: "get_weather",
                arguments: #"{"location":"Toronto"}"#
            )
        )
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(
                chunks: [
                    AFMServerStreamChunk(text: "", toolCalls: [toolCall]),
                    AFMServerStreamChunk(text: "", promptTokens: 415, completionTokens: 44, cachedTokens: 0, promptTime: 1.5, generateTime: 1.8),
                ],
                toolCallStartTag: nil,
                toolCallEndTag: nil
            )
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true, toolChoiceJSON: "\"required\"")
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"tool_calls\"")
            XCTAssertContains(res.body.string, "\"get_weather\"")
            XCTAssertContains(res.body.string, "\\\"location\\\":\\\"Toronto\\\"")
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
            XCTAssertFalse(res.body.string.contains("DSML"))
        }
    }

    func testRawOutputPreservesStructuralTagsAtGenerationSource() async throws {
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "<|START_TEXT|>answer<|END_TEXT|>"),
                AFMServerStreamChunk(text: "", promptTokens: 4, completionTokens: 3, cachedTokens: 0, promptTime: 0.01, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil,
            rawOutput: true
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "<|START_TEXT|>answer<|END_TEXT|>")
        }
        XCTAssertEqual(service.recordedPreserveStructuralTags, [true])
    }

    func testStreamingControllerSerializesCompletedBatchToolCalls() async throws {
        let toolCall = ResponseToolCall(
            index: 0,
            id: "call_batch",
            type: "function",
            function: ResponseToolCallFunction(
                name: "read_file",
                arguments: #"{"path":"README.md"}"#
            )
        )
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCalls: [toolCall]),
                AFMServerStreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 4, promptTime: 0.03, generateTime: 0.02),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"tool_calls\"")
            XCTAssertContains(res.body.string, "\"read_file\"")
            XCTAssertContains(res.body.string, "\\\"path\\\":\\\"README.md\\\"")
            XCTAssertContains(res.body.string, "\"index\":0")
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerDoesNotDuplicateVendorCallAfterRawXMLStart() async throws {
        let vendorCall = ResponseToolCall(
            index: 0,
            id: "call_vendor",
            type: "function",
            function: ResponseToolCallFunction(
                name: "get_weather",
                arguments: #"{"location":"Berlin"}"#
            )
        )
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "<tool_call>"),
                AFMServerStreamChunk(text: "<function=get_weather>"),
                AFMServerStreamChunk(text: "<parameter=location>Berlin</parameter>"),
                AFMServerStreamChunk(text: "", toolCalls: [vendorCall]),
                AFMServerStreamChunk(text: "</tool_call>"),
                AFMServerStreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 0, promptTime: 0.03, generateTime: 0.02),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(
                res.body.string.components(separatedBy: #"\"location\":\"Berlin\""#).count - 1,
                1,
                "the vendor completion must not repeat arguments already owned by the raw XML runtime"
            )
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerSerializesBatchToolCallDeltasBeforeCompletedCall() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: "call_batch",
                        type: "function",
                        function: StreamDeltaFunction(
                            name: "read_file",
                            arguments: "{\"path\":\"README.md\"}"
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCalls: [
                    ResponseToolCall(
                        index: 0,
                        id: "call_batch",
                        type: "function",
                        function: ResponseToolCallFunction(
                            name: "read_file",
                            arguments: #"{"path":"README.md"}"#
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 4, promptTime: 0.03, generateTime: 0.02),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"tool_calls\"")
            XCTAssertContains(res.body.string, "\"id\":\"call_batch\"")
            XCTAssertContains(res.body.string, "\"name\":\"read_file\"")
            XCTAssertContains(res.body.string, "\\\"path\\\":\\\"README.md\\\"")
            XCTAssertEqual(
                res.body.string.components(separatedBy: "\\\"path\\\":\\\"README.md\\\"").count - 1,
                1,
                "completed batch tool calls must not repeat arguments already emitted as deltas"
            )
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerSuppressesDuplicateCompletedArgumentClose() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: "call_weather",
                        type: "function",
                        function: StreamDeltaFunction(name: "get_weather", arguments: nil)
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(name: nil, arguments: #"{"location":"Sydney""#)
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(name: nil, arguments: #","unit":"celsius""#)
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(name: nil, arguments: "}")
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(name: nil, arguments: "}")
                    )
                ]),
                AFMServerStreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 0, promptTime: 0.03, generateTime: 0.02),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            let argumentDeltas = Self.streamingToolArgumentDeltas(from: res.body.string)
            XCTAssertEqual(argumentDeltas, [#"{"location":"Sydney""#, #","unit":"celsius""#, "}"])
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerSuppressesRestartedArgumentObject() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: "call_todos",
                        type: "function",
                        function: StreamDeltaFunction(name: "create_todos", arguments: nil)
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(
                            name: nil,
                            arguments: #"{"todos":["Walk dog","Read book","Cook dinner"]"#
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(
                            name: nil,
                            arguments: #"{"todos":"[\"Walk dog\", \"Read book\", \"Cook dinner\"]""#
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: nil,
                        type: nil,
                        function: StreamDeltaFunction(name: nil, arguments: "}")
                    )
                ]),
                AFMServerStreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 0, promptTime: 0.03, generateTime: 0.02),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true, toolsJSON: Self.todoToolsJSON)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            let argumentDeltas = Self.streamingToolArgumentDeltas(from: res.body.string)
            XCTAssertEqual(argumentDeltas, [#"{"todos":["Walk dog","Read book","Cook dinner"]"#, "}"])
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerFiltersToolCallsToNamedFunctionChoice() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", toolCallDeltas: [
                    StreamDeltaToolCall(
                        index: 0,
                        id: "call_weather",
                        type: "function",
                        function: StreamDeltaFunction(
                            name: "get_weather",
                            arguments: "{\"location\":\"Berlin\"}"
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", toolCalls: [
                    ResponseToolCall(
                        index: 0,
                        id: "call_weather",
                        type: "function",
                        function: ResponseToolCallFunction(
                            name: "get_weather",
                            arguments: #"{"location":"Berlin"}"#
                        )
                    )
                ]),
                AFMServerStreamChunk(text: "", promptTokens: 12, completionTokens: 4, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: true,
            toolsJSON: Self.dualToolsJSON,
            toolChoiceJSON: #"{"type":"function","function":{"name":"read_file"}}"#
        )
        let headers = requestHeaders(for: body)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertFalse(res.body.string.contains("\"get_weather\""))
            XCTAssertFalse(res.body.string.contains("\"tool_calls\""))
            XCTAssertContains(res.body.string, "\"finish_reason\":\"stop\"")
        }
    }

    func testNonStreamingControllerFiltersToolCallsToNamedFunctionChoice() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            generateResult: (
                modelID: "test-model",
                content: "No matching tool call",
                promptTokens: 10,
                completionTokens: 4,
                tokenLogprobs: nil,
                toolCalls: [
                    ResponseToolCall(
                        index: 0,
                        id: "call_weather",
                        type: "function",
                        function: ResponseToolCallFunction(
                            name: "get_weather",
                            arguments: #"{"location":"Berlin"}"#
                        )
                    )
                ],
                cachedTokens: 0,
                promptTime: 0.02,
                generateTime: 0.01,
                stoppedBySequence: false
            ),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: false,
            toolsJSON: Self.dualToolsJSON,
            toolChoiceJSON: #"{"type":"function","function":{"name":"read_file"}}"#
        )
        let headers = requestHeaders(for: body)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertFalse(res.body.string.contains("\"tool_calls\""))
            XCTAssertFalse(res.body.string.contains("\"get_weather\""))
            XCTAssertContains(res.body.string, "\"content\":\"No matching tool call\"")
            XCTAssertContains(res.body.string, "\"finish_reason\":\"stop\"")
        }
    }

    func testStreamingControllerNarrowsToolsToNamedFunctionChoiceBeforeGeneration() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", promptTokens: 12, completionTokens: 0, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: true,
            toolsJSON: Self.dualToolsJSON,
            toolChoiceJSON: #"{"type":"function","function":{"name":"read_file"}}"#
        )
        let headers = requestHeaders(for: body)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
        }

        XCTAssertEqual(service.recordedStreamingToolNames.first, ["read_file"])
        XCTAssertEqual(service.recordedStreamingToolChoices.first, "function:read_file")
    }

    func testNonStreamingControllerNarrowsToolsToNamedFunctionChoiceBeforeGeneration() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            generateResult: (
                modelID: "test-model",
                content: "ok",
                promptTokens: 10,
                completionTokens: 1,
                tokenLogprobs: nil,
                toolCalls: nil,
                cachedTokens: 0,
                promptTime: 0.02,
                generateTime: 0.01,
                stoppedBySequence: false
            ),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: false,
            toolsJSON: Self.dualToolsJSON,
            toolChoiceJSON: #"{"type":"function","function":{"name":"read_file"}}"#
        )
        let headers = requestHeaders(for: body)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
        }

        XCTAssertEqual(service.recordedGenerateToolNames.first, ["read_file"])
        XCTAssertEqual(service.recordedGenerateToolChoices.first, "function:read_file")
    }

    func testNamedFunctionChoiceMissingFromToolsReturnsBadRequest() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: false,
            toolsJSON: Self.weatherToolsJSON,
            toolChoiceJSON: #"{"type":"function","function":{"name":"read_file"}}"#
        )
        let headers = requestHeaders(for: body)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "tool_choice specifies function")
            XCTAssertContains(res.body.string, "\"type\":\"invalid_request_error\"")
        }
    }

    func testConcurrentStreamingRequestsKeepToolCallStateIsolated() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingHandler: { messages in
                let prompt = messages.last?.textContent ?? ""
                if prompt.contains("weather") {
                    return Self.makeDelayedStreamingResult(
                        modelID: "test-model",
                        chunks: [
                            AFMServerStreamChunk(text: "", toolCallDeltas: [
                                StreamDeltaToolCall(
                                    index: 0,
                                    id: "call_weather_berlin",
                                    type: "function",
                                    function: StreamDeltaFunction(
                                        name: "get_weather",
                                        arguments: "{\"location\":\"Berlin\"}"
                                    )
                                )
                            ]),
                            AFMServerStreamChunk(text: "", toolCalls: [
                                ResponseToolCall(
                                    index: 0,
                                    id: "call_weather_berlin",
                                    type: "function",
                                    function: ResponseToolCallFunction(
                                        name: "get_weather",
                                        arguments: #"{"location":"Berlin"}"#
                                    )
                                )
                            ]),
                            AFMServerStreamChunk(text: "", promptTokens: 12, completionTokens: 4, cachedTokens: 0, promptTime: 0.02, generateTime: 0.02),
                        ],
                        delayNanoseconds: 5_000_000
                    )
                }

                return Self.makeDelayedStreamingResult(
                    modelID: "test-model",
                    chunks: [
                        AFMServerStreamChunk(text: "", toolCallDeltas: [
                            StreamDeltaToolCall(
                                index: 0,
                                id: "call_batch_readme",
                                type: "function",
                                function: StreamDeltaFunction(
                                    name: "read_file",
                                    arguments: "{\"path\":\"README.md\"}"
                                )
                            )
                        ]),
                        AFMServerStreamChunk(text: "", toolCalls: [
                            ResponseToolCall(
                                index: 0,
                                id: "call_batch_readme",
                                type: "function",
                                function: ResponseToolCallFunction(
                                    name: "read_file",
                                    arguments: #"{"path":"README.md"}"#
                                )
                            )
                        ]),
                        AFMServerStreamChunk(text: "", promptTokens: 18, completionTokens: 5, cachedTokens: 2, promptTime: 0.03, generateTime: 0.02),
                    ],
                    delayNanoseconds: 5_000_000
                )
            }
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let tester = try app.testable()
        let weatherBody = try requestBody(
            prompt: "What is the weather in Berlin?",
            toolsJSON: Self.weatherToolsJSON
        )
        let readmeBody = try requestBody(
            prompt: "Read the README file.",
            toolsJSON: Self.readFileToolsJSON
        )

        // Compute headers up front so the `async let` child tasks don't capture
        // `self` (XCTestCase is non-Sendable) via requestHeaders(for:).
        let weatherHeaders = requestHeaders(for: weatherBody)
        let readmeHeaders = requestHeaders(for: readmeBody)
        async let weatherResponse: XCTHTTPResponse = tester.sendRequest(
            .POST,
            "/v1/chat/completions",
            headers: weatherHeaders,
            body: weatherBody
        )

        async let readmeResponse: XCTHTTPResponse = tester.sendRequest(
            .POST,
            "/v1/chat/completions",
            headers: readmeHeaders,
            body: readmeBody
        )

        let weather = try await weatherResponse
        let readme = try await readmeResponse
        XCTAssertEqual(weather.status, .ok)
        XCTAssertEqual(readme.status, .ok)

        XCTAssertContains(weather.body.string, "\"get_weather\"")
        XCTAssertContains(weather.body.string, "\\\"location\\\":\\\"Berlin\\\"")
        XCTAssertFalse(weather.body.string.contains("\"read_file\""))

        XCTAssertContains(readme.body.string, "\"read_file\"")
        XCTAssertContains(readme.body.string, "\\\"path\\\":\\\"README.md\\\"")
        XCTAssertFalse(readme.body.string.contains("\"get_weather\""))
    }

    func testNonStreamingStructuredOutputStripsMarkdownFences() async throws {
        let service = FakeMLXChatService(
            generateResult: (
                modelID: "test-model",
                content: "```json\n{\"ok\":true}\n```",
                promptTokens: 8,
                completionTokens: 4,
                tokenLogprobs: nil,
                toolCalls: nil,
                cachedTokens: 0,
                promptTime: 0.01,
                generateTime: 0.01,
                stoppedBySequence: false
            ),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: false,
            toolsJSON: "[]",
            responseFormatJSON: #"{"type":"json_object"}"#
        )

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            guard let response = try? JSONDecoder().decode(ChatCompletionResponse.self, from: Data(res.body.string.utf8)) else {
                XCTFail("Expected decodable ChatCompletionResponse: \(res.body.string)")
                return
            }
            XCTAssertEqual(response.choices.first?.message.content, #"{"ok":true}"#)
            XCTAssertFalse(res.body.string.contains("```"))
        }
    }

    func testStreamingStructuredOutputStripsMarkdownFences() async throws {
        let service = FakeMLXChatService(
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "```json\n"),
                AFMServerStreamChunk(text: "{\"ok\":true}\n"),
                AFMServerStreamChunk(text: "```"),
                AFMServerStreamChunk(text: "", promptTokens: 8, completionTokens: 4, cachedTokens: 0, promptTime: 0.01, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: true,
            toolsJSON: "[]",
            responseFormatJSON: #"{"type":"json_object"}"#
        )

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertFalse(res.body.string.contains("```"))
            let payloads = res.body.string
                .split(separator: "\n")
                .compactMap { line -> [String: Any]? in
                    guard line.hasPrefix("data: "),
                          line != "data: [DONE]" else { return nil }
                    let json = String(line.dropFirst(6))
                    let data = Data(json.utf8)
                    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                }

            let contentValues = payloads.compactMap { payload -> String? in
                let choices = payload["choices"] as? [[String: Any]]
                let delta = choices?.first?["delta"] as? [String: Any]
                return delta?["content"] as? String
            }
            XCTAssertTrue(contentValues.contains(#"{"ok":true}"#), res.body.string)
        }
    }

    func testStrictToolGrammarHeaderSkippedWhenFormatUnsupported() async throws {
        let service = FakeMLXChatService(
            supportsStrictToolGrammar: false,
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "", promptTokens: 2, completionTokens: 1, cachedTokens: 0, promptTime: 0.01, generateTime: 0.01),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(
            stream: false,
            toolsJSON: """
            [
              {
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "strict": true,
                  "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
              }
            ]
            """
        )

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertNil(res.headers.first(name: "X-Grammar-Constraints"))
        }
    }

    func testDeclaredMediaWithoutMediaCapabilityReturnsTypedError() async throws {
        let service = FakeMLXChatService(
            mediaValidationError: .unsupportedMediaInput(model: "test-model", kind: "image"),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let body = mediaRequestBody(stream: false, imageURLJSON: #"{"url":"data:image/png;base64,AA=="}"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, #""code":"unsupported_media_input""#)
            XCTAssertContains(res.body.string, #""type":"invalid_request_error""#)
            XCTAssertEqual(service.mediaPreflightCount, 0)
        }
    }

    func testMalformedDeclaredMediaReturnsTypedErrorBeforePreflight() async throws {
        let service = FakeMLXChatService(streamingResult: makeStreamingResult(chunks: []))
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let body = mediaRequestBody(stream: false, imageURLJSON: "null")
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, #""code":"invalid_media_input""#)
            XCTAssertEqual(service.mediaPreflightCount, 0)
        }
    }

    func testMediaRequestPreflightsExactlyOnceAndUsesResolvedMessages() async throws {
        let service = FakeMLXChatService(streamingResult: makeStreamingResult(chunks: []))
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ4QAAAABJRU5ErkJggg=="
        let body = mediaRequestBody(
            stream: false,
            imageURLJSON: #"{"url":"data:image/png;base64,\#(png)"}"#
        )
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(service.mediaPreflightCount, 1)
            XCTAssertEqual(service.withPreflightedMediaCount, 1)
            XCTAssertEqual(service.recordedGenerateMediaPartCounts, [1])
        }
    }

    func testMediaPreflightFailureReleasesReservedSlot() async throws {
        let releases = OSAllocatedUnfairLock(initialState: 0)
        let admitter = AnyAFMGenerationAdmitter { _ in
            AFMGenerationLease(
                telemetryToken: AFMInferenceRequestToken(rawValue: UUID()),
                release: { releases.withLock { $0 += 1 } }
            )
        }
        let service = FakeMLXChatService(
            maxConcurrent: 2,
            mediaPreflightError: .invalidMediaInput("preflight rejected"),
            providerGenerationAdmitter: admitter,
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ4QAAAABJRU5ErkJggg=="
        let body = mediaRequestBody(
            stream: false,
            imageURLJSON: #"{"url":"data:image/png;base64,\#(png)"}"#
        )
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, #""code":"invalid_media_input""#)
            XCTAssertEqual(service.mediaPreflightCount, 1)
            XCTAssertEqual(releases.withLock { $0 }, 1)
            XCTAssertEqual(service.releaseSlotCount, 0)
        }
    }

    func testCancellationDuringMediaPreflightReleasesSlotAndRegistry() async throws {
        let probe = CancellationProbe()
        let service = FakeMLXChatService(
            mediaPreflightProbe: probe,
            streamingResult: makeStreamingResult(chunks: [])
        )
        app.middleware.use(RequestIDMiddleware())
        try CancelController().boot(routes: app)
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let requestID = "req_cancel_preflight"
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ4QAAAABJRU5ErkJggg=="
        let body = mediaRequestBody(
            stream: false,
            imageURLJSON: #"{"url":"data:image/png;base64,\#(png)"}"#
        )
        let headers = requestHeaders(for: body, requestID: requestID)
        let tester = try app.testable()
        async let completion = tester.sendRequest(
            .POST, "/v1/chat/completions", headers: headers, body: body
        )

        let registry = app.inflightRegistry
        let didStart = await probe.waitUntilStarted()
        XCTAssertTrue(didStart)
        let registeredCount = await registry.count
        XCTAssertEqual(registeredCount, 1)
        let didCancel = await registry.cancel(id: requestID)
        XCTAssertTrue(didCancel)
        _ = try? await completion

        let providerObservedCancellation = await probe.waitUntilCancelled()
        XCTAssertTrue(providerObservedCancellation)
        let released = await waitUntil { service.releaseSlotCount == 1 }
        XCTAssertTrue(released)
        XCTAssertEqual(service.releaseSlotCount, 1)
        let registryCleaned = await waitUntil { await registry.count == 0 }
        XCTAssertTrue(registryCleaned)
    }

    func testCancellationDuringSerialGenerationReleasesSlotAndRegistry() async throws {
        let probe = CancellationProbe()
        let service = FakeMLXChatService(
            maxConcurrent: 1,
            generateProbe: probe,
            streamingResult: makeStreamingResult(chunks: [])
        )
        app.middleware.use(RequestIDMiddleware())
        try CancelController().boot(routes: app)
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let requestID = "req_cancel_serial"
        let body = try requestBody(stream: false)
        let headers = requestHeaders(for: body, requestID: requestID)
        let tester = try app.testable()
        async let completion = tester.sendRequest(
            .POST, "/v1/chat/completions", headers: headers, body: body
        )

        let registry = app.inflightRegistry
        let didStart = await probe.waitUntilStarted()
        XCTAssertTrue(didStart)
        let registeredCount = await registry.count
        XCTAssertEqual(registeredCount, 1)
        let didCancel = await registry.cancel(id: requestID)
        XCTAssertTrue(didCancel)
        _ = try? await completion

        let providerObservedCancellation = await probe.waitUntilCancelled()
        XCTAssertTrue(providerObservedCancellation)
        let released = await waitUntil { service.releaseSlotCount == 1 }
        XCTAssertTrue(released)
        XCTAssertEqual(service.releaseSlotCount, 1)
        let registryCleaned = await waitUntil { await registry.count == 0 }
        XCTAssertTrue(registryCleaned)
    }

    func testCancellationDuringConcurrentCollectionReleasesSlotAndRegistry() async throws {
        let probe = CancellationProbe()
        let releases = OSAllocatedUnfairLock(initialState: 0)
        let admitter = AnyAFMGenerationAdmitter { _ in
            AFMGenerationLease(
                telemetryToken: AFMInferenceRequestToken(rawValue: UUID()),
                release: { releases.withLock { $0 += 1 } }
            )
        }
        let service = FakeMLXChatService(
            maxConcurrent: 2,
            streamCollectionProbe: probe,
            providerGenerationAdmitter: admitter,
            streamingResult: makeStreamingResult(chunks: [])
        )
        app.middleware.use(RequestIDMiddleware())
        try CancelController().boot(routes: app)
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let requestID = "req_cancel_concurrent"
        let body = try requestBody(stream: false)
        let headers = requestHeaders(for: body, requestID: requestID)
        let tester = try app.testable()
        async let completion = tester.sendRequest(
            .POST, "/v1/chat/completions", headers: headers, body: body
        )

        let registry = app.inflightRegistry
        let didStart = await probe.waitUntilStarted()
        XCTAssertTrue(didStart)
        let registeredCount = await registry.count
        XCTAssertEqual(registeredCount, 1)
        let didCancel = await registry.cancel(id: requestID)
        XCTAssertTrue(didCancel)
        _ = try? await completion

        let providerObservedCancellation = await probe.waitUntilCancelled()
        XCTAssertTrue(providerObservedCancellation)
        let released = await waitUntil { releases.withLock { $0 } == 1 }
        XCTAssertTrue(released)
        XCTAssertEqual(releases.withLock { $0 }, 1)
        let registryCleaned = await waitUntil { await registry.count == 0 }
        XCTAssertTrue(registryCleaned)
    }

    func testStreamingMediaPreflightFailureOccursBeforeSSEHeaders() async throws {
        let service = FakeMLXChatService(
            mediaPreflightError: .invalidMediaInput("preflight rejected"),
            streamingResult: makeStreamingResult(chunks: [])
        )
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ4QAAAABJRU5ErkJggg=="
        let body = mediaRequestBody(
            stream: true,
            imageURLJSON: #"{"url":"data:image/png;base64,\#(png)"}"#
        )
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertEqual(res.headers.contentType, .json)
            XCTAssertFalse(res.body.string.contains("data: [DONE]"))
            XCTAssertEqual(service.releaseSlotCount, 1)
        }
    }

    func testStreamingMediaPreflightsOnceAndHandsOffResolvedMessages() async throws {
        let service = FakeMLXChatService(
            streamingHandler: { messages in
                XCTAssertEqual(
                    AFMMLXMediaSecurityPolicy.declaredMediaKinds(in: messages).count,
                    1
                )
                return self.makeStreamingResult(chunks: [
                    AFMServerStreamChunk(
                        text: "seen",
                        promptTokens: 4,
                        completionTokens: 1,
                        promptTime: 0.01,
                        generateTime: 0.01
                    )
                ])
            }
        )
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ4QAAAABJRU5ErkJggg=="
        let body = mediaRequestBody(
            stream: true,
            imageURLJSON: #"{"url":"data:image/png;base64,\#(png)"}"#
        )
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.headers.contentType, .init(type: "text", subType: "event-stream"))
            XCTAssertContains(res.body.string, #""content":"seen""#)
            XCTAssertEqual(service.mediaPreflightCount, 1)
            XCTAssertEqual(service.withPreflightedMediaCount, 1)
        }
    }

    func testLateStreamingFailureUsesOpenAIErrorEventWithoutAssistantWarning() async throws {
        let stream = AsyncThrowingStream<AFMServerStreamChunk, Error> { continuation in
            continuation.yield(AFMServerStreamChunk(text: "partial"))
            continuation.finish(throwing: MLXServiceError.invalidMediaInput("late media failure"))
        }
        let service = FakeMLXChatService(streamingResult: (
            modelID: "test-model",
            stream: stream,
            promptTokens: 2,
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            thinkStartTag: nil,
            thinkEndTag: nil
        ))
        try MLXChatCompletionsController(
            modelID: "test-model", service: service, temperature: nil, repetitionPenalty: nil
        ).boot(routes: app)

        let body = try requestBody(stream: true)
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/chat/completions", headers: requestHeaders(for: body), body: body
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""code":"invalid_media_input""#)
            XCTAssertContains(res.body.string, "data: [DONE]")
            XCTAssertFalse(res.body.string.contains("⚠️"))
        }
    }

    func testProviderOwnedLeaseIsReleasedAfterConcurrentNonStreamingCollection() async throws {
        let releases = OSAllocatedUnfairLock(initialState: 0)
        let admitter = AnyAFMGenerationAdmitter { _ in
            AFMGenerationLease(
                telemetryToken: AFMInferenceRequestToken(rawValue: UUID()),
                release: { releases.withLock { $0 += 1 } }
            )
        }
        let service = FakeMLXChatService(
            maxConcurrent: 2,
            providerGenerationAdmitter: admitter,
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(
                    text: "ok",
                    promptTokens: 2,
                    completionTokens: 1
                ),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        for _ in 0..<2 {
            let body = ByteBuffer(string: #"{"model":"test-model","stream":false,"ignore_eos":true,"messages":[{"role":"user","content":"hello"}]}"#)
            try await app.testable().test(
                .POST,
                "/v1/chat/completions",
                headers: requestHeaders(for: body),
                body: body
            ) { response async in
                XCTAssertEqual(response.status, .ok)
            }
        }
        XCTAssertEqual(releases.withLock { $0 }, 2)
        XCTAssertEqual(service.recordedIgnoreEndOfSequence, [true, true])
    }

    func testProviderOwnedLeaseIsReleasedAfterStreamingBodyTerminates() async throws {
        let releases = OSAllocatedUnfairLock(initialState: 0)
        let admitter = AnyAFMGenerationAdmitter { _ in
            AFMGenerationLease(
                telemetryToken: AFMInferenceRequestToken(rawValue: UUID()),
                release: { releases.withLock { $0 += 1 } }
            )
        }
        let service = FakeMLXChatService(
            maxConcurrent: 2,
            providerGenerationAdmitter: admitter,
            streamingResult: makeStreamingResult(chunks: [
                AFMServerStreamChunk(text: "ok", promptTokens: 2, completionTokens: 1),
            ])
        )
        try MLXChatCompletionsController(
            modelID: "test-model",
            service: service,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)

        let body = ByteBuffer(string: #"{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hello"}]}"#)
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: body),
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertTrue(response.body.string.contains("data: [DONE]"))
        }
        XCTAssertEqual(releases.withLock { $0 }, 1)
    }

    private func requestBody(
        stream: Bool = true,
        prompt: String = "What is the weather in Berlin?",
        toolsJSON: String = weatherToolsJSON,
        toolChoiceJSON: String? = nil,
        responseFormatJSON: String? = nil,
        stopJSON: String? = nil
    ) throws -> ByteBuffer {
        let toolChoiceLine = toolChoiceJSON.map { "\n          \"tool_choice\": \($0)," } ?? ""
        let responseFormatLine = responseFormatJSON.map { "\n          \"response_format\": \($0)," } ?? ""
        let stopLine = stopJSON.map { "\n          \"stop\": \($0)," } ?? ""
        let json = """
        {
          "model": "test-model",
          "stream": \(stream ? "true" : "false"),
          "messages": [
            { "role": "user", "content": "\(prompt)" }
          ],\(toolChoiceLine)\(responseFormatLine)\(stopLine)
          "tools": \(toolsJSON)
        }
        """
        var buffer = ByteBufferAllocator().buffer(capacity: json.utf8.count)
        buffer.writeString(json)
        return buffer
    }

    private func requestHeaders(for body: ByteBuffer, requestID: String? = nil) -> HTTPHeaders {
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)
        if let requestID {
            headers.replaceOrAdd(name: "X-Request-ID", value: requestID)
        }
        return headers
    }

    private func waitUntil(
        timeoutIterations: Int = 200,
        condition: @escaping @Sendable () async -> Bool
    ) async -> Bool {
        for _ in 0..<timeoutIterations {
            if await condition() { return true }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return false
    }

    private func mediaRequestBody(stream: Bool, imageURLJSON: String) -> ByteBuffer {
        let json = """
        {
          "model": "test-model",
          "stream": \(stream ? "true" : "false"),
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "Describe this image." },
                { "type": "image_url", "image_url": \(imageURLJSON) }
              ]
            }
          ]
        }
        """
        var buffer = ByteBufferAllocator().buffer(capacity: json.utf8.count)
        buffer.writeString(json)
        return buffer
    }

    private func makeStreamingResult(chunks: [AFMServerStreamChunk]) -> AFMChatStreamingResult {
        Self.makeDelayedStreamingResult(modelID: "test-model", chunks: chunks, delayNanoseconds: nil)
    }

    private static func streamingToolArgumentDeltas(from body: String) -> [String] {
        body
            .split(separator: "\n")
            .compactMap { line -> String? in
                guard line.hasPrefix("data: "),
                      line != "data: [DONE]" else { return nil }
                let json = String(line.dropFirst(6))
                guard let payload = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any],
                      let choices = payload["choices"] as? [[String: Any]],
                      let delta = choices.first?["delta"] as? [String: Any],
                      let toolCalls = delta["tool_calls"] as? [[String: Any]],
                      let function = toolCalls.first?["function"] as? [String: Any] else {
                    return nil
                }
                return function["arguments"] as? String
            }
    }

    private func makeStreamingResult(
        chunks: [AFMServerStreamChunk],
        toolCallStartTag: String?,
        toolCallEndTag: String?
    ) -> AFMChatStreamingResult {
        Self.makeDelayedStreamingResult(
            modelID: "test-model",
            chunks: chunks,
            delayNanoseconds: nil,
            toolCallStartTag: toolCallStartTag,
            toolCallEndTag: toolCallEndTag
        )
    }

    private static func makeDelayedStreamingResult(modelID: String, chunks: [AFMServerStreamChunk], delayNanoseconds: UInt64?) -> AFMChatStreamingResult {
        Self.makeDelayedStreamingResult(
            modelID: modelID,
            chunks: chunks,
            delayNanoseconds: delayNanoseconds,
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>"
        )
    }

    private static func makeDelayedStreamingResult(
        modelID: String,
        chunks: [AFMServerStreamChunk],
        delayNanoseconds: UInt64?,
        toolCallStartTag: String?,
        toolCallEndTag: String?
    ) -> AFMChatStreamingResult {
        let stream = AsyncThrowingStream<AFMServerStreamChunk, Error> { continuation in
            Task {
                for chunk in chunks {
                    continuation.yield(chunk)
                    if let delayNanoseconds {
                        try? await Task.sleep(nanoseconds: delayNanoseconds)
                    }
                }
                continuation.finish()
            }
        }
        return (
            modelID: modelID,
            stream: stream,
            promptTokens: 8,
            toolCallStartTag: toolCallStartTag,
            toolCallEndTag: toolCallEndTag,
            thinkStartTag: nil,
            thinkEndTag: nil
        )
    }

    private static let weatherToolsJSON = """
    [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": { "type": "string" }
            },
            "required": ["location"]
          }
        }
      }
    ]
    """

    private static let readFileToolsJSON = """
    [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": { "type": "string" }
            },
            "required": ["path"]
          }
        }
      }
    ]
    """

    private static let todoToolsJSON = """
    [
      {
        "type": "function",
        "function": {
          "name": "create_todos",
          "description": "Create a todo list",
          "parameters": {
            "type": "object",
            "properties": {
              "todos": {
                "type": "array",
                "items": { "type": "string" }
              }
            },
            "required": ["todos"]
          }
        }
      }
    ]
    """

    private static let dualToolsJSON = """
    [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": { "type": "string" }
            },
            "required": ["location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": { "type": "string" }
            },
            "required": ["path"]
          }
        }
      }
    ]
    """
}

private final class FakeMLXChatService: AFMChatServing, AFMMLXMediaRequestServing,
    AFMGenerationAdmitterProviding, @unchecked Sendable
{
    let maxConcurrent: Int
    var generatedStreamOwnsSlotReservation: Bool { maxConcurrent >= 2 }
    let toolCallParser: String?
    let supportsStrictToolGrammar: Bool
    let thinkStartTag: String?
    let thinkEndTag: String?
    let responseChannelFormat: AFMResponseChannelFormat
    let fixToolArgs: Bool
    let enableGrammarConstraints: Bool = false
    let providerGenerationAdmitter: AnyAFMGenerationAdmitter?
    var servingConfiguration: AFMChatServingConfiguration {
        AFMChatServingConfiguration(
            toolCallParser: toolCallParser,
            supportsStrictToolGrammar: supportsStrictToolGrammar,
            thinkStartTag: thinkStartTag,
            thinkEndTag: thinkEndTag,
            responseChannelFormat: responseChannelFormat,
            fixToolArguments: fixToolArgs,
            grammarConstraintsEnabled: enableGrammarConstraints
        )
    }
    private let generateResult: AFMChatGenerationResult
    private let streamingResult: AFMChatStreamingResult
    private let streamingHandler: (([Message]) -> AFMChatStreamingResult)?
    private let mediaValidationError: MLXServiceError?
    private let mediaPreflightError: MLXServiceError?
    private let mediaPreflightProbe: CancellationProbe?
    private let generateProbe: CancellationProbe?
    private let streamCollectionProbe: CancellationProbe?
    private let stateLock = NSLock()
    private(set) var recordedGenerateToolNames: [[String]] = []
    private(set) var recordedStreamingToolNames: [[String]] = []
    private(set) var recordedGenerateToolChoices: [String] = []
    private(set) var recordedStreamingToolChoices: [String] = []
    private(set) var recordedPreserveStructuralTags: [Bool] = []
    private var _mediaPreflightCount = 0
    private var _withPreflightedMediaCount = 0
    private var _releaseSlotCount = 0
    private var _generateCount = 0
    private var _recordedIgnoreEndOfSequence: [Bool] = []
    private(set) var recordedGenerateMediaPartCounts: [Int] = []
    var mediaPreflightCount: Int { stateLock.withLock { _mediaPreflightCount } }
    var withPreflightedMediaCount: Int { stateLock.withLock { _withPreflightedMediaCount } }
    var releaseSlotCount: Int { stateLock.withLock { _releaseSlotCount } }
    var generateCount: Int { stateLock.withLock { _generateCount } }
    var recordedIgnoreEndOfSequence: [Bool] { stateLock.withLock { _recordedIgnoreEndOfSequence } }

    init(
        maxConcurrent: Int = 1,
        toolCallParser: String? = nil,
        supportsStrictToolGrammar: Bool = false,
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        responseChannelFormat: AFMResponseChannelFormat = .none,
        fixToolArgs: Bool = false,
        mediaValidationError: MLXServiceError? = nil,
        mediaPreflightError: MLXServiceError? = nil,
        mediaPreflightProbe: CancellationProbe? = nil,
        generateProbe: CancellationProbe? = nil,
        streamCollectionProbe: CancellationProbe? = nil,
        providerGenerationAdmitter: AnyAFMGenerationAdmitter? = nil,
        generateResult: AFMChatGenerationResult? = nil,
        streamingResult: AFMChatStreamingResult
    ) {
        self.maxConcurrent = maxConcurrent
        self.toolCallParser = toolCallParser
        self.supportsStrictToolGrammar = supportsStrictToolGrammar
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.responseChannelFormat = responseChannelFormat
        self.fixToolArgs = fixToolArgs
        self.mediaValidationError = mediaValidationError
        self.mediaPreflightError = mediaPreflightError
        self.mediaPreflightProbe = mediaPreflightProbe
        self.generateProbe = generateProbe
        self.streamCollectionProbe = streamCollectionProbe
        self.providerGenerationAdmitter = providerGenerationAdmitter
        self.generateResult = generateResult ?? (
            modelID: "test-model",
            content: "",
            promptTokens: 0,
            completionTokens: 0,
            tokenLogprobs: nil,
            toolCalls: nil,
            cachedTokens: 0,
            promptTime: 0,
            generateTime: 0,
            stoppedBySequence: false
        )
        self.streamingResult = streamingResult
        self.streamingHandler = nil
    }

    init(
        maxConcurrent: Int = 1,
        toolCallParser: String? = nil,
        supportsStrictToolGrammar: Bool = false,
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        responseChannelFormat: AFMResponseChannelFormat = .none,
        fixToolArgs: Bool = false,
        mediaValidationError: MLXServiceError? = nil,
        mediaPreflightError: MLXServiceError? = nil,
        mediaPreflightProbe: CancellationProbe? = nil,
        generateProbe: CancellationProbe? = nil,
        streamCollectionProbe: CancellationProbe? = nil,
        providerGenerationAdmitter: AnyAFMGenerationAdmitter? = nil,
        streamingHandler: @escaping ([Message]) -> AFMChatStreamingResult
    ) {
        self.maxConcurrent = maxConcurrent
        self.toolCallParser = toolCallParser
        self.supportsStrictToolGrammar = supportsStrictToolGrammar
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.responseChannelFormat = responseChannelFormat
        self.fixToolArgs = fixToolArgs
        self.mediaValidationError = mediaValidationError
        self.mediaPreflightError = mediaPreflightError
        self.mediaPreflightProbe = mediaPreflightProbe
        self.generateProbe = generateProbe
        self.streamCollectionProbe = streamCollectionProbe
        self.providerGenerationAdmitter = providerGenerationAdmitter
        self.generateResult = (
            modelID: "test-model",
            content: "",
            promptTokens: 0,
            completionTokens: 0,
            tokenLogprobs: nil,
            toolCalls: nil,
            cachedTokens: 0,
            promptTime: 0,
            generateTime: 0,
            stoppedBySequence: false
        )
        self.streamingResult = FakeMLXChatService.emptyStreamingResult
        self.streamingHandler = streamingHandler
    }

    func normalizeModel(_ raw: String) -> String { raw }
    func resolvedToolCallParser(logBypass: Bool) -> String? { toolCallParser }
    func tryReserveSlot() -> Bool { true }
    func releaseSlot() { stateLock.withLock { _releaseSlotCount += 1 } }
    func ensureBatchMode(concurrency: Int) async throws {}
    func releaseBatchReference() {}
    func cancelBatchSlots(ids: Set<UUID>) async {}
    func startAPIProfile() {}
    func stopAPIProfile(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfile {
        AFMProfile(gpuPowerAvgW: nil, gpuPowerPeakW: nil, gpuSamples: nil, memoryWeightsGiB: nil, memoryKvGiB: nil, memoryPeakGiB: nil, prefillTokS: nil, decodeTokS: nil, chip: nil, theoreticalBwGbs: nil, estBandwidthGbs: nil)
    }
    func stopAPIProfileExtended(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfileExtended {
        AFMProfileExtended(summary: stopAPIProfile(promptTokens: promptTokens, completionTokens: completionTokens, promptTime: promptTime, generateTime: generateTime), samples: [])
    }

    func loadedModelDescriptor(model: String) -> AFMModelDescriptor? { nil }

    func validateMediaRequestCapabilities(model: String, messages: [Message]) throws {
        if let mediaValidationError { throw mediaValidationError }
        do {
            try AFMMLXMediaSecurityPolicy.validateReferences(in: messages)
        } catch {
            throw MLXServiceError.invalidMediaInput(error.localizedDescription)
        }
    }

    func preflightMediaRequest(
        model: String,
        messages: [Message]
    ) async throws -> AFMMLXResolvedMediaRequest {
        stateLock.withLock { _mediaPreflightCount += 1 }
        if let mediaPreflightProbe {
            try await mediaPreflightProbe.suspendUntilCancelled()
        }
        if let mediaPreflightError { throw mediaPreflightError }
        do {
            return try await AFMMLXMediaSecurityPolicy.resolveRequest(in: messages)
        } catch {
            throw MLXServiceError.invalidMediaInput(error.localizedDescription)
        }
    }

    func withPreflightedMediaRequest<Result: Sendable>(
        _ request: AFMMLXResolvedMediaRequest,
        operation: ([Message]) async throws -> Result
    ) async throws -> Result {
        stateLock.withLock { _withPreflightedMediaCount += 1 }
        return try await operation(request.messages)
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMChatGenerationResult {
        stateLock.withLock {
            _generateCount += 1
            _recordedIgnoreEndOfSequence.append(AFMGenerationContext.ignoreEndOfSequence)
        }
        recordGenerateTools(tools)
        if let generateProbe {
            try await generateProbe.suspendUntilCancelled()
        }
        stateLock.withLock {
            recordedGenerateMediaPartCounts.append(
                AFMMLXMediaSecurityPolicy.declaredMediaKinds(in: messages).count
            )
        }
        return generateResult
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMChatGenerationResult {
        recordGenerateToolChoice(toolChoice)
        return try await generate(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMChatStreamingResult {
        stateLock.withLock { _recordedIgnoreEndOfSequence.append(AFMGenerationContext.ignoreEndOfSequence) }
        recordStreamingTools(tools)
        if let streamCollectionProbe {
            let stream = AsyncThrowingStream<AFMServerStreamChunk, Error> { continuation in
                let task = Task {
                    do {
                        try await streamCollectionProbe.suspendUntilCancelled()
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { [weak self] _ in
                    task.cancel()
                    self?.releaseSlot()
                }
            }
            return (
                modelID: "test-model",
                stream: stream,
                promptTokens: 0,
                toolCallStartTag: "<tool_call>",
                toolCallEndTag: "</tool_call>",
                thinkStartTag: nil,
                thinkEndTag: nil
            )
        }
        return streamingHandler?(messages) ?? streamingResult
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMChatStreamingResult {
        recordStreamingToolChoice(toolChoice)
        recordPreserveStructuralTags(preserveStructuralTags)
        return try await generateStreaming(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMChatStreamingResult {
        recordPreserveStructuralTags(preserveStructuralTags)
        return try await generateStreaming(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    private func recordGenerateTools(_ tools: [RequestTool]?) {
        stateLock.lock()
        recordedGenerateToolNames.append(tools?.map(\.function.name) ?? [])
        stateLock.unlock()
    }

    private func recordStreamingTools(_ tools: [RequestTool]?) {
        stateLock.lock()
        recordedStreamingToolNames.append(tools?.map(\.function.name) ?? [])
        stateLock.unlock()
    }

    private func recordPreserveStructuralTags(_ preserve: Bool) {
        stateLock.lock()
        recordedPreserveStructuralTags.append(preserve)
        stateLock.unlock()
    }

    private func recordGenerateToolChoice(_ toolChoice: ToolChoice?) {
        stateLock.lock()
        recordedGenerateToolChoices.append(Self.toolChoiceLabel(toolChoice))
        stateLock.unlock()
    }

    private func recordStreamingToolChoice(_ toolChoice: ToolChoice?) {
        stateLock.lock()
        recordedStreamingToolChoices.append(Self.toolChoiceLabel(toolChoice))
        stateLock.unlock()
    }

    private static func toolChoiceLabel(_ toolChoice: ToolChoice?) -> String {
        switch toolChoice {
        case .mode(let mode):
            return "mode:\(mode)"
        case .function(let choice):
            return "function:\(choice.function.name)"
        case nil:
            return "none"
        }
    }

    private static let emptyStreamingResult: AFMChatStreamingResult = (
        modelID: "test-model",
        stream: AsyncThrowingStream { $0.finish() },
        promptTokens: 0,
        toolCallStartTag: "<tool_call>",
        toolCallEndTag: "</tool_call>",
        thinkStartTag: nil,
        thinkEndTag: nil
    )
}

private final class CancellationProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var _started = false
    private var _cancelled = false

    func suspendUntilCancelled() async throws {
        lock.withLock { _started = true }
        do {
            try await Task.sleep(nanoseconds: 60_000_000_000)
        } catch {
            lock.withLock { _cancelled = true }
            throw error
        }
    }

    func waitUntilStarted() async -> Bool {
        await waitUntil { self.lock.withLock { self._started } }
    }

    func waitUntilCancelled() async -> Bool {
        await waitUntil { self.lock.withLock { self._cancelled } }
    }

    private func waitUntil(_ condition: @escaping @Sendable () -> Bool) async -> Bool {
        for _ in 0..<200 {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return false
    }
}
