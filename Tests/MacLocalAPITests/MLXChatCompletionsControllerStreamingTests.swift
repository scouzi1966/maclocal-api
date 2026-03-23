import XCTest
import Vapor
import XCTVapor

@testable import MacLocalAPI

final class MLXChatCompletionsControllerStreamingTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testStreamingControllerParsesRawToolCallTextIntoSSEToolCalls() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                StreamChunk(text: "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Berlin\"}}</tool_call>"),
                StreamChunk(text: "", promptTokens: 14, completionTokens: 3, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
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
                StreamChunk(text: "", toolCalls: [toolCall]),
                StreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 4, promptTime: 0.03, generateTime: 0.02),
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

    func testStreamingControllerSerializesBatchToolCallDeltasBeforeCompletedCall() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                StreamChunk(text: "", toolCallDeltas: [
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
                StreamChunk(text: "", toolCalls: [
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
                StreamChunk(text: "", promptTokens: 20, completionTokens: 5, cachedTokens: 4, promptTime: 0.03, generateTime: 0.02),
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
            XCTAssertContains(res.body.string, "\"finish_reason\":\"tool_calls\"")
        }
    }

    func testStreamingControllerFiltersToolCallsToNamedFunctionChoice() async throws {
        let service = FakeMLXChatService(
            toolCallParser: "afm_adaptive_xml",
            streamingResult: makeStreamingResult(chunks: [
                StreamChunk(text: "", toolCallDeltas: [
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
                StreamChunk(text: "", toolCalls: [
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
                StreamChunk(text: "", promptTokens: 12, completionTokens: 4, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
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
                StreamChunk(text: "", promptTokens: 12, completionTokens: 0, cachedTokens: 0, promptTime: 0.02, generateTime: 0.01),
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
                            StreamChunk(text: "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Berlin\"}}</tool_call>"),
                            StreamChunk(text: "", promptTokens: 12, completionTokens: 4, cachedTokens: 0, promptTime: 0.02, generateTime: 0.02),
                        ],
                        delayNanoseconds: 5_000_000
                    )
                }

                return Self.makeDelayedStreamingResult(
                    modelID: "test-model",
                    chunks: [
                        StreamChunk(text: "", toolCallDeltas: [
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
                        StreamChunk(text: "", toolCalls: [
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
                        StreamChunk(text: "", promptTokens: 18, completionTokens: 5, cachedTokens: 2, promptTime: 0.03, generateTime: 0.02),
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

        async let weatherResponse: XCTHTTPResponse = tester.sendRequest(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: weatherBody),
            body: weatherBody
        )

        async let readmeResponse: XCTHTTPResponse = tester.sendRequest(
            .POST,
            "/v1/chat/completions",
            headers: requestHeaders(for: readmeBody),
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

    private func requestBody(
        stream: Bool = true,
        prompt: String = "What is the weather in Berlin?",
        toolsJSON: String = weatherToolsJSON,
        toolChoiceJSON: String? = nil
    ) throws -> ByteBuffer {
        let toolChoiceLine = toolChoiceJSON.map { "\n          \"tool_choice\": \($0)," } ?? ""
        let json = """
        {
          "model": "test-model",
          "stream": \(stream ? "true" : "false"),
          "messages": [
            { "role": "user", "content": "\(prompt)" }
          ],\(toolChoiceLine)
          "tools": \(toolsJSON)
        }
        """
        var buffer = ByteBufferAllocator().buffer(capacity: json.utf8.count)
        buffer.writeString(json)
        return buffer
    }

    private func requestHeaders(for body: ByteBuffer) -> HTTPHeaders {
        var headers = HTTPHeaders()
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: body.readableBytes.description)
        return headers
    }

    private func makeStreamingResult(chunks: [StreamChunk]) -> ChatStreamingResult {
        Self.makeDelayedStreamingResult(modelID: "test-model", chunks: chunks, delayNanoseconds: nil)
    }

    private static func makeDelayedStreamingResult(modelID: String, chunks: [StreamChunk], delayNanoseconds: UInt64?) -> ChatStreamingResult {
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
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
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
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

private final class FakeMLXChatService: MLXChatServing {
    let maxConcurrent: Int
    let toolCallParser: String?
    let thinkStartTag: String?
    let thinkEndTag: String?
    let fixToolArgs: Bool
    private let generateResult: ChatGenerationResult
    private let streamingResult: ChatStreamingResult
    private let streamingHandler: (([Message]) -> ChatStreamingResult)?
    private let stateLock = NSLock()
    private(set) var recordedGenerateToolNames: [[String]] = []
    private(set) var recordedStreamingToolNames: [[String]] = []

    init(
        maxConcurrent: Int = 1,
        toolCallParser: String? = nil,
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        fixToolArgs: Bool = false,
        generateResult: ChatGenerationResult? = nil,
        streamingResult: ChatStreamingResult
    ) {
        self.maxConcurrent = maxConcurrent
        self.toolCallParser = toolCallParser
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.fixToolArgs = fixToolArgs
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
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        fixToolArgs: Bool = false,
        streamingHandler: @escaping ([Message]) -> ChatStreamingResult
    ) {
        self.maxConcurrent = maxConcurrent
        self.toolCallParser = toolCallParser
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.fixToolArgs = fixToolArgs
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
    func tryReserveSlot() -> Bool { true }
    func releaseSlot() {}

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
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatGenerationResult {
        recordGenerateTools(tools)
        return generateResult
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
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatStreamingResult {
        recordStreamingTools(tools)
        return streamingHandler?(messages) ?? streamingResult
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

    private static let emptyStreamingResult: ChatStreamingResult = (
        modelID: "test-model",
        stream: AsyncThrowingStream { $0.finish() },
        promptTokens: 0,
        toolCallStartTag: "<tool_call>",
        toolCallEndTag: "</tool_call>",
        thinkStartTag: nil,
        thinkEndTag: nil
    )
}
