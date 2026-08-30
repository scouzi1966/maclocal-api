import Foundation
import Testing
import Vapor
import XCTest
import XCTVapor

@testable import AFMKit
@testable import AFMServer

/// Tests for T1.6 (tokenize / count_tokens request decoding) and T1.7
/// (`/openapi.json` integrity).
struct TokenizeAndOpenAPITests {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.6 — request shape decoding
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.6 tokenize request decodes `text` field")
    func decodeTextField() throws {
        let json = #"{"model":"m","text":"hello world"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.model == "m")
        #expect(req.text == "hello world")
        #expect(req.prompt == nil)
        #expect(req.effectiveText == "hello world")
    }

    @Test("T1.6 tokenize request decodes vLLM-style `prompt` alias")
    func decodePromptField() throws {
        let json = #"{"model":"m","prompt":"hi"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.text == nil)
        #expect(req.prompt == "hi")
        #expect(req.effectiveText == "hi")
    }

    @Test("T1.6 effectiveText prefers text over prompt when both supplied")
    func textWinsOverPrompt() throws {
        let json = #"{"text":"a","prompt":"b"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.effectiveText == "a")
    }

    @Test("T1.6 effectiveText is empty when neither field is supplied")
    func effectiveTextEmpty() throws {
        let json = #"{"model":"m"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.effectiveText == "")
    }

    @Test("T1.6 TokenizeResponse round-trips with snake_case max_model_len")
    func tokenizeResponseEncoding() throws {
        let resp = TokenizeResponse(tokens: [1, 2, 3], count: 3, model: "m", maxModelLen: 32768)
        let data = try JSONEncoder().encode(resp)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(json.contains("\"tokens\":[1,2,3]"))
        #expect(json.contains("\"count\":3"))
        #expect(json.contains("\"max_model_len\":32768"))
    }

    @Test("T1.6 CountTokensResponse uses Anthropic input_tokens key")
    func countResponseEncoding() throws {
        let resp = CountTokensResponse(inputTokens: 42, model: "m")
        let data = try JSONEncoder().encode(resp)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(json.contains("\"input_tokens\":42"))
        #expect(json.contains("\"model\":\"m\""))
    }

    @Test("Messages count-tokens accepts Anthropic string and block content")
    func messagesCountRequestDecoding() throws {
        let json = #"""
        {
          "model": "m",
          "system": [{"type":"text","text":"Be concise."}],
          "messages": [
            {"role":"user","content":"Hello"},
            {"role":"assistant","content":[{"type":"thinking","text":"Plan"},{"type":"text","text":"Hi"}]}
          ]
        }
        """#
        let request = try JSONDecoder().decode(MessagesCountTokensRequest.self, from: Data(json.utf8))
        #expect(request.model == "m")
        #expect(request.effectiveText == "Be concise.\nHello\nPlan\nHi")
    }

    @Test("Messages adapter forwards block text and preserves Anthropic SSE lifecycle")
    func messagesAdapterTranslation() throws {
        let request = try MessagesController.makeChatRequest(
            object: [
                "system": .array([.object(["type": .string("text"), "text": .string("Be concise.")])]),
                "stop_sequences": .array([.string("END")]),
                "thinking": .object(["type": .string("enabled"), "budget_tokens": .number(256)])
            ],
            sourceMessages: [.object([
                "role": .string("user"),
                "content": .array([.object(["type": .string("text"), "text": .string("Say hi")])])
            ])],
            maxTokens: 12,
            defaultModel: "test-model"
        )
        #expect(request["messages"]?.arrayValue?.count == 2)
        #expect(request["messages"]?.arrayValue?[1]["content"]?.stringValue == "Say hi")
        #expect(request["stop"]?.arrayValue?.first?.stringValue == "END")
        #expect(request["reasoning_effort"]?.stringValue == "low")

        let message = try MessagesController.makeMessage(
            chat: .object([
                "model": .string("test-model"),
                "_afm_matched_stop": .string("END"),
                "choices": .array([.object(["finish_reason": .string("stop"), "message": .object(["content": .string("hello")])])]),
                "usage": .object(["prompt_tokens": .number(3), "completion_tokens": .number(1)])
            ]),
            request: [:],
            defaultModel: "test-model"
        )
        let types = MessagesController.streamingEvents(for: message).compactMap { $0["type"]?.stringValue }
        #expect(message["stop_reason"]?.stringValue == "stop_sequence")
        #expect(message["stop_sequence"]?.stringValue == "END")
        #expect(types.first == "message_start")
        #expect(types.last == "message_stop")
        #expect(types.contains("content_block_start"))
        #expect(types.contains("content_block_stop"))
    }

    @Test("Messages adapter translates Anthropic tools, tool history, and base64 images")
    func messagesAdapterToolAndVisionTranslation() throws {
        let request = try MessagesController.makeChatRequest(
            object: [
                "tools": .array([.object([
                    "name": .string("weather"), "description": .string("Forecast"),
                    "input_schema": .object(["type": .string("object")])
                ])]),
                "tool_choice": .object(["type": .string("tool"), "name": .string("weather")])
            ],
            sourceMessages: [
                .object(["role": .string("assistant"), "content": .array([.object([
                    "type": .string("tool_use"), "id": .string("call_1"), "name": .string("weather"),
                    "input": .object(["city": .string("Toronto")])
                ])])]),
                .object(["role": .string("user"), "content": .array([
                    .object(["type": .string("tool_result"), "tool_use_id": .string("call_1"), "content": .string("sunny")]),
                    .object(["type": .string("image"), "source": .object([
                        "type": .string("base64"), "media_type": .string("image/png"), "data": .string("AAAA")
                    ])])
                ])])
            ],
            maxTokens: 12,
            defaultModel: "test-model"
        )
        #expect(request["tools"]?.arrayValue?.first?["function"]?["parameters"]?["type"]?.stringValue == "object")
        #expect(request["tool_choice"]?["function"]?["name"]?.stringValue == "weather")
        let messages = request["messages"]?.arrayValue ?? []
        #expect(messages.contains { $0["tool_calls"]?.arrayValue?.first?["id"]?.stringValue == "call_1" })
        #expect(messages.contains { $0["role"]?.stringValue == "tool" && $0["tool_call_id"]?.stringValue == "call_1" })
        #expect(messages.contains { $0["content"]?.arrayValue?.contains { $0["image_url"]?["url"]?.stringValue == "data:image/png;base64,AAAA" } == true })

        let response = try MessagesController.makeMessage(
            chat: .object(["choices": .array([.object([
                "finish_reason": .string("tool_calls"), "message": .object(["tool_calls": .array([.object([
                    "id": .string("call_2"), "function": .object(["name": .string("weather"), "arguments": .string(#"{"city":"Toronto"}"#)])
                ])])])
            ])])]),
            request: [:], defaultModel: "test-model"
        )
        #expect(response["stop_reason"]?.stringValue == "tool_use")
        #expect(response["content"]?.arrayValue?.first?["type"]?.stringValue == "tool_use")
        let events = MessagesController.streamingEvents(for: response)
        #expect(events.contains { $0["content_block"]?["type"]?.stringValue == "tool_use" })
        #expect(events.contains { $0["delta"]?["type"]?.stringValue == "input_json_delta" })
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.7 — OpenAPI spec integrity
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.7 openapi.json is valid JSON and reports OpenAPI 3.1")
    func openAPISpecParses() throws {
        let data = Data(OpenAPIController.specJSON.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(parsed != nil)
        #expect((parsed?["openapi"] as? String) == "3.1.0")
        #expect((parsed?["info"] as? [String: Any])?["title"] != nil)
    }

    @Test("T1.7 openapi.json declares the agent-relevant endpoints")
    func openAPICoversAgentEndpoints() throws {
        let data = Data(OpenAPIController.specJSON.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let paths = parsed?["paths"] as? [String: Any] ?? [:]
        let expected = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/responses/{response_id}",
            "/v1/chat/completions/{id}/cancel",
            "/v1/tokenize",
            "/v1/count_tokens",
            "/v1/messages/count_tokens",
            "/v1/messages",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/audio/speech",
            "/v1/ocr",
            "/v1/batch/completions",
            "/v1/files",
            "/v1/models",
            "/health",
            "/metrics"
        ]
        for path in expected {
            #expect(paths[path] != nil, "missing path in OpenAPI spec: \(path)")
        }
    }

    @Test("T1.7 docs page references /openapi.json on same origin")
    func docsHTMLReferencesSpec() {
        let html = OpenAPIController.docsHTML
        #expect(html.contains("/openapi.json"))
        #expect(html.contains("scalar"))
    }
}

final class TokenizeControllerIntegrationTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testTokenizeUsesPortableAFMKitCapability() async throws {
        try TokenizeController(
            mlxModelID: "test/model",
            tokenizer: FixedTokenizer(tokens: [11, 22, 33]),
            contextWindow: 8_192
        ).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"test/model","text":"hello"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/tokenize",
            headers: headers,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertContains(response.body.string, #""tokens":[11,22,33]"#)
            XCTAssertContains(response.body.string, #""count":3"#)
            XCTAssertContains(response.body.string, #""max_model_len":8192"#)
        }
    }

    func testTokenizeWithoutCapabilityReturnsUnprocessableEntity() async throws {
        try TokenizeController(
            mlxModelID: nil,
            tokenizer: nil,
            contextWindow: nil
        ).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"text":"hello"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/tokenize",
            headers: headers,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .unprocessableEntity)
        }
    }

    func testMessagesCountTokensAcceptsConversationShape() async throws {
        try TokenizeController(
            mlxModelID: "test/model",
            tokenizer: FixedTokenizer(tokens: [1, 2, 3, 4]),
            contextWindow: 8_192
        ).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"""
        {"model":"test/model","messages":[{"role":"user","content":"hello"}]}
        """#)

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/messages/count_tokens",
            headers: headers,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertContains(response.body.string, #""input_tokens":4"#)
        }
    }

    func testMessagesRequiresMaxTokensWithAnthropicErrorEnvelope() async throws {
        try MessagesController(model: "test-model", chatHandler: { _ in
            throw Abort(.notImplemented)
        }).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/messages",
            headers: headers,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            XCTAssertContains(response.body.string, #""type":"error""#)
            XCTAssertContains(response.body.string, #""invalid_request_error""#)
        }
    }
}

private struct FixedTokenizer: AFMTextTokenizing {
    let tokens: [Int]

    func tokenize(text: String) async throws -> [Int] {
        tokens
    }
}
