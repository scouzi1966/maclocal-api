import XCTest
import Vapor
import XCTVapor

@testable import AFMServer

final class ResponsesControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testNonStreamingResponseTranslatesVisionInputAndUsage() async throws {
        let recorder = ResponsesChatRecorder()
        try register(recorder: recorder, content: "I see a red square.")

        let body = #"""
        {
          "model":"test-model",
          "input":[{"role":"user","content":[
            {"type":"input_text","text":"What is shown?"},
            {"type":"input_image","image_url":"data:image/png;base64,ZmFrZQ=="}
          ]}],
          "max_output_tokens":32
        }
        """#

        try await post(body) { response in
            XCTAssertEqual(response.status, .ok)
            let json = try Self.json(response.body.string)
            XCTAssertEqual(json["object"] as? String, "response")
            XCTAssertEqual(json["status"] as? String, "completed")
            XCTAssertEqual(json["model"] as? String, "test-model")
            let output = try XCTUnwrap(json["output"] as? [[String: Any]])
            let content = try XCTUnwrap(output.last?["content"] as? [[String: Any]])
            XCTAssertEqual(content.first?["text"] as? String, "I see a red square.")
            let usage = try XCTUnwrap(json["usage"] as? [String: Any])
            XCTAssertEqual(usage["input_tokens"] as? Int, 7)
            XCTAssertEqual(usage["output_tokens"] as? Int, 5)
        }

        let lastRecorded = await recorder.last()
        let recorded = try XCTUnwrap(lastRecorded)
        let chatBody = recorded.foundationObject
        XCTAssertEqual(chatBody["stream"] as? Bool, false)
        XCTAssertEqual(chatBody["max_tokens"] as? Int, 32)
        let messages = try XCTUnwrap(chatBody["messages"] as? [[String: Any]])
        let parts = try XCTUnwrap(messages.last?["content"] as? [[String: Any]])
        XCTAssertEqual(parts.last?["type"] as? String, "image_url")
        let imageURL = try XCTUnwrap(parts.last?["image_url"] as? [String: Any])
        XCTAssertEqual(imageURL["url"] as? String, "data:image/png;base64,ZmFrZQ==")
    }

    func testStreamingResponseEmitsOrderedResponsesLifecycle() async throws {
        try register(content: "hello")
        let body = #"{"model":"test-model","input":"Say hello.","stream":true}"#

        try await post(body) { response in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(response.headers.contentType?.type, "text")
            XCTAssertEqual(response.headers.contentType?.subType, "event-stream")
            let types = Self.ssePayloads(response.body.string).compactMap { payload -> String? in
                guard payload != "[DONE]",
                      let data = payload.data(using: .utf8),
                      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    return nil
                }
                return object["type"] as? String
            }
            XCTAssertEqual(types.first, "response.created")
            XCTAssertEqual(types.last, "response.completed")
            XCTAssertLessThan(
                try XCTUnwrap(types.firstIndex(of: "response.output_item.added")),
                try XCTUnwrap(types.firstIndex(of: "response.output_item.done"))
            )
            XCTAssertLessThan(
                try XCTUnwrap(types.firstIndex(of: "response.output_text.delta")),
                try XCTUnwrap(types.firstIndex(of: "response.completed"))
            )
            XCTAssertTrue(response.body.string.hasSuffix("data: [DONE]\n\n"))
        }
    }

    func testPreviousResponseIDRestoresPriorConversation() async throws {
        let recorder = ResponsesChatRecorder()
        try register(recorder: recorder, content: "Ada")

        var firstID = ""
        try await post(#"{"input":"My name is Ada. Remember it."}"#) { response in
            let json = try Self.json(response.body.string)
            firstID = try XCTUnwrap(json["id"] as? String)
        }
        try await post(#"{"input":"What is my name?","previous_response_id":"\#(firstID)"}"#) { response in
            XCTAssertEqual(response.status, .ok)
        }

        let requests = (await recorder.all()).map(\.foundationObject)
        XCTAssertEqual(requests.count, 2)
        let messages = try XCTUnwrap(requests.last?["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[0]["content"] as? String, "My name is Ada. Remember it.")
        XCTAssertEqual(messages[1]["role"] as? String, "assistant")
        XCTAssertEqual(messages[1]["content"] as? String, "Ada")
        XCTAssertEqual(messages[2]["content"] as? String, "What is my name?")
    }

    func testBackgroundResponseReturnsInProgressImmediately() async throws {
        try register(content: "hi")
        try await post(#"{"input":"Say hi.","background":true}"#) { response in
            XCTAssertEqual(response.status, .accepted)
            let json = try Self.json(response.body.string)
            XCTAssertEqual(json["status"] as? String, "in_progress")
            XCTAssertEqual(json["background"] as? Bool, true)
            XCTAssertNotNil(json["id"] as? String)
        }
    }

    private func register(
        recorder: ResponsesChatRecorder = ResponsesChatRecorder(),
        content: String
    ) throws {
        try app.register(collection: ResponsesController(defaultModelID: "test-model") { request in
            if let body = request.body.data {
                await recorder.record(Data(buffer: body))
            }
            let escaped = content.replacingOccurrences(of: "\"", with: "\\\"")
            let response = Response(status: .ok)
            response.headers.contentType = .json
            response.body = .init(string: #"{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"\#(escaped)"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}"#)
            return response
        })
    }

    private func post(
        _ body: String,
        afterResponse: @escaping (XCTHTTPResponse) throws -> Void
    ) async throws {
        var headers = HTTPHeaders()
        headers.contentType = .json
        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/responses",
            headers: headers,
            body: ByteBuffer(string: body)
        ) { response async in
            do {
                try afterResponse(response)
            } catch {
                XCTFail("Response assertion failed: \(error)")
            }
        }
    }

    private static func json(_ string: String) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(string.utf8)) as? [String: Any]
        )
    }

    private static func ssePayloads(_ string: String) -> [String] {
        string.components(separatedBy: "\n\n").compactMap { frame in
            frame.split(separator: "\n")
                .first(where: { $0.hasPrefix("data: ") })
                .map { String($0.dropFirst(6)) }
        }
    }
}

private actor ResponsesChatRecorder {
    private var requests: [[String: AnySendableJSON]] = []

    func record(_ data: Data) {
        guard let decoded = try? JSONDecoder().decode([String: AnySendableJSON].self, from: data) else {
            return
        }
        requests.append(decoded)
    }

    func last() -> [String: AnySendableJSON]? { requests.last }

    func all() -> [[String: AnySendableJSON]] { requests }
}

private enum AnySendableJSON: Codable, Sendable {
    case object([String: AnySendableJSON])
    case array([AnySendableJSON])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([AnySendableJSON].self) { self = .array(value) }
        else { self = .object(try container.decode([String: AnySendableJSON].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var foundationObject: Any {
        switch self {
        case .object(let value): return value.foundationObject
        case .array(let value): return value.map(\.foundationObject)
        case .string(let value): return value
        case .number(let value): return value.rounded() == value ? Int(value) : value
        case .bool(let value): return value
        case .null: return NSNull()
        }
    }
}

private extension Dictionary where Key == String, Value == AnySendableJSON {
    var foundationObject: [String: Any] { mapValues(\.foundationObject) }
}
