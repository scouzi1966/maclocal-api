import XCTest
import Vapor
import XCTVapor
@testable import AFMServer

/// CPU-only adapter contracts. Capture the actual translated request; model
/// output cannot establish that a template option reached the provider.
final class ResponsesReasoningContractTests: XCTestCase {
    private actor Capture {
        var body = Data()
        func record(_ value: Data) { body = value }
        func latest() -> Data { body }
    }

    private func translated(_ input: String) async throws -> [String: Any] {
        let app = try await Application.make(.testing)
        let capture = Capture()
        do {
            try app.register(collection: ResponsesController(defaultModelID: "contract") { request in
                if let body = request.body.data { await capture.record(Data(buffer: body)) }
                let response = Response(status: .ok)
                response.headers.contentType = .json
                response.body = .init(string: #"{"model":"contract","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#)
                return response
            })
            var headers = HTTPHeaders()
            headers.contentType = .json
            try await app.testable(method: .running(port: 0)).test(.POST, "/v1/responses",
                headers: headers, body: ByteBuffer(string: input)) { response async in
                XCTAssertEqual(response.status, .ok)
            }
            let data = await capture.latest()
            let result = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            try await app.asyncShutdown()
            return result
        } catch {
            try await app.asyncShutdown()
            throw error
        }
    }

    func testReturnedReasoningDoesNotCreateAnEmptyUserTurn() async throws {
        let result = try await translated(#"""
        {"input":[
          {"role":"user","content":"Read the file"},
          {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"inspect first"}]},
          {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},
          {"type":"function_call_output","call_id":"call_1","output":"file contents"},
          {"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"done"}]},
          {"role":"user","content":"Continue"}
        ]}
        """#)
        let messages = try XCTUnwrap(result["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.compactMap { $0["role"] as? String }, ["user", "assistant", "tool", "user"])
        XCTAssertEqual(messages.filter { ($0["role"] as? String) == "user" }.compactMap { $0["content"] as? String }, ["Read the file", "Continue"])
        XCTAssertEqual(messages.first { ($0["role"] as? String) == "tool" }?["tool_call_id"] as? String, "call_1")
    }

    func testReasoningOnlyItemDoesNotAlterOrdinaryMessageHistory() async throws {
        let plain = try await translated(#"{"input":[{"role":"user","content":"hello"}]}"#)
        let echoed = try await translated(#"{"input":[{"type":"reasoning","summary":[],"encrypted_content":"opaque"},{"role":"user","content":"hello"}]}"#)
        XCTAssertEqual(plain["messages"] as? NSArray, echoed["messages"] as? NSArray)
    }

    func testExplicitTemplateOffSurvivesMediumEffort() async throws {
        let result = try await translated(#"{"input":"hello","reasoning":{"effort":"medium"},"chat_template_kwargs":{"enable_thinking":false,"custom_marker":"keep"}}"#)
        XCTAssertEqual(result["reasoning_effort"] as? String, "medium")
        let kwargs = try XCTUnwrap(result["chat_template_kwargs"] as? [String: Any])
        XCTAssertEqual(kwargs["enable_thinking"] as? Bool, false)
        XCTAssertEqual(kwargs["custom_marker"] as? String, "keep")
    }

    func testEffortNoneOverridesExplicitTemplateOnWithoutDroppingOtherKwargs() async throws {
        let result = try await translated(#"{"input":"hello","reasoning":{"effort":"none"},"chat_template_kwargs":{"enable_thinking":true,"custom_marker":"keep"}}"#)
        let kwargs = try XCTUnwrap(result["chat_template_kwargs"] as? [String: Any])
        XCTAssertEqual(result["reasoning_effort"] as? String, "none")
        XCTAssertEqual(kwargs["enable_thinking"] as? Bool, false)
        XCTAssertEqual(kwargs["custom_marker"] as? String, "keep")
    }

    func testTemplateAndSamplingOptionsSurviveBothExternalTransportModes() async throws {
        for stream in [false, true] {
            let result = try await translated(#"{"input":"hello","stream":\#(stream),"seed":123,"top_k":0,"temperature":0,"top_p":1,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"high"}}"#)
            let kwargs = try XCTUnwrap(result["chat_template_kwargs"] as? [String: Any])
            XCTAssertEqual(kwargs["enable_thinking"] as? Bool, true)
            XCTAssertEqual(kwargs["reasoning_effort"] as? String, "high")
            XCTAssertEqual(result["seed"] as? Int, 123)
            XCTAssertEqual(result["top_k"] as? Int, 0)
            XCTAssertEqual(result["temperature"] as? Int, 0)
            XCTAssertEqual(result["top_p"] as? Int, 1)
        }
    }
}
