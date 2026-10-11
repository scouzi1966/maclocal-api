import XCTest
import Vapor
import XCTVapor
@testable import AFMServer

/// CPU-only adapter contracts. Capture the actual translated request; model
/// output cannot establish that a template option reached the provider.
final class ResponsesReasoningContractTests: XCTestCase {
    func testStreamingItemsReconstructWithoutDuplicateTextOrArgumentsBeforeDone() async throws {
        let app = try await Application.make(.testing)
        do {
            try app.register(collection: ResponsesController(defaultModelID: "contract") { _ in
                let response = Response(status: .ok)
                response.headers.contentType = .json
                response.body = .init(string: #"{"model":"contract","choices":[{"message":{"role":"assistant","content":"Ready.","reasoning_content":"Read first.","tool_calls":[{"id":"call_read","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}]}"#)
                return response
            })
            var headers = HTTPHeaders()
            headers.contentType = .json
            try await app.testable(method: .running(port: 0)).test(.POST, "/v1/responses",
                headers: headers, body: ByteBuffer(string: #"{"input":"Read README.md","stream":true}"#)) { response async throws in
                XCTAssertEqual(response.status, .ok)
                let events = try response.body.string.split(separator: "\n").compactMap { line -> [String: Any]? in
                    guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
                    return try JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any]
                }
                var live: [String: [String: Any]] = [:]
                var doneCount = 0
                for event in events {
                    let type = event["type"] as? String ?? ""
                    if type == "response.output_item.added" {
                        let item = try XCTUnwrap(event["item"] as? [String: Any])
                        let id = try XCTUnwrap(item["id"] as? String)
                        switch item["type"] as? String {
                        case "reasoning": XCTAssertEqual((item["summary"] as? [Any])?.count, 0)
                        case "message": XCTAssertEqual((item["content"] as? [Any])?.count, 0)
                        case "function_call": XCTAssertEqual(item["arguments"] as? String, "")
                        default: XCTFail("Unexpected fixture item")
                        }
                        if item["status"] != nil { XCTAssertEqual(item["status"] as? String, "in_progress") }
                        live[id] = item
                    } else if type == "response.output_item.done" {
                        let final = try XCTUnwrap(event["item"] as? [String: Any])
                        let id = try XCTUnwrap(final["id"] as? String)
                        let accumulated = try XCTUnwrap(live[id])
                        // Compare BEFORE canonical item/terminal replacement can
                        // mask a duplicate introduced by additive SDK reducers.
                        for field in ["content", "summary"] where final[field] != nil {
                            XCTAssertEqual(accumulated[field] as? NSArray, final[field] as? NSArray)
                        }
                        if final["arguments"] != nil {
                            XCTAssertEqual(accumulated["arguments"] as? String, final["arguments"] as? String)
                            let arguments = try XCTUnwrap(accumulated["arguments"] as? String)
                            XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(arguments.utf8)))
                        }
                        doneCount += 1
                    } else if let id = event["item_id"] as? String, var item = live[id] {
                        if type == "response.function_call_arguments.delta" {
                            item["arguments"] = (item["arguments"] as? String ?? "") + (event["delta"] as? String ?? "")
                        } else if type == "response.content_part.added" || type == "response.reasoning_summary_part.added" {
                            let field = type == "response.content_part.added" ? "content" : "summary"
                            var parts = item[field] as? [[String: Any]] ?? []
                            parts.append(try XCTUnwrap(event["part"] as? [String: Any]))
                            item[field] = parts
                        } else if type == "response.output_text.delta" || type == "response.reasoning_summary_text.delta" {
                            let field = type == "response.output_text.delta" ? "content" : "summary"
                            let indexField = field == "content" ? "content_index" : "summary_index"
                            let index = try XCTUnwrap(event[indexField] as? Int)
                            var parts = try XCTUnwrap(item[field] as? [[String: Any]])
                            XCTAssertTrue(parts.indices.contains(index))
                            guard parts.indices.contains(index) else { continue }
                            parts[index]["text"] = (parts[index]["text"] as? String ?? "") + (event["delta"] as? String ?? "")
                            item[field] = parts
                        }
                        live[id] = item
                    }
                }
                XCTAssertEqual(doneCount, 3)
            }
            try await app.asyncShutdown()
        } catch {
            try await app.asyncShutdown()
            throw error
        }
    }

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
