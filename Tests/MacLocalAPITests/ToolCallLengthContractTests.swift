import Foundation
import XCTest
import Vapor
import XCTVapor
@testable import AFMKit
@testable import AFMServer

final class ToolCallLengthContractTests: XCTestCase {
    private static let maximumTokens = 20
    private static let reasoning = "Inspect the file before editing."
    private static let call = ResponseToolCall(
        index: 0, id: "call_read", type: "function",
        function: .init(name: "read_file", arguments: #"{"path":"README.md"}"#)
    )

    private static func finalize(tokens: Int, stoppedBySequence: Bool = false) -> FinalizedAssistantTurn {
        MLXChatCompletionsController.finalizeAssistantTurn(
            content: "<think>\(reasoning)</think>", toolCalls: [call], toolChoice: nil,
            extractThinking: true, thinkStartTag: "<think>", thinkEndTag: "</think>",
            stoppedBySequence: stoppedBySequence, completionTokens: tokens,
            maxTokens: maximumTokens, sanitizeContent: { $0 }
        )
    }

    private static func response(tokens: Int, stoppedBySequence: Bool = false) -> ChatCompletionResponse {
        let turn = finalize(tokens: tokens, stoppedBySequence: stoppedBySequence)
        return ChatCompletionResponse(
            model: "test-model", toolCalls: turn.toolCalls ?? [],
            reasoningContent: turn.reasoningContent, finishReason: turn.finishReason,
            promptTokens: 10, completionTokens: tokens
        )
    }

    func testUnderBudgetToolTurnKeepsToolCallsReasonAndReasoning() {
        let turn = Self.finalize(tokens: Self.maximumTokens - 1)
        XCTAssertEqual(turn.finishReason, "tool_calls")
        XCTAssertEqual(turn.reasoningContent, Self.reasoning)
        XCTAssertEqual(turn.toolCalls?.first?.function.arguments, Self.call.function.arguments)
        XCTAssertNil(turn.content)
    }

    func testToolTurnAtOrAboveBudgetKeepsDataButReportsLength() {
        // The provider exposes a count, not exact EOS-vs-budget termination
        // evidence. Apply the same conservative convention as ordinary text.
        for count in [Self.maximumTokens, Self.maximumTokens + 1] {
            let turn = Self.finalize(tokens: count)
            XCTAssertEqual(turn.finishReason, "length")
            XCTAssertEqual(turn.reasoningContent, Self.reasoning)
            XCTAssertEqual(turn.toolCalls?.first?.function.arguments, Self.call.function.arguments)
            XCTAssertNil(turn.content)
        }
    }

    func testExplicitStopAtBudgetDoesNotBecomeLength() {
        let turn = Self.finalize(tokens: Self.maximumTokens, stoppedBySequence: true)
        XCTAssertEqual(turn.finishReason, "tool_calls")
        XCTAssertEqual(turn.toolCalls?.first?.id, Self.call.id)
    }

    func testNonStreamingToolResponsePreservesFinalizedLength() throws {
        let encoded = try JSONEncoder().encode(Self.response(tokens: Self.maximumTokens))
        let object = try Self.json(encoded)
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "length")
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        XCTAssertEqual(message["reasoning_content"] as? String, Self.reasoning)
        XCTAssertEqual((message["tool_calls"] as? [[String: Any]])?.count, 1)
    }

    func testResponsesToolAtBudgetIsIncompleteWithArgumentsRetained() async throws {
        try await withResponses(tokens: Self.maximumTokens, streaming: false) { body in
            try Self.assertIncompleteResource(Self.json(Data(body.utf8)))
        }
    }

    func testStreamingResponsesToolAtBudgetEmitsIncompleteTerminalAndItem() async throws {
        try await withResponses(tokens: Self.maximumTokens, streaming: true) { body in
            let events = try Self.events(body)
            let types = events.compactMap { $0["type"] as? String }
            XCTAssertEqual(types.last, "response.incomplete")
            XCTAssertFalse(types.contains("response.completed"))
            let terminal = try XCTUnwrap(events.last?["response"] as? [String: Any])
            try Self.assertIncompleteResource(terminal)
            let completedItemEvents = events.filter { $0["type"] as? String == "response.output_item.done" }
            let callItem = try XCTUnwrap(completedItemEvents.compactMap { $0["item"] as? [String: Any] }
                .first { $0["type"] as? String == "function_call" })
            XCTAssertEqual(callItem["status"] as? String, "incomplete")
            XCTAssertEqual(callItem["arguments"] as? String, Self.call.function.arguments)
            XCTAssertTrue(body.hasSuffix("data: [DONE]\n\n"))
        }
    }

    func testResponsesUnderBudgetToolStillCompletes() async throws {
        try await withResponses(tokens: Self.maximumTokens - 1, streaming: false) { body in
            let resource = try Self.json(Data(body.utf8))
            XCTAssertEqual(resource["status"] as? String, "completed")
            XCTAssertTrue(resource["incomplete_details"] is NSNull)
            let output = try XCTUnwrap(resource["output"] as? [[String: Any]])
            let tool = try XCTUnwrap(output.first { $0["type"] as? String == "function_call" })
            XCTAssertEqual(tool["status"] as? String, "completed")
        }
    }

    func testTruncatedTextItemIsIncompleteInBothResponseTransports() async throws {
        for streaming in [false, true] {
            try await withResponses(tokens: Self.maximumTokens, streaming: streaming, contentOnly: true) { body in
                let resource: [String: Any]
                if streaming {
                    let terminal = try XCTUnwrap(Self.events(body).last)
                    XCTAssertEqual(terminal["type"] as? String, "response.incomplete")
                    resource = try XCTUnwrap(terminal["response"] as? [String: Any])
                } else {
                    resource = try Self.json(Data(body.utf8))
                }
                XCTAssertEqual(resource["status"] as? String, "incomplete")
                let output = try XCTUnwrap(resource["output"] as? [[String: Any]])
                let message = try XCTUnwrap(output.first { $0["type"] as? String == "message" })
                XCTAssertEqual(message["status"] as? String, "incomplete")
                XCTAssertEqual((message["content"] as? [[String: Any]])?.first?["text"] as? String, "Partial answer")
            }
        }
    }

    private func withResponses(
        tokens: Int, streaming: Bool, contentOnly: Bool = false,
        check: @escaping (String) throws -> Void
    ) async throws {
        let app = try await Application.make(.testing)
        do {
            let chatData = contentOnly
                ? Data(#"{"model":"test-model","choices":[{"message":{"role":"assistant","content":"Partial answer"},"finish_reason":"length"}]}"#.utf8)
                : try JSONEncoder().encode(Self.response(tokens: tokens))
            try app.register(collection: ResponsesController(defaultModelID: "test-model") { _ in
                let response = Response(status: .ok)
                response.headers.contentType = .json
                response.body = .init(data: chatData)
                return response
            })
            var headers = HTTPHeaders()
            headers.contentType = .json
            let body = #"{"input":"Inspect README.md","max_output_tokens":\#(Self.maximumTokens),"stream":\#(streaming)}"#
            try await app.testable(method: .running(port: 0)).test(
                .POST, "/v1/responses", headers: headers, body: ByteBuffer(string: body)
            ) { response async in
                XCTAssertEqual(response.status, .ok)
                do { try check(response.body.string) }
                catch { XCTFail("Response assertion failed: \(error)") }
            }
            try await app.asyncShutdown()
        } catch {
            try await app.asyncShutdown()
            throw error
        }
    }

    private static func assertIncompleteResource(_ resource: [String: Any]) throws {
        XCTAssertEqual(resource["status"] as? String, "incomplete")
        let details = try XCTUnwrap(resource["incomplete_details"] as? [String: Any])
        XCTAssertEqual(details["reason"] as? String, "max_output_tokens")
        let output = try XCTUnwrap(resource["output"] as? [[String: Any]])
        let tool = try XCTUnwrap(output.first { $0["type"] as? String == "function_call" })
        XCTAssertEqual(tool["status"] as? String, "incomplete")
        XCTAssertEqual(tool["arguments"] as? String, call.function.arguments)
        let thought = try XCTUnwrap(output.first { $0["type"] as? String == "reasoning" })
        XCTAssertEqual((thought["summary"] as? [[String: Any]])?.first?["text"] as? String, reasoning)
    }

    private static func json(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func events(_ body: String) throws -> [[String: Any]] {
        try body.components(separatedBy: "\n\n").compactMap { frame in
            guard let line = frame.split(separator: "\n").first(where: { $0.hasPrefix("data: ") }) else {
                return nil
            }
            let payload = String(line.dropFirst(6))
            guard payload != "[DONE]" else { return nil }
            return try json(Data(payload.utf8))
        }
    }
}
