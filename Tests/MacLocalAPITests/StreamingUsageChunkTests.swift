import XCTest
@testable import MacLocalAPI

final class StreamingUsageChunkTests: XCTestCase {
// dimensions: streaming=true, execution=serial
    func testFinalizeAssistantTurnPrefersToolCallsOverContent() {
        let toolCall = ResponseToolCall(
            index: 0,
            id: "call_test",
            type: "function",
            function: ResponseToolCallFunction(name: "read_file", arguments: #"{"path":"README.md"}"#)
        )

        let finalized = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "ignored",
            toolCalls: [toolCall],
            toolChoice: nil,
            extractThinking: true,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 5,
            maxTokens: 20,
            sanitizeContent: { $0 }
        )

        XCTAssertEqual(finalized.finishReason, "tool_calls")
        XCTAssertNil(finalized.content)
        XCTAssertNil(finalized.reasoningContent)
        XCTAssertEqual(finalized.toolCalls?.count, 1)
        XCTAssertEqual(finalized.toolCalls?.first?.function.name, "read_file")
    }

    func testFinalizeAssistantTurnExtractsThinkingAndComputesLengthFinishReason() {
        let finalized = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "<think>plan</think>\nAnswer",
            toolCalls: nil,
            toolChoice: nil,
            extractThinking: true,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 20,
            maxTokens: 20,
            sanitizeContent: { $0 }
        )

        XCTAssertEqual(finalized.finishReason, "length")
        XCTAssertEqual(finalized.content, "Answer")
        XCTAssertEqual(finalized.reasoningContent, "plan")
        XCTAssertNil(finalized.toolCalls)
    }

    func testFinalizeAssistantTurnFiltersToolCallsToNamedChoice() {
        let wrongToolCall = ResponseToolCall(
            index: 0,
            id: "call_wrong",
            type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: #"{"location":"Berlin"}"#)
        )

        let finalized = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "fallback",
            toolCalls: [wrongToolCall],
            toolChoice: .function(.init(type: "function", function: .init(name: "read_file"))),
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 5,
            maxTokens: 20,
            sanitizeContent: { $0 }
        )

        XCTAssertEqual(finalized.finishReason, "stop")
        XCTAssertEqual(finalized.content, "fallback")
        XCTAssertNil(finalized.toolCalls)
    }

    func testNonStreamingResponsePreservesExplicitFinishReason() throws {
        let response = ChatCompletionResponse(
            id: "chat-1234",
            model: "test-model",
            content: "{\"ok\":true}",
            finishReason: "length",
            promptTokens: 12,
            completionTokens: 24
        )

        let data = try JSONEncoder().encode(response)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        let firstChoice = try XCTUnwrap(choices.first)

        XCTAssertEqual(firstChoice["finish_reason"] as? String, "length")
    }

    func testUsageSummaryChunkEncodesEmptyChoices() throws {
        let usage = StreamUsage(promptTokens: 10, completionTokens: 4, completionTime: 0.5, promptTime: 0.1)
        let chunk = ChatCompletionStreamResponse(
            id: "stream-1234",
            model: "test-model",
            usage: usage,
            timings: StreamTimings(prompt_n: 10, prompt_ms: 100, predicted_n: 4, predicted_ms: 500)
        )

        let data = try JSONEncoder().encode(chunk)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["object"] as? String, "chat.completion.chunk")
        XCTAssertEqual((json["choices"] as? [Any])?.count, 0)

        let usageJSON = try XCTUnwrap(json["usage"] as? [String: Any])
        XCTAssertEqual(usageJSON["prompt_tokens"] as? Int, 10)
        XCTAssertEqual(usageJSON["completion_tokens"] as? Int, 4)
    }

    func testFoundationCommonPrefixLengthHandlesMutableSnapshots() {
        let first = #"{"age": 50, "name": ""}"#
        let second = #"{"age": 50, "name": "Katherine Johnson", "occupation": "mathematician"}"#

        let prefixLength = FoundationModelService.commonPrefixLength(first, second)
        XCTAssertEqual(String(second.prefix(prefixLength)), #"{"age": 50, "name": ""#)
    }

    func testTerminalFinishChunkOmitsUsageSummaryShape() throws {
        let chunk = ChatCompletionStreamResponse(
            id: "stream-1234",
            model: "test-model",
            content: "",
            isFinished: true,
            finishReason: "stop"
        )

        let data = try JSONEncoder().encode(chunk)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual((json["choices"] as? [Any])?.count, 1)
        XCTAssertNil(json["usage"])
    }
}
