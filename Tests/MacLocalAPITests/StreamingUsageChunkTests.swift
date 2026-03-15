import XCTest
@testable import MacLocalAPI

final class StreamingUsageChunkTests: XCTestCase {
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
