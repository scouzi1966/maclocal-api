import XCTest
import AFMOpenAICompat
@testable import AFMServer

final class CompletedToolCallAccumulatorTests: XCTestCase {
    private func call(_ index: Int?, id: String, arguments: String = "{}") -> ResponseToolCall {
        ResponseToolCall(index: index, id: id, type: "function",
                         function: .init(name: "get_weather", arguments: arguments))
    }

    func testSingletonChunksPreserveBothCallsAndOrder() {
        var accumulator = CompletedToolCallAccumulator()
        accumulator.consume([call(0, id: "first")])
        accumulator.consume([call(1, id: "second")])
        XCTAssertEqual(accumulator.toolCalls?.map(\.id), ["first", "second"])
    }

    func testRepeatedIndexReplacesSnapshotWithoutDuplicatingOrConcatenatingArguments() {
        var accumulator = CompletedToolCallAccumulator()
        accumulator.consume([call(0, id: "first"), call(1, id: "second")])
        let updated = call(0, id: "first", arguments: #"{"location":"Berlin"}"#)
        accumulator.consume([updated])
        accumulator.consume([updated, call(1, id: "second")])
        XCTAssertEqual(accumulator.toolCalls?.map(\.id), ["first", "second"])
        XCTAssertEqual(accumulator.toolCalls?.first?.function.arguments, updated.function.arguments)
    }

    func testUnindexedCallsUseIdentityAndEmptyChunksDoNotClearCalls() {
        var accumulator = CompletedToolCallAccumulator()
        XCTAssertNil(accumulator.toolCalls)
        accumulator.consume([])
        XCTAssertNil(accumulator.toolCalls)
        accumulator.consume([call(nil, id: "first"), call(nil, id: "second")])
        accumulator.consume([call(nil, id: "first", arguments: #"{"location":"Paris"}"#)])
        accumulator.consume([])
        XCTAssertEqual(accumulator.toolCalls?.map(\.id), ["first", "second"])
        XCTAssertEqual(accumulator.toolCalls?.first?.function.arguments, #"{"location":"Paris"}"#)
    }

    func testStreamCollectorPreservesCompletedCallsAcrossUsageAndRepeatedSnapshots() async throws {
        let first = call(0, id: "first", arguments: #"{"location":"Berlin"}"#)
        let second = call(1, id: "second")
        let stream = AsyncThrowingStream<AFMServerStreamChunk, Error> { continuation in
            continuation.yield(.init(text: "", toolCalls: [first]))
            continuation.yield(.init(text: "", toolCalls: [second]))
            continuation.yield(.init(text: "", toolCalls: [first]))
            continuation.yield(.init(text: "", promptTokens: 10, completionTokens: 71))
            continuation.finish()
        }
        let result: AFMChatStreamingResult = ("fixture", stream, 10, nil, nil, nil, nil)
        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        XCTAssertEqual(collected.toolCalls?.map(\.id), ["first", "second"])
        XCTAssertEqual(collected.finishReason, "tool_calls")
        XCTAssertEqual(collected.completionTokens, 71)
    }
}
