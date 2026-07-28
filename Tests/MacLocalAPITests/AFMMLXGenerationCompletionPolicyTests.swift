import XCTest
import AFMKitCore
import AFMKitMLX

final class AFMMLXGenerationCompletionPolicyTests: XCTestCase {
    func testSummaryUsesFinalContentForNormalStop() {
        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: false)
        _ = reducer.processChunk("hello")
        _ = reducer.processChunk(" world")

        let summary = AFMMLXGenerationCompletionPolicy.summary(
            finalState: reducer.finalState(),
            tokenCount: reducer.tokenCount,
            localShouldStop: false,
            stopRequested: false,
            maxTokens: 10
        )

        XCTAssertEqual(summary.tokenCount, 2)
        XCTAssertEqual(summary.maxTokens, 10)
        XCTAssertEqual(summary.finishReason, .stop)
        XCTAssertFalse(summary.reachedMaxTokens)
        XCTAssertEqual(summary.finalContent, "hello world")
        XCTAssertEqual(summary.historyText, "hello world")
        XCTAssertFalse(summary.hasReasoning)
    }

    func testSummaryReportsLengthAtMaxTokens() {
        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: false)
        _ = reducer.processChunk("a")
        _ = reducer.processChunk("b")

        let summary = AFMMLXGenerationCompletionPolicy.summary(
            finalState: reducer.finalState(),
            tokenCount: reducer.tokenCount,
            localShouldStop: false,
            stopRequested: false,
            maxTokens: 2
        )

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertTrue(summary.reachedMaxTokens)
    }

    func testSummaryReportsCancellationWhenStopObserved() {
        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: false)
        _ = reducer.processChunk("partial")

        let summary = AFMMLXGenerationCompletionPolicy.summary(
            finalState: reducer.finalState(),
            tokenCount: reducer.tokenCount,
            localShouldStop: true,
            stopRequested: false,
            maxTokens: 20
        )

        XCTAssertEqual(summary.finishReason, .cancelled)
        XCTAssertFalse(summary.reachedMaxTokens)
    }

    func testSummaryReportsCancellationWhenStopRequestedBeforePoll() {
        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: false)
        _ = reducer.processChunk("partial")

        let summary = AFMMLXGenerationCompletionPolicy.summary(
            finalState: reducer.finalState(),
            tokenCount: reducer.tokenCount,
            localShouldStop: false,
            stopRequested: true,
            maxTokens: 20
        )

        XCTAssertEqual(summary.finishReason, .cancelled)
    }

    func testSummaryUsesReasoningAsHistoryFallback() {
        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: true)
        _ = reducer.processChunk("<think>hidden")

        let summary = AFMMLXGenerationCompletionPolicy.summary(
            finalState: reducer.finalState(),
            tokenCount: reducer.tokenCount,
            localShouldStop: true,
            stopRequested: true,
            maxTokens: 20
        )

        XCTAssertEqual(summary.finalContent, "")
        XCTAssertEqual(summary.historyText, "hidden")
        XCTAssertTrue(summary.hasReasoning)
        XCTAssertEqual(summary.finishReason, .cancelled)
    }
}
