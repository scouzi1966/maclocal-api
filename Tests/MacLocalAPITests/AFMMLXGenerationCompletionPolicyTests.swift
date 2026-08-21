import XCTest
import AFMKitCore
@testable import AFMKitMLX

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

    func testSerialTelemetryReasonRejectsCancellationBeforeSuccess() async {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try MLXModelService.serialTelemetryFinishReason(
                generatedTokens: 1,
                maximumOutputTokens: 10
            )
        }

        do {
            _ = try await task.value
            XCTFail("cancelled serial generation must not report success")
        } catch is CancellationError {
            // Expected: the caller records AFMInferenceFailureReason.cancelled.
        } catch {
            XCTFail("unexpected cancellation error: \(error)")
        }
    }

    func testSerialTelemetryReasonPreservesStopAndLength() throws {
        XCTAssertEqual(
            try MLXModelService.serialTelemetryFinishReason(
                generatedTokens: 2,
                maximumOutputTokens: 2
            ),
            .length
        )
        XCTAssertEqual(
            try MLXModelService.serialTelemetryFinishReason(
                generatedTokens: 1,
                maximumOutputTokens: 2,
                stoppedBySequence: true
            ),
            .stop
        )
    }
}
