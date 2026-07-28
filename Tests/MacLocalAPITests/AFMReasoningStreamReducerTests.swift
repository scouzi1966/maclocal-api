import AFMKitCore
import XCTest

final class AFMReasoningStreamReducerTests: XCTestCase {
    func testPlainTextAccumulatesAndEntersFastPathAfterDetectionWindow() {
        var reducer = AFMReasoningStreamReducer(
            hasReasoningOutput: false,
            detectionWindow: 4
        )

        XCTAssertEqual(reducer.processChunk("a").outputChunk, "a")
        XCTAssertEqual(reducer.processChunk("b").outputChunk, "b")
        XCTAssertEqual(reducer.processChunk("c").outputChunk, "c")
        XCTAssertFalse(reducer.skipReasoningParser)

        XCTAssertEqual(reducer.processChunk("d").outputChunk, "d")
        XCTAssertTrue(reducer.skipReasoningParser)

        XCTAssertEqual(reducer.processChunk("e").outputChunk, "e")
        XCTAssertEqual(reducer.tokenCount, 5)
        XCTAssertEqual(reducer.accumulatedText, "abcde")

        let finalState = reducer.finalState()
        XCTAssertEqual(finalState.finalization.finalContent, "abcde")
        XCTAssertEqual(finalState.finalization.historyText, "abcde")
        XCTAssertFalse(finalState.finalization.hasReasoning)
    }

    func testReasoningChunkYieldsEmptyToKeepStreamAlive() {
        var reducer = AFMReasoningStreamReducer(
            hasReasoningOutput: false,
            detectionWindow: 4
        )

        let update = reducer.processChunk("<think>")

        XCTAssertEqual(update.outputChunk, "")
        XCTAssertFalse(reducer.skipReasoningParser)
    }

    func testExplicitReasoningFinalizesWithReasoningAndFinalContent() {
        var reducer = AFMReasoningStreamReducer(
            hasReasoningOutput: false,
            detectionWindow: 4
        )

        _ = reducer.processChunk("<think>")
        _ = reducer.processChunk("reason")
        _ = reducer.processChunk("</think>")
        let finalUpdate = reducer.processChunk("answer")

        XCTAssertEqual(finalUpdate.outputChunk, "answer")

        let finalState = reducer.finalState()
        XCTAssertEqual(finalState.parsedResult.reasoning, "reason")
        XCTAssertEqual(finalState.finalization.finalContent, "answer")
        XCTAssertEqual(finalState.finalization.historyText, "answer")
        XCTAssertTrue(finalState.finalization.hasReasoning)
    }

    func testDrainReasoningUpdateFlushesPendingPhaseState() {
        var reducer = AFMReasoningStreamReducer(
            hasReasoningOutput: false,
            detectionWindow: 4
        )

        _ = reducer.processChunk("<think>")

        let update = reducer.drainReasoningUpdate()
        XCTAssertEqual(update?.isReasoning, true)
        XCTAssertNil(update?.content)
        XCTAssertNotNil(update?.duration)
    }
}
