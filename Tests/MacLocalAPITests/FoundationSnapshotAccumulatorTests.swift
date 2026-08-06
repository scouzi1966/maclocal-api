#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationSnapshotAccumulatorTests: XCTestCase {
    func testComputesResponseDeltasAndReplacementSnapshots() {
        var accumulator = AFMFoundationSnapshotAccumulator<[String]>(initialProgressState: [])

        let first = accumulator.consume(content: "Hello", progressState: [])
        let second = accumulator.consume(content: "Hello there", progressState: [])
        let replacement = accumulator.consume(content: "Reset", progressState: [])

        XCTAssertEqual(first.responseDelta, "Hello")
        XCTAssertTrue(first.firstChunkStarted)
        XCTAssertEqual(first.streamChunkCount, 1)
        XCTAssertEqual(second.responseDelta, " there")
        XCTAssertEqual(second.streamChunkCount, 2)
        XCTAssertEqual(replacement.responseDelta, "Reset")
        XCTAssertEqual(replacement.streamChunkCount, 3)
    }

    func testTracksReasoningProgressSeparatelyFromResponseText() {
        var accumulator = AFMFoundationSnapshotAccumulator<[String]>(initialProgressState: [])

        let reasoning = accumulator.consume(
            content: "",
            progressState: [],
            reasoningContent: "Check constraints"
        )
        let response = accumulator.consume(
            content: "Final",
            progressState: [],
            reasoningContent: "Check constraints"
        )

        XCTAssertNil(reasoning.responseDelta)
        XCTAssertTrue(reasoning.shouldYieldProgressUpdate)
        XCTAssertTrue(reasoning.firstChunkStarted)
        XCTAssertEqual(reasoning.reasoningContent, "Check constraints")
        XCTAssertTrue(reasoning.isInReasoningPhase)
        XCTAssertEqual(reasoning.streamChunkCount, 1)
        XCTAssertEqual(response.responseDelta, "Final")
        XCTAssertEqual(response.reasoningContent, "Check constraints")
        XCTAssertFalse(response.isInReasoningPhase)
        XCTAssertEqual(response.streamChunkCount, 2)
    }

    func testYieldsProgressUpdateWithoutStartingFirstChunkForToolOnlyChanges() {
        var accumulator = AFMFoundationSnapshotAccumulator<[String]>(initialProgressState: [])

        let update = accumulator.consume(content: "", progressState: ["tool-1"])

        XCTAssertNil(update.responseDelta)
        XCTAssertTrue(update.shouldYieldProgressUpdate)
        XCTAssertFalse(update.firstChunkStarted)
        XCTAssertEqual(update.streamChunkCount, 0)
        XCTAssertEqual(update.progressState, ["tool-1"])
    }
}
#endif
