import XCTest
@testable import AFMKitMLX

final class AFMMLXGenerationLoopTimingPolicyTests: XCTestCase {
    func testIterationOverheadClampsNegativeDuration() {
        XCTAssertEqual(
            AFMMLXGenerationLoopTimingPolicy.iterationOverhead(
                startTime: 10,
                endTime: 9
            ),
            0
        )
    }

    func testStateAccumulatesIterationOverhead() {
        var state = AFMMLXGenerationLoopTimingState(generationStartTime: 100)

        state.observeIteration(startTime: 101, endTime: 101.002)
        state.observeIteration(startTime: 102, endTime: 102.003)

        XCTAssertEqual(state.totalLoopOverhead, 0.005, accuracy: 0.000_001)
    }

    func testSummaryComputesAverageAndPercentage() throws {
        var state = AFMMLXGenerationLoopTimingState(generationStartTime: 10)
        state.observeIteration(startTime: 11, endTime: 11.001)
        state.observeIteration(startTime: 12, endTime: 12.003)

        let summary = try XCTUnwrap(
            state.summary(tokenCount: 2, endTime: 14)
        )

        XCTAssertEqual(summary.tokenCount, 2)
        XCTAssertEqual(summary.totalGenerationTime, 4, accuracy: 0.000_001)
        XCTAssertEqual(summary.totalLoopOverhead, 0.004, accuracy: 0.000_001)
        XCTAssertEqual(summary.averageOverheadMicroseconds, 2_000, accuracy: 0.001)
        XCTAssertEqual(summary.overheadPercentage, 0.1, accuracy: 0.000_001)
    }

    func testSummarySkipsZeroTokensAndZeroGenerationTime() {
        XCTAssertNil(
            AFMMLXGenerationLoopTimingPolicy.summary(
                tokenCount: 0,
                generationStartTime: 1,
                endTime: 2,
                totalLoopOverhead: 1
            )
        )
        XCTAssertNil(
            AFMMLXGenerationLoopTimingPolicy.summary(
                tokenCount: 1,
                generationStartTime: 1,
                endTime: 1,
                totalLoopOverhead: 1
            )
        )
    }

    func testSummaryClampsNegativeLoopOverhead() throws {
        let summary = try XCTUnwrap(
            AFMMLXGenerationLoopTimingPolicy.summary(
                tokenCount: 4,
                generationStartTime: 10,
                endTime: 12,
                totalLoopOverhead: -1
            )
        )

        XCTAssertEqual(summary.totalLoopOverhead, 0)
        XCTAssertEqual(summary.averageOverheadMicroseconds, 0)
        XCTAssertEqual(summary.overheadPercentage, 0)
    }
}
