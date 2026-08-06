@testable import AFMKitCore
import XCTest

final class AFMGenerationLoopPolicyTests: XCTestCase {
    func testStopPollingUsesDefaultInterval() {
        XCTAssertFalse(
            AFMGenerationLoopPolicy.shouldPollStop(tokensSinceLastPoll: 7)
        )
        XCTAssertTrue(
            AFMGenerationLoopPolicy.shouldPollStop(tokensSinceLastPoll: 8)
        )
        XCTAssertTrue(
            AFMGenerationLoopPolicy.shouldPollStop(tokensSinceLastPoll: 9)
        )
    }

    func testStopPollingSupportsCustomInterval() {
        XCTAssertFalse(
            AFMGenerationLoopPolicy.shouldPollStop(
                tokensSinceLastPoll: 2,
                interval: 3
            )
        )
        XCTAssertTrue(
            AFMGenerationLoopPolicy.shouldPollStop(
                tokensSinceLastPoll: 3,
                interval: 3
            )
        )
    }

    func testReasoningFlushUsesSharedReasoningPolicy() {
        XCTAssertFalse(
            AFMGenerationLoopPolicy.shouldFlushReasoningUpdate(tokensSinceLastFlush: 3)
        )
        XCTAssertTrue(
            AFMGenerationLoopPolicy.shouldFlushReasoningUpdate(tokensSinceLastFlush: 4)
        )
    }

    func testStopPollStateOnlyReadsStopAtInterval() {
        var state = AFMStopPollState()

        XCTAssertFalse(state.observeToken(stopRequested: true, interval: 3))
        XCTAssertFalse(state.observeToken(stopRequested: true, interval: 3))
        XCTAssertTrue(state.observeToken(stopRequested: true, interval: 3))
        XCTAssertEqual(state.tokensSinceLastPoll, 0)
    }

    func testStopPollStatePreservesStopOnceObserved() {
        var state = AFMStopPollState()

        XCTAssertTrue(state.observeToken(stopRequested: true, interval: 1))
        XCTAssertTrue(state.observeToken(stopRequested: false, interval: 8))
    }

    func testFinishReasonUsesCancelledBeforeTokenLimit() {
        XCTAssertEqual(
            AFMGenerationLoopPolicy.finishReason(
                localShouldStop: true,
                tokenCount: 4,
                maxTokens: 4
            ),
            .cancelled
        )
    }

    func testFinishReasonUsesLengthWhenTokenLimitReached() {
        XCTAssertEqual(
            AFMGenerationLoopPolicy.finishReason(
                localShouldStop: false,
                tokenCount: 8,
                maxTokens: 8
            ),
            .length
        )
    }

    func testFinishReasonUsesStopBeforeTokenLimit() {
        XCTAssertEqual(
            AFMGenerationLoopPolicy.finishReason(
                localShouldStop: false,
                tokenCount: 7,
                maxTokens: 8
            ),
            .stop
        )
    }
}
