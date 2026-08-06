#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationGenerationTelemetryTests: XCTestCase {
    func testComputesTokenThroughputOnlyAfterMultipleChunks() {
        XCTAssertNil(
            AFMFoundationGenerationTelemetryCalculator.decodeTokensPerSecond(
                outputTokens: 10,
                streamingMilliseconds: 1_000,
                streamChunkCount: 1
            )
        )

        XCTAssertEqual(
            AFMFoundationGenerationTelemetryCalculator.decodeTokensPerSecond(
                outputTokens: 10,
                streamingMilliseconds: 2_000,
                streamChunkCount: 2
            ),
            5
        )
    }

    func testBuildsAndFinalizesTelemetry() {
        let clock = ContinuousClock()
        let startedAt = clock.now
        let firstChunkAt = startedAt.advanced(by: .milliseconds(250))
        let sampledAt = startedAt.advanced(by: .milliseconds(1_250))
        let completedAt = startedAt.advanced(by: .milliseconds(2_250))

        let telemetry = AFMFoundationGenerationTelemetryCalculator.telemetry(
            inputTokens: 11,
            cachedInputTokens: 3,
            outputTokens: 20,
            reasoningTokens: 7,
            toolNames: ["search"],
            contextAction: "append",
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: 2
        )

        XCTAssertEqual(telemetry.inputTokens, 11)
        XCTAssertEqual(telemetry.cachedInputTokens, 3)
        XCTAssertEqual(telemetry.outputTokens, 20)
        XCTAssertEqual(telemetry.reasoningTokens, 7)
        XCTAssertEqual(telemetry.toolNames, ["search"])
        XCTAssertEqual(telemetry.contextAction, "append")
        XCTAssertEqual(try XCTUnwrap(telemetry.timeToFirstTokenMilliseconds), 250, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(telemetry.streamingMilliseconds), 1_000, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(telemetry.totalMilliseconds), 1_250, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(telemetry.tokensPerSecond), 20, accuracy: 0.001)

        let finalized = AFMFoundationGenerationTelemetryCalculator.finalize(
            telemetry,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            completedAt: completedAt,
            streamChunkCount: 3
        )

        XCTAssertEqual(try XCTUnwrap(finalized.streamingMilliseconds), 2_000, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(finalized.totalMilliseconds), 2_250, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(finalized.tokensPerSecond), 10, accuracy: 0.001)
        XCTAssertEqual(finalized.streamChunkCount, 3)
    }
}
#endif
