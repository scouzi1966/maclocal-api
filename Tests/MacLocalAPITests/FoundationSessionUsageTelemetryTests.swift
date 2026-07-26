#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import FoundationModels
import XCTest

@available(macOS 27.0, *)
final class FoundationSessionUsageTelemetryTests: XCTestCase {
    func testMapsLanguageModelSessionUsageIntoTelemetry() {
        let clock = ContinuousClock()
        let startedAt = clock.now
        let firstChunkAt = startedAt.advanced(by: .milliseconds(100))
        let sampledAt = startedAt.advanced(by: .milliseconds(600))
        let usage = LanguageModelSession.Usage(
            input: .init(totalTokenCount: 12, cachedTokenCount: 4),
            output: .init(totalTokenCount: 30, reasoningTokenCount: 8)
        )

        let telemetry = AFMFoundationGenerationTelemetryCalculator.telemetry(
            usage: usage,
            toolNames: ["lookup"],
            contextAction: "truncate",
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: 3
        )

        XCTAssertEqual(telemetry.inputTokens, 12)
        XCTAssertEqual(telemetry.cachedInputTokens, 4)
        XCTAssertEqual(telemetry.outputTokens, 30)
        XCTAssertEqual(telemetry.reasoningTokens, 8)
        XCTAssertEqual(telemetry.toolNames, ["lookup"])
        XCTAssertEqual(telemetry.contextAction, "truncate")
        XCTAssertEqual(try XCTUnwrap(telemetry.timeToFirstTokenMilliseconds), 100, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(telemetry.streamingMilliseconds), 500, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(telemetry.tokensPerSecond), 60, accuracy: 0.001)
    }
}
#endif
