#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationStringStreamProcessorTests: XCTestCase {
    func testNeutralConsumeTracksProgressReasoningAndTelemetry() {
        var processor = AFMFoundationStringStreamProcessor<[String]>(initialProgressState: [])
        let startedAt = ContinuousClock.now
        let firstAt = startedAt.advanced(by: .milliseconds(100))
        let secondAt = startedAt.advanced(by: .milliseconds(600))

        let reasoning = processor.consume(
            content: "",
            usage: AFMFoundationGenerationUsage(
                inputTokens: 10,
                cachedInputTokens: 2,
                outputTokens: 0,
                reasoningTokens: 4
            ),
            progressState: [],
            progressNames: [],
            reasoningContent: "Thinking",
            contextAction: "trimmed",
            startedAt: startedAt,
            sampledAt: firstAt
        )
        let response = processor.consume(
            content: "Answer",
            usage: AFMFoundationGenerationUsage(
                inputTokens: 10,
                cachedInputTokens: 2,
                outputTokens: 6,
                reasoningTokens: 4
            ),
            progressState: ["search"],
            progressNames: ["search"],
            reasoningContent: "Thinking",
            contextAction: "trimmed",
            startedAt: startedAt,
            sampledAt: secondAt
        )

        XCTAssertNil(reasoning.snapshotUpdate.responseDelta)
        XCTAssertTrue(reasoning.snapshotUpdate.isInReasoningPhase)
        XCTAssertEqual(response.snapshotUpdate.responseDelta, "Answer")
        XCTAssertEqual(response.snapshotUpdate.progressState, ["search"])
        XCTAssertEqual(response.telemetry.inputTokens, 10)
        XCTAssertEqual(response.telemetry.cachedInputTokens, 2)
        XCTAssertEqual(response.telemetry.outputTokens, 6)
        XCTAssertEqual(response.telemetry.reasoningTokens, 4)
        XCTAssertEqual(response.telemetry.toolNames, ["search"])
        XCTAssertEqual(response.telemetry.contextAction, "trimmed")
        XCTAssertEqual(response.telemetry.timeToFirstTokenMilliseconds ?? 0, 100, accuracy: 1)
        XCTAssertEqual(response.telemetry.streamChunkCount, 2)
        XCTAssertEqual(processor.firstChunkAt, firstAt)
        XCTAssertEqual(processor.streamChunkCount, 2)
    }

    func testRunnerLoopsSnapshotsAndFinalizesLatestTelemetry() async throws {
        struct Frame {
            let content: String
            let usage: AFMFoundationGenerationUsage
            let progress: [String]
        }

        let frames = AsyncStream { continuation in
            continuation.yield(
                Frame(
                    content: "Hello",
                    usage: AFMFoundationGenerationUsage(
                        inputTokens: 4,
                        cachedInputTokens: 0,
                        outputTokens: 1,
                        reasoningTokens: 0
                    ),
                    progress: []
                )
            )
            continuation.yield(
                Frame(
                    content: "Hello there",
                    usage: AFMFoundationGenerationUsage(
                        inputTokens: 4,
                        cachedInputTokens: 0,
                        outputTokens: 2,
                        reasoningTokens: 0
                    ),
                    progress: ["lookup"]
                )
            )
            continuation.finish()
        }

        let startedAt = ContinuousClock.now
        var deltas: [String] = []
        let result = try await AFMFoundationStringStreamRunner.run(
            frames,
            initialProgressState: [],
            contextAction: nil,
            startedAt: startedAt
        ) { frame, processor, contextAction, sampledAt in
            processor.consume(
                content: frame.content,
                usage: frame.usage,
                progressState: frame.progress,
                progressNames: frame.progress,
                contextAction: contextAction,
                startedAt: startedAt,
                sampledAt: sampledAt
            )
        } receive: { update in
            if let delta = update.snapshotUpdate.responseDelta {
                deltas.append(delta)
            }
        }

        XCTAssertEqual(deltas, ["Hello", " there"])
        XCTAssertEqual(result.progressState, ["lookup"])
        XCTAssertEqual(result.streamChunkCount, 2)
        XCTAssertNotNil(result.firstChunkAt)
        XCTAssertEqual(result.telemetry?.outputTokens, 2)
        XCTAssertEqual(result.telemetry?.toolNames, ["lookup"])
        XCTAssertEqual(result.telemetry?.streamChunkCount, 2)
        XCTAssertNotNil(result.telemetry?.totalMilliseconds)
    }

    func testNativeDecodeRateSurvivesSingleSnapshotFinalization() {
        var processor = AFMFoundationStringStreamProcessor<[String]>(initialProgressState: [])
        let startedAt = ContinuousClock.now
        let completedAt = startedAt.advanced(by: .seconds(2))

        let update = processor.consume(
            content: "Answer",
            usage: AFMFoundationGenerationUsage(
                inputTokens: 8,
                cachedInputTokens: 0,
                outputTokens: 64,
                reasoningTokens: 0
            ),
            progressState: [],
            progressNames: [],
            contextAction: nil,
            nativeTokensPerSecond: 128,
            startedAt: startedAt,
            sampledAt: completedAt
        )
        let finalized = processor.finalize(
            update.telemetry,
            startedAt: startedAt,
            completedAt: completedAt
        )

        XCTAssertEqual(update.telemetry.tokensPerSecond, 128)
        XCTAssertEqual(finalized.tokensPerSecond, 128)
        XCTAssertEqual(finalized.streamChunkCount, 1)
    }
}
#endif
