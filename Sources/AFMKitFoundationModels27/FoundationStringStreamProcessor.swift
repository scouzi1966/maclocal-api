#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationStringStreamUpdate<ProgressState: Equatable>: Equatable {
    public let snapshotUpdate: AFMFoundationSnapshotUpdate<ProgressState>
    public let telemetry: AFMFoundationGenerationTelemetry

    public init(
        snapshotUpdate: AFMFoundationSnapshotUpdate<ProgressState>,
        telemetry: AFMFoundationGenerationTelemetry
    ) {
        self.snapshotUpdate = snapshotUpdate
        self.telemetry = telemetry
    }
}

@available(macOS 27.0, *)
public struct AFMFoundationStringStreamProcessor<ProgressState: Equatable> {
    private var accumulator: AFMFoundationSnapshotAccumulator<ProgressState>
    public private(set) var firstChunkAt: ContinuousClock.Instant?

    public var streamChunkCount: Int {
        accumulator.streamChunkCount
    }

    public init(initialProgressState: ProgressState) {
        self.accumulator = AFMFoundationSnapshotAccumulator(initialProgressState: initialProgressState)
    }

    public mutating func consume(
        snapshot: LanguageModelSession.ResponseStream<String>.Snapshot,
        progressState: ProgressState,
        progressNames: [String],
        reasoningContent: String = "",
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        sampledAt: ContinuousClock.Instant = ContinuousClock.now
    ) -> AFMFoundationStringStreamUpdate<ProgressState> {
        let update = accumulator.consume(
            content: snapshot.content,
            progressState: progressState,
            reasoningContent: reasoningContent
        )
        if update.firstChunkStarted, firstChunkAt == nil {
            firstChunkAt = sampledAt
        }
        let telemetry = AFMFoundationGenerationTelemetryCalculator.telemetry(
            usage: snapshot.usage,
            toolNames: progressNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: update.streamChunkCount
        )
        return AFMFoundationStringStreamUpdate(snapshotUpdate: update, telemetry: telemetry)
    }
}
#endif
