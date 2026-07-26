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
        content: String,
        usage: AFMFoundationGenerationUsage,
        progressState: ProgressState,
        progressNames: [String],
        reasoningContent: String = "",
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        sampledAt: ContinuousClock.Instant = ContinuousClock.now
    ) -> AFMFoundationStringStreamUpdate<ProgressState> {
        let update = accumulator.consume(
            content: content,
            progressState: progressState,
            reasoningContent: reasoningContent
        )
        if update.firstChunkStarted, firstChunkAt == nil {
            firstChunkAt = sampledAt
        }
        let telemetry = AFMFoundationGenerationTelemetryCalculator.telemetry(
            usage: usage,
            toolNames: progressNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: update.streamChunkCount
        )
        return AFMFoundationStringStreamUpdate(snapshotUpdate: update, telemetry: telemetry)
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
        consume(
            content: snapshot.content,
            usage: AFMFoundationGenerationTelemetryCalculator.usage(from: snapshot.usage),
            progressState: progressState,
            progressNames: progressNames,
            reasoningContent: reasoningContent,
            contextAction: contextAction,
            startedAt: startedAt,
            sampledAt: sampledAt
        )
    }

    public func finalize(
        _ telemetry: AFMFoundationGenerationTelemetry,
        startedAt: ContinuousClock.Instant,
        completedAt: ContinuousClock.Instant
    ) -> AFMFoundationGenerationTelemetry {
        AFMFoundationGenerationTelemetryCalculator.finalize(
            telemetry,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            completedAt: completedAt,
            streamChunkCount: streamChunkCount
        )
    }
}
#endif
