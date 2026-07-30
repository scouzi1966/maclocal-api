#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationStringStreamUpdate<ProgressState: Equatable & Sendable>: Equatable, Sendable {
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
public struct AFMFoundationStringStreamProcessor<ProgressState: Equatable & Sendable>: Sendable {
    private var accumulator: AFMFoundationSnapshotAccumulator<ProgressState>
    private var nativeTokensPerSecond: Double?
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
        nativeTokensPerSecond: Double? = nil,
        startedAt: ContinuousClock.Instant,
        sampledAt: ContinuousClock.Instant = ContinuousClock.now
    ) -> AFMFoundationStringStreamUpdate<ProgressState> {
        if let nativeTokensPerSecond, nativeTokensPerSecond.isFinite, nativeTokensPerSecond > 0 {
            self.nativeTokensPerSecond = nativeTokensPerSecond
        }
        let update = accumulator.consume(
            content: content,
            progressState: progressState,
            reasoningContent: reasoningContent
        )
        if update.firstChunkStarted, firstChunkAt == nil {
            firstChunkAt = sampledAt
        }
        var telemetry = AFMFoundationGenerationTelemetryCalculator.telemetry(
            usage: usage,
            toolNames: progressNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: update.streamChunkCount
        )
        telemetry.tokensPerSecond = self.nativeTokensPerSecond ?? telemetry.tokensPerSecond
        return AFMFoundationStringStreamUpdate(snapshotUpdate: update, telemetry: telemetry)
    }

    public mutating func consume(
        snapshot: LanguageModelSession.ResponseStream<String>.Snapshot,
        progressState: ProgressState,
        progressNames: [String],
        reasoningContent: String = "",
        contextAction: String?,
        nativeTokensPerSecond: Double? = nil,
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
            nativeTokensPerSecond: nativeTokensPerSecond,
            startedAt: startedAt,
            sampledAt: sampledAt
        )
    }

    public func finalize(
        _ telemetry: AFMFoundationGenerationTelemetry,
        startedAt: ContinuousClock.Instant,
        completedAt: ContinuousClock.Instant
    ) -> AFMFoundationGenerationTelemetry {
        var finalized = AFMFoundationGenerationTelemetryCalculator.finalize(
            telemetry,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            completedAt: completedAt,
            streamChunkCount: streamChunkCount
        )
        finalized.tokensPerSecond = nativeTokensPerSecond ?? finalized.tokensPerSecond
        return finalized
    }
}
#endif
