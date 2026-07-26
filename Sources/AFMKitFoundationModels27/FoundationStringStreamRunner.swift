#if canImport(FoundationModels)
import Foundation

@available(macOS 27.0, *)
public struct AFMFoundationStringStreamRunResult<ProgressState: Equatable>: Equatable {
    public let telemetry: AFMFoundationGenerationTelemetry?
    public let progressState: ProgressState
    public let firstChunkAt: ContinuousClock.Instant?
    public let streamChunkCount: Int

    public init(
        telemetry: AFMFoundationGenerationTelemetry?,
        progressState: ProgressState,
        firstChunkAt: ContinuousClock.Instant?,
        streamChunkCount: Int
    ) {
        self.telemetry = telemetry
        self.progressState = progressState
        self.firstChunkAt = firstChunkAt
        self.streamChunkCount = streamChunkCount
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationStringStreamRunner {
    public static func run<S: AsyncSequence, ProgressState: Equatable>(
        _ snapshots: S,
        initialProgressState: ProgressState,
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        consume: (
            S.Element,
            inout AFMFoundationStringStreamProcessor<ProgressState>,
            String?,
            ContinuousClock.Instant
        ) async throws -> AFMFoundationStringStreamUpdate<ProgressState>,
        receive: (AFMFoundationStringStreamUpdate<ProgressState>) async throws -> Void
    ) async throws -> AFMFoundationStringStreamRunResult<ProgressState> {
        var processor = AFMFoundationStringStreamProcessor<ProgressState>(
            initialProgressState: initialProgressState
        )
        var progressState = initialProgressState
        var telemetry: AFMFoundationGenerationTelemetry?

        for try await snapshot in snapshots {
            try Task.checkCancellation()
            let sampledAt = ContinuousClock.now
            let update = try await consume(snapshot, &processor, contextAction, sampledAt)
            progressState = update.snapshotUpdate.progressState
            telemetry = update.telemetry
            try await receive(update)
        }

        let completedAt = ContinuousClock.now
        return AFMFoundationStringStreamRunResult(
            telemetry: telemetry.map {
                processor.finalize(
                    $0,
                    startedAt: startedAt,
                    completedAt: completedAt
                )
            },
            progressState: progressState,
            firstChunkAt: processor.firstChunkAt,
            streamChunkCount: processor.streamChunkCount
        )
    }
}
#endif
