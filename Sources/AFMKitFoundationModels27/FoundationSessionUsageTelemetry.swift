#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public extension AFMFoundationGenerationTelemetryCalculator {
    static func telemetry(
        usage: LanguageModelSession.Usage,
        toolNames: [String],
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        firstChunkAt: ContinuousClock.Instant?,
        sampledAt: ContinuousClock.Instant,
        streamChunkCount: Int
    ) -> AFMFoundationGenerationTelemetry {
        telemetry(
            inputTokens: usage.input.totalTokenCount,
            cachedInputTokens: usage.input.cachedTokenCount,
            outputTokens: usage.output.totalTokenCount,
            reasoningTokens: usage.output.reasoningTokenCount,
            toolNames: toolNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: streamChunkCount
        )
    }

    static func singleResponseTelemetry(
        usage: LanguageModelSession.Usage,
        toolNames: [String],
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        completedAt: ContinuousClock.Instant
    ) -> AFMFoundationGenerationTelemetry {
        telemetry(
            usage: usage,
            toolNames: toolNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: completedAt,
            sampledAt: completedAt,
            streamChunkCount: 1
        )
    }
}
#endif
