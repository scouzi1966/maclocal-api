import Foundation

public struct AFMFoundationGenerationUsage: Equatable, Sendable {
    public var inputTokens: Int
    public var cachedInputTokens: Int
    public var outputTokens: Int
    public var reasoningTokens: Int

    public init(
        inputTokens: Int,
        cachedInputTokens: Int,
        outputTokens: Int,
        reasoningTokens: Int
    ) {
        self.inputTokens = inputTokens
        self.cachedInputTokens = cachedInputTokens
        self.outputTokens = outputTokens
        self.reasoningTokens = reasoningTokens
    }
}

public struct AFMFoundationGenerationTelemetry: Equatable, Sendable {
    public var inputTokens: Int
    public var cachedInputTokens: Int
    public var outputTokens: Int
    public var reasoningTokens: Int
    public var toolNames: [String]
    public var contextAction: String?
    public var timeToFirstTokenMilliseconds: Double?
    public var streamingMilliseconds: Double?
    public var totalMilliseconds: Double?
    public var tokensPerSecond: Double?
    public var streamChunkCount: Int

    public init(
        inputTokens: Int,
        cachedInputTokens: Int,
        outputTokens: Int,
        reasoningTokens: Int,
        toolNames: [String],
        contextAction: String? = nil,
        timeToFirstTokenMilliseconds: Double? = nil,
        streamingMilliseconds: Double? = nil,
        totalMilliseconds: Double? = nil,
        tokensPerSecond: Double? = nil,
        streamChunkCount: Int = 0
    ) {
        self.inputTokens = inputTokens
        self.cachedInputTokens = cachedInputTokens
        self.outputTokens = outputTokens
        self.reasoningTokens = reasoningTokens
        self.toolNames = toolNames
        self.contextAction = contextAction
        self.timeToFirstTokenMilliseconds = timeToFirstTokenMilliseconds
        self.streamingMilliseconds = streamingMilliseconds
        self.totalMilliseconds = totalMilliseconds
        self.tokensPerSecond = tokensPerSecond
        self.streamChunkCount = streamChunkCount
    }
}

public enum AFMFoundationGenerationTelemetryCalculator {
    public static func telemetry(
        usage: AFMFoundationGenerationUsage,
        toolNames: [String],
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        firstChunkAt: ContinuousClock.Instant?,
        sampledAt: ContinuousClock.Instant,
        streamChunkCount: Int
    ) -> AFMFoundationGenerationTelemetry {
        telemetry(
            inputTokens: usage.inputTokens,
            cachedInputTokens: usage.cachedInputTokens,
            outputTokens: usage.outputTokens,
            reasoningTokens: usage.reasoningTokens,
            toolNames: toolNames,
            contextAction: contextAction,
            startedAt: startedAt,
            firstChunkAt: firstChunkAt,
            sampledAt: sampledAt,
            streamChunkCount: streamChunkCount
        )
    }

    public static func telemetry(
        inputTokens: Int,
        cachedInputTokens: Int,
        outputTokens: Int,
        reasoningTokens: Int,
        toolNames: [String],
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        firstChunkAt: ContinuousClock.Instant?,
        sampledAt: ContinuousClock.Instant,
        streamChunkCount: Int
    ) -> AFMFoundationGenerationTelemetry {
        let streamingMilliseconds = firstChunkAt.map {
            elapsedMilliseconds(from: $0, to: sampledAt)
        }
        return AFMFoundationGenerationTelemetry(
            inputTokens: inputTokens,
            cachedInputTokens: cachedInputTokens,
            outputTokens: outputTokens,
            reasoningTokens: reasoningTokens,
            toolNames: toolNames,
            contextAction: contextAction,
            timeToFirstTokenMilliseconds: firstChunkAt.map {
                elapsedMilliseconds(from: startedAt, to: $0)
            },
            streamingMilliseconds: streamingMilliseconds,
            totalMilliseconds: elapsedMilliseconds(from: startedAt, to: sampledAt),
            tokensPerSecond: decodeTokensPerSecond(
                outputTokens: outputTokens,
                streamingMilliseconds: streamingMilliseconds,
                streamChunkCount: streamChunkCount
            ),
            streamChunkCount: streamChunkCount
        )
    }

    public static func finalize(
        _ telemetry: AFMFoundationGenerationTelemetry,
        startedAt: ContinuousClock.Instant,
        firstChunkAt: ContinuousClock.Instant?,
        completedAt: ContinuousClock.Instant,
        streamChunkCount: Int
    ) -> AFMFoundationGenerationTelemetry {
        var finalized = telemetry
        finalized.totalMilliseconds = elapsedMilliseconds(from: startedAt, to: completedAt)
        finalized.streamingMilliseconds = firstChunkAt.map {
            elapsedMilliseconds(from: $0, to: completedAt)
        }
        finalized.tokensPerSecond = decodeTokensPerSecond(
            outputTokens: finalized.outputTokens,
            streamingMilliseconds: finalized.streamingMilliseconds,
            streamChunkCount: streamChunkCount
        )
        finalized.streamChunkCount = streamChunkCount
        return finalized
    }

    public static func elapsedMilliseconds(
        from start: ContinuousClock.Instant,
        to end: ContinuousClock.Instant
    ) -> Double {
        let components = start.duration(to: end).components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }

    public static func decodeTokensPerSecond(
        outputTokens: Int,
        streamingMilliseconds: Double?,
        streamChunkCount: Int
    ) -> Double? {
        guard streamChunkCount > 1,
              outputTokens > 0,
              let streamingMilliseconds,
              streamingMilliseconds > 0 else { return nil }
        return Double(outputTokens) / (streamingMilliseconds / 1_000)
    }
}
