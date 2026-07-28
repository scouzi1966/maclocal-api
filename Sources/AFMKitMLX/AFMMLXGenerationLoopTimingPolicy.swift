import Foundation

public struct AFMMLXGenerationLoopTimingSummary: Equatable, Sendable {
    public let tokenCount: Int
    public let totalGenerationTime: TimeInterval
    public let totalLoopOverhead: TimeInterval
    public let averageOverheadMicroseconds: Double
    public let overheadPercentage: Double

    public init(
        tokenCount: Int,
        totalGenerationTime: TimeInterval,
        totalLoopOverhead: TimeInterval,
        averageOverheadMicroseconds: Double,
        overheadPercentage: Double
    ) {
        self.tokenCount = tokenCount
        self.totalGenerationTime = totalGenerationTime
        self.totalLoopOverhead = totalLoopOverhead
        self.averageOverheadMicroseconds = averageOverheadMicroseconds
        self.overheadPercentage = overheadPercentage
    }
}

public struct AFMMLXGenerationLoopTimingState: Equatable, Sendable {
    public let generationStartTime: TimeInterval
    public private(set) var totalLoopOverhead: TimeInterval

    public init(
        generationStartTime: TimeInterval,
        totalLoopOverhead: TimeInterval = 0
    ) {
        self.generationStartTime = generationStartTime
        self.totalLoopOverhead = totalLoopOverhead
    }

    @discardableResult
    public mutating func observeIteration(
        startTime: TimeInterval,
        endTime: TimeInterval
    ) -> TimeInterval {
        let overhead = AFMMLXGenerationLoopTimingPolicy.iterationOverhead(
            startTime: startTime,
            endTime: endTime
        )
        totalLoopOverhead += overhead
        return overhead
    }

    public func summary(
        tokenCount: Int,
        endTime: TimeInterval
    ) -> AFMMLXGenerationLoopTimingSummary? {
        AFMMLXGenerationLoopTimingPolicy.summary(
            tokenCount: tokenCount,
            generationStartTime: generationStartTime,
            endTime: endTime,
            totalLoopOverhead: totalLoopOverhead
        )
    }
}

public enum AFMMLXGenerationLoopTimingPolicy {
    public static func iterationOverhead(
        startTime: TimeInterval,
        endTime: TimeInterval
    ) -> TimeInterval {
        max(0, endTime - startTime)
    }

    public static func summary(
        tokenCount: Int,
        generationStartTime: TimeInterval,
        endTime: TimeInterval,
        totalLoopOverhead: TimeInterval
    ) -> AFMMLXGenerationLoopTimingSummary? {
        guard tokenCount > 0 else { return nil }

        let totalGenerationTime = max(0, endTime - generationStartTime)
        guard totalGenerationTime > 0 else { return nil }

        let normalizedLoopOverhead = max(0, totalLoopOverhead)
        return AFMMLXGenerationLoopTimingSummary(
            tokenCount: tokenCount,
            totalGenerationTime: totalGenerationTime,
            totalLoopOverhead: normalizedLoopOverhead,
            averageOverheadMicroseconds: (normalizedLoopOverhead / Double(tokenCount)) * 1_000_000,
            overheadPercentage: (normalizedLoopOverhead / totalGenerationTime) * 100
        )
    }
}
