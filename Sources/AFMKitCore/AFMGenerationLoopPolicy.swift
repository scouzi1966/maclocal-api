import Foundation

public enum AFMGenerationLoopPolicy {
    public static let defaultStopCheckInterval = 8
    public static let defaultReasoningUpdateInterval = AFMReasoningStreamPolicy.defaultReasoningUpdateInterval
    public static let defaultDirectUIUpdateInterval: TimeInterval = 0.1

    public static func shouldPollStop(
        tokensSinceLastPoll: Int,
        interval: Int = defaultStopCheckInterval
    ) -> Bool {
        tokensSinceLastPoll >= interval
    }

    public static func shouldFlushReasoningUpdate(
        tokensSinceLastFlush: Int,
        interval: Int = defaultReasoningUpdateInterval
    ) -> Bool {
        AFMReasoningStreamPolicy.shouldFlushReasoningUpdate(
            tokensSinceLastFlush: tokensSinceLastFlush,
            interval: interval
        )
    }

    public static func finishReason(
        localShouldStop: Bool,
        stopRequested: Bool = false,
        tokenCount: Int,
        maxTokens: Int
    ) -> AFMFinishReason {
        if localShouldStop || stopRequested {
            return .cancelled
        }
        if tokenCount >= maxTokens {
            return .length
        }
        return .stop
    }
}

public struct AFMStopPollState: Equatable, Sendable {
    public private(set) var tokensSinceLastPoll: Int
    public private(set) var shouldStop: Bool

    public init(
        tokensSinceLastPoll: Int = 0,
        shouldStop: Bool = false
    ) {
        self.tokensSinceLastPoll = tokensSinceLastPoll
        self.shouldStop = shouldStop
    }

    public mutating func observeToken(
        stopRequested: Bool,
        interval: Int = AFMGenerationLoopPolicy.defaultStopCheckInterval
    ) -> Bool {
        tokensSinceLastPoll += 1
        if AFMGenerationLoopPolicy.shouldPollStop(
            tokensSinceLastPoll: tokensSinceLastPoll,
            interval: interval
        ) {
            shouldStop = stopRequested
            tokensSinceLastPoll = 0
        }
        return shouldStop
    }
}
