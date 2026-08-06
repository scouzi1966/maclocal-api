import Foundation
import AFMKitCore

public struct AFMMLXRuntimeInfo: Equatable, Sendable {
    public let promptTime: Double
    public let tokensPerSecond: Double

    public init(
        promptTime: Double,
        tokensPerSecond: Double
    ) {
        self.promptTime = promptTime
        self.tokensPerSecond = tokensPerSecond
    }
}

public enum AFMMLXRuntimeEvent: Equatable, Sendable {
    case chunk(String)
    case info(AFMMLXRuntimeInfo)
    case toolCall
    case tokenLogprobs
}

public struct AFMMLXReducedGenerationProgress: Sendable {
    public let accumulatedText: String
    public let reasoningUpdate: AFMReasoningPendingUpdate?

    public init(
        accumulatedText: String,
        reasoningUpdate: AFMReasoningPendingUpdate?
    ) {
        self.accumulatedText = accumulatedText
        self.reasoningUpdate = reasoningUpdate
    }
}

public struct AFMMLXReducedGenerationResult: Sendable {
    public let finalState: AFMReasoningStreamFinalState
    public let finalReasoningUpdate: AFMReasoningPendingUpdate?
    public let tokenCount: Int
    public let stopped: Bool
    public let runtimeInfo: AFMMLXRuntimeInfo?

    public init(
        finalState: AFMReasoningStreamFinalState,
        finalReasoningUpdate: AFMReasoningPendingUpdate?,
        tokenCount: Int,
        stopped: Bool,
        runtimeInfo: AFMMLXRuntimeInfo?
    ) {
        self.finalState = finalState
        self.finalReasoningUpdate = finalReasoningUpdate
        self.tokenCount = tokenCount
        self.stopped = stopped
        self.runtimeInfo = runtimeInfo
    }
}

public enum AFMMLXRuntimePolicy {
    public static let defaultImageProcessingSize = 1024
}
