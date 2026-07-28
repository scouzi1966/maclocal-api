import Foundation
import AFMKitCore

public struct AFMMLXGenerationCompletionSummary: Equatable, Sendable {
    public let tokenCount: Int
    public let maxTokens: Int
    public let finishReason: AFMFinishReason
    public let reachedMaxTokens: Bool
    public let finalContent: String
    public let historyText: String
    public let hasReasoning: Bool

    public init(
        tokenCount: Int,
        maxTokens: Int,
        finishReason: AFMFinishReason,
        reachedMaxTokens: Bool,
        finalContent: String,
        historyText: String,
        hasReasoning: Bool
    ) {
        self.tokenCount = tokenCount
        self.maxTokens = maxTokens
        self.finishReason = finishReason
        self.reachedMaxTokens = reachedMaxTokens
        self.finalContent = finalContent
        self.historyText = historyText
        self.hasReasoning = hasReasoning
    }
}

public enum AFMMLXGenerationCompletionPolicy {
    public static func summary(
        finalState: AFMReasoningStreamFinalState,
        tokenCount: Int,
        localShouldStop: Bool,
        stopRequested: Bool,
        maxTokens: Int
    ) -> AFMMLXGenerationCompletionSummary {
        return AFMMLXGenerationCompletionSummary(
            tokenCount: tokenCount,
            maxTokens: maxTokens,
            finishReason: AFMGenerationLoopPolicy.finishReason(
                localShouldStop: localShouldStop,
                stopRequested: stopRequested,
                tokenCount: tokenCount,
                maxTokens: maxTokens
            ),
            reachedMaxTokens: tokenCount >= maxTokens,
            finalContent: finalState.finalization.finalContent,
            historyText: finalState.finalization.historyText,
            hasReasoning: finalState.finalization.hasReasoning
        )
    }
}
