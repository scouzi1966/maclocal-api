import Foundation

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

public enum AFMMLXRuntimePolicy {
    public static let defaultImageProcessingSize = 1024
}
