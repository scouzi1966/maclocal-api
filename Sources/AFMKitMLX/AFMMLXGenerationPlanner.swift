import Foundation

public struct AFMMLXGenerationHiddenOverrides: Equatable, Sendable {
    public let maxKVSize: Int?
    public let kvBits: Int?
    public let kvGroupSize: Int?
    public let quantizedKVStart: Int?
    public let prefillStepSize: Int?
    public let repetitionContextSize: Int?

    public init(
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int? = nil,
        quantizedKVStart: Int? = nil,
        prefillStepSize: Int? = nil,
        repetitionContextSize: Int? = nil
    ) {
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.prefillStepSize = prefillStepSize
        self.repetitionContextSize = repetitionContextSize
    }
}

public enum AFMMLXThinkingContext: Equatable, Sendable {
    case enableThinking(Bool)

    public var additionalContext: [String: any Sendable] {
        switch self {
        case .enableThinking(let enabled):
            return ["enable_thinking": enabled]
        }
    }
}

public struct AFMMLXGenerationPlan: Equatable, Sendable {
    public let hasReasoningOutput: Bool
    public let thinkingContext: AFMMLXThinkingContext?
    public let parameters: AFMMLXGenerationParameterRequest

    public init(
        hasReasoningOutput: Bool,
        thinkingContext: AFMMLXThinkingContext?,
        parameters: AFMMLXGenerationParameterRequest
    ) {
        self.hasReasoningOutput = hasReasoningOutput
        self.thinkingContext = thinkingContext
        self.parameters = parameters
    }

    public var additionalContext: [String: any Sendable]? {
        thinkingContext?.additionalContext
    }
}

public enum AFMMLXGenerationPlanner {
    public static func make(
        maxTokens: Int?,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        topK: Int = 0,
        minP: Double = 0.0,
        presencePenalty: Double = 0.0,
        fallbackPrefillStepSize: Int,
        hiddenOverrides: AFMMLXGenerationHiddenOverrides,
        supportsThinkingToggle: Bool,
        enableThinking: Bool,
        modelHasImplicitReasoning: Bool
    ) -> AFMMLXGenerationPlan {
        let hasReasoningOutput = supportsThinkingToggle && !enableThinking
            ? false
            : modelHasImplicitReasoning
        let thinkingContext: AFMMLXThinkingContext? = supportsThinkingToggle
            ? .enableThinking(enableThinking)
            : nil

        return AFMMLXGenerationPlan(
            hasReasoningOutput: hasReasoningOutput,
            thinkingContext: thinkingContext,
            parameters: AFMMLXGenerationParameterRequest(
                maxTokens: maxTokens,
                maxKVSize: hiddenOverrides.maxKVSize,
                kvBits: hiddenOverrides.kvBits,
                kvGroupSize: hiddenOverrides.kvGroupSize ?? 64,
                quantizedKVStart: hiddenOverrides.quantizedKVStart ?? 0,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: hiddenOverrides.repetitionContextSize ?? 64,
                topK: topK,
                minP: minP,
                presencePenalty: presencePenalty,
                prefillStepSize: hiddenOverrides.prefillStepSize ?? fallbackPrefillStepSize
            )
        )
    }
}
