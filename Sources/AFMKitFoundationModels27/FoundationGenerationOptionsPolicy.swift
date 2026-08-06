import Foundation
import FoundationModels

public enum AFMFoundationToolCallingDecision: Equatable, Sendable {
    case disallowed
    case allowed
    case required
}

public struct AFMFoundationGenerationParameters: Equatable, Sendable {
    public let useProviderDefaults: Bool
    public let temperature: Double
    public let topP: Double
    public let maxTokens: Int?

    public init(
        useProviderDefaults: Bool = false,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        maxTokens: Int? = nil
    ) {
        self.useProviderDefaults = useProviderDefaults
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
    }
}

public struct AFMFoundationGenerationOptionPlan: Equatable, Sendable {
    public enum Sampling: Equatable, Sendable {
        case providerDefault
        case greedy
        case random(probabilityThreshold: Double)
    }

    public let sampling: Sampling
    public let temperature: Double?
    public let maximumResponseTokens: Int?
    public let toolCalling: AFMFoundationToolCallingDecision

    public init(
        sampling: Sampling,
        temperature: Double?,
        maximumResponseTokens: Int?,
        toolCalling: AFMFoundationToolCallingDecision
    ) {
        self.sampling = sampling
        self.temperature = temperature
        self.maximumResponseTokens = maximumResponseTokens
        self.toolCalling = toolCalling
    }
}

public enum AFMFoundationReasoningLevel: Equatable, Sendable {
    case automatic
    case light
    case moderate
    case deep
}

public enum AFMFoundationGenerationOptionsPolicy {
    public static func plan(
        from parameters: AFMFoundationGenerationParameters,
        allowsToolCalling: Bool,
        toolsEnabled: Bool,
        requiresToolCalling: Bool
    ) -> AFMFoundationGenerationOptionPlan {
        let toolCalling = toolCallingDecision(
            allowsTools: allowsToolCalling,
            toolsEnabled: toolsEnabled,
            requiresToolCalling: requiresToolCalling
        )
        guard !parameters.useProviderDefaults else {
            return AFMFoundationGenerationOptionPlan(
                sampling: .providerDefault,
                temperature: nil,
                maximumResponseTokens: nil,
                toolCalling: toolCalling
            )
        }

        let sampling: AFMFoundationGenerationOptionPlan.Sampling
        let temperature: Double?
        if parameters.temperature <= 0 {
            sampling = .greedy
            temperature = nil
        } else {
            sampling = .random(probabilityThreshold: min(max(parameters.topP, 0.0), 1.0))
            temperature = parameters.temperature
        }

        return AFMFoundationGenerationOptionPlan(
            sampling: sampling,
            temperature: temperature,
            maximumResponseTokens: parameters.maxTokens,
            toolCalling: toolCalling
        )
    }

    public static func toolCallingDecision(
        allowsTools: Bool,
        toolsEnabled: Bool,
        requiresToolCalling: Bool
    ) -> AFMFoundationToolCallingDecision {
        guard allowsTools else { return .disallowed }
        return toolsEnabled && requiresToolCalling ? .required : .allowed
    }

    @available(macOS 27.0, *)
    public static func generationOptions(from plan: AFMFoundationGenerationOptionPlan) -> GenerationOptions {
        let toolCallingMode = foundationToolCallingMode(from: plan.toolCalling)
        switch plan.sampling {
        case .providerDefault:
            return GenerationOptions(toolCallingMode: toolCallingMode)
        case .greedy:
            return GenerationOptions(
                samplingMode: .greedy,
                temperature: nil,
                maximumResponseTokens: plan.maximumResponseTokens,
                toolCallingMode: toolCallingMode
            )
        case .random(let probabilityThreshold):
            return GenerationOptions(
                samplingMode: .random(probabilityThreshold: probabilityThreshold),
                temperature: plan.temperature,
                maximumResponseTokens: plan.maximumResponseTokens,
                toolCallingMode: toolCallingMode
            )
        }
    }

    @available(macOS 27.0, *)
    public static func foundationToolCallingMode(
        from decision: AFMFoundationToolCallingDecision
    ) -> GenerationOptions.ToolCallingMode {
        switch decision {
        case .disallowed: return .disallowed
        case .allowed: return .allowed
        case .required: return .required
        }
    }

    @available(macOS 27.0, *)
    public static func contextReasoningLevel(
        from level: AFMFoundationReasoningLevel?
    ) -> ContextOptions.ReasoningLevel? {
        switch level {
        case .light: return .light
        case .moderate: return .moderate
        case .deep: return .deep
        case .automatic, .none: return nil
        }
    }
}
