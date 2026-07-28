#if canImport(FoundationModels)
import AFMKit

/// Reusable projection from an AFMKit model descriptor into the macOS 27 MLX
/// `LanguageModel` configuration.
public struct AFMMLXFoundationLanguageModelPlan: Hashable, Sendable {
    public let modelID: String
    public let defaultMaximumResponseTokens: Int
    public let enablePrefixCaching: Bool
    public let supportsVision: Bool
    public let supportsReasoning: Bool
    public let supportsToolCalling: Bool
    public let supportsGuidedGeneration: Bool

    public init(
        modelID: String,
        defaultMaximumResponseTokens: Int = 2_048,
        enablePrefixCaching: Bool = true,
        supportsVision: Bool = false,
        supportsReasoning: Bool = false,
        supportsToolCalling: Bool = false,
        supportsGuidedGeneration: Bool = false
    ) {
        self.modelID = modelID
        self.defaultMaximumResponseTokens = defaultMaximumResponseTokens
        self.enablePrefixCaching = enablePrefixCaching
        self.supportsVision = supportsVision
        self.supportsReasoning = supportsReasoning
        self.supportsToolCalling = supportsToolCalling
        self.supportsGuidedGeneration = supportsGuidedGeneration
    }

    public static func make(
        modelID: String,
        descriptor: AFMModelDescriptor,
        defaultMaximumResponseTokens: Int = 2_048,
        enablePrefixCaching: Bool = true
    ) -> AFMMLXFoundationLanguageModelPlan {
        AFMMLXFoundationLanguageModelPlan(
            modelID: modelID,
            defaultMaximumResponseTokens: defaultMaximumResponseTokens,
            enablePrefixCaching: enablePrefixCaching,
            supportsVision: descriptor.capabilities.contains(.vision),
            supportsReasoning: descriptor.capabilities.contains(.reasoning),
            supportsToolCalling: descriptor.capabilities.contains(.toolCalling),
            supportsGuidedGeneration: descriptor.capabilities.contains(.structuredOutput)
        )
    }

    public func acceptsImageInput(_ requestedImageInput: Bool) -> Bool {
        requestedImageInput && supportsVision
    }

    @available(macOS 27.0, *)
    public func languageModel() -> MLXLanguageModel {
        MLXLanguageModel(
            modelID: modelID,
            enablePrefixCaching: enablePrefixCaching,
            defaultMaximumResponseTokens: defaultMaximumResponseTokens,
            supportsVision: supportsVision,
            supportsReasoning: supportsReasoning,
            supportsToolCalling: supportsToolCalling,
            supportsGuidedGeneration: supportsGuidedGeneration
        )
    }
}
#endif
