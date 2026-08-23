import AFMKitCore
import AFMKitInference
import AFMOpenAICompat

public extension AFMRequest {
    /// Legacy AFM entry point forwarding to the standalone inference adapter.
    init(openAIMessages: [Message], generationConfig: GenerationConfig) throws {
        try self.init(
            openAIMessages: openAIMessages,
            generationConfig: generationConfig.inferenceConfiguration
        )
    }
}
