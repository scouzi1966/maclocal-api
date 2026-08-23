import Foundation
import AFMOpenAICompat
import AFMKitFoundationModels

#if compiler(>=6.4)
/// Adapts Apple's Foundation Models service to AFMKit's provider-neutral facade.
@available(macOS 26.0, *)
extension FoundationModelService: AFMLanguageModel {
    public var isAvailable: Bool { true }

    public func respond(to messages: [Message], options: GenerationConfig) async throws -> AFMResponse {
        let text = try await generateResponse(
            for: messages,
            temperature: options.temperature,
            maxTokens: options.maxTokens,
            stop: options.stop
        )
        return AFMResponse(content: text)
    }

    public func streamResponse(
        to messages: [Message],
        options: GenerationConfig
    ) -> AsyncThrowingStream<String, Error> {
        generateNativeStreamingResponse(
            for: messages,
            temperature: options.temperature,
            maxTokens: options.maxTokens,
            stop: options.stop
        )
    }
}
#endif
