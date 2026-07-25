import Foundation
import AFMOpenAICompat

public protocol AFMMLXOpenAIChatGenerating: Sendable {
    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMMLXChatGenerationResult

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMMLXChatStreamingResult
}

public protocol AFMMLXOpenAIChatServing:
    AFMMLXAPIProfiling,
    AFMMLXRequestScheduling,
    AFMMLXBatchControlling,
    AFMMLXServingConfigurationProviding,
    AFMMLXOpenAIChatGenerating
{
    var defaultGuidedJsonSchema: ResponseFormat? { get }

    /// Resolve effective response format: per-request format wins, falls back to server default.
    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat?
}

public extension AFMMLXOpenAIChatServing {
    var defaultGuidedJsonSchema: ResponseFormat? { nil }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        requestFormat ?? defaultGuidedJsonSchema
    }
}

public extension AFMMLXOpenAIChatGenerating {
    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMMLXChatStreamingResult {
        try await generateStreaming(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs,
            preserveStructuralTags: false,
            requestId: nil
        )
    }
}

extension MLXModelService: AFMMLXOpenAIChatServing {}
