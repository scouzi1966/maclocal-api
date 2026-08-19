import Foundation
import AFMOpenAICompat
import MLX

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
        toolChoice: ToolChoice?,
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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMMLXChatStreamingResult

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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?
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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?,
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

    /// Reset and read provider-specific request memory telemetry. Providers
    /// that do not execute through MLX must not initialize MLX merely to serve
    /// an OpenAI-compatible request.
    func resetRequestPeakMemory()
    func currentRequestPeakMemoryGib() -> Double?

    /// Resolve effective response format: per-request format wins, falls back to server default.
    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat?
}

public extension AFMMLXOpenAIChatServing {
    var defaultGuidedJsonSchema: ResponseFormat? { nil }

    func resetRequestPeakMemory() {}

    func currentRequestPeakMemoryGib() -> Double? { nil }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        OpenAIResponseFormatPolicy.effectiveResponseFormat(
            requestFormat: requestFormat,
            serverDefault: defaultGuidedJsonSchema
        )
    }
}

public extension AFMMLXOpenAIChatGenerating {
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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?
    ) async throws -> AFMMLXChatGenerationResult {
        if speculativeDecoding?.requirement?.lowercased() == "required" {
            throw NSError(
                domain: "AFMMLXSpeculativeDecoding",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Provider cannot honor required speculative decoding"])
        }
        return try await generate(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            toolChoice: toolChoice, parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs)
    }

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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMMLXChatStreamingResult {
        if speculativeDecoding?.requirement?.lowercased() == "required" {
            throw NSError(
                domain: "AFMMLXSpeculativeDecoding",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Provider cannot honor required speculative decoding"])
        }
        return try await generateStreaming(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            toolChoice: toolChoice, parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs,
            preserveStructuralTags: preserveStructuralTags, requestId: requestId)
    }

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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMMLXChatGenerationResult {
        try await generate(
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
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
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
            preserveStructuralTags: preserveStructuralTags,
            requestId: requestId
        )
    }

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

extension MLXModelService {
    public func generate(
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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?
    ) async throws -> AFMMLXChatGenerationResult {
        try await generate(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs,
            speculativeDecoding: speculativeDecoding)
    }

    public func generateStreaming(
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
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMMLXChatStreamingResult {
        try await generateStreaming(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs,
            speculativeDecoding: speculativeDecoding,
            preserveStructuralTags: preserveStructuralTags, requestId: requestId)
    }

    public func generate(
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
    ) async throws -> AFMMLXChatGenerationResult {
        try await generate(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs,
            speculativeDecoding: nil)
    }

    public func generateStreaming(
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
    ) async throws -> AFMMLXChatStreamingResult {
        try await generateStreaming(
            model: model, messages: messages, temperature: temperature,
            maxTokens: maxTokens, topP: topP, repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, presencePenalty: presencePenalty, seed: seed,
            logprobs: logprobs, topLogprobs: topLogprobs, tools: tools,
            parallelToolCalls: parallelToolCalls, stop: stop,
            responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs,
            speculativeDecoding: nil,
            preserveStructuralTags: preserveStructuralTags, requestId: requestId)
    }
}

extension MLXModelService: AFMMLXOpenAIChatServing {
    public func resetRequestPeakMemory() {
        GPU.resetPeakMemory()
    }

    public func currentRequestPeakMemoryGib() -> Double? {
        let gib = 1024.0 * 1024.0 * 1024.0
        return (Double(Memory.snapshot().peakMemory) / gib * 10).rounded() / 10
    }
}
