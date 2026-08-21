import Foundation
import AFMKitCore
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

    /// Resolve and validate media against the immutable capability state of the
    /// active container. Remote media is returned as a bounded canonical data URL
    /// so generation never performs a second network fetch.
    func preflightMediaRequest(
        model: String,
        messages: [Message]
    ) async throws -> AFMMLXResolvedMediaRequest

    /// Run one generation operation with a provider-issued media preflight
    /// result. Implementations may use this scope to avoid inspecting the same
    /// canonical payload again; callers cannot construct the token themselves.
    func withPreflightedMediaRequest<Result: Sendable>(
        _ request: AFMMLXResolvedMediaRequest,
        operation: ([Message]) async throws -> Result
    ) async throws -> Result

    /// Runtime-usable descriptor for the active model, if this service owns one.
    func loadedModelDescriptor(model: String) -> AFMModelDescriptor?
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

    func preflightMediaRequest(
        model: String,
        messages: [Message]
    ) async throws -> AFMMLXResolvedMediaRequest {
        do {
            return try await AFMMLXMediaSecurityPolicy.resolveRequest(in: messages)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw MLXServiceError.invalidMediaInput(error.localizedDescription)
        }
    }

    func withPreflightedMediaRequest<Result: Sendable>(
        _ request: AFMMLXResolvedMediaRequest,
        operation: ([Message]) async throws -> Result
    ) async throws -> Result {
        try await operation(request.messages)
    }

    func loadedModelDescriptor(model: String) -> AFMModelDescriptor? { nil }
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

extension MLXModelService: AFMMLXOpenAIChatServing {
    public func resetRequestPeakMemory() {
        GPU.resetPeakMemory()
    }

    public func currentRequestPeakMemoryGib() -> Double? {
        let gib = 1024.0 * 1024.0 * 1024.0
        return (Double(Memory.snapshot().peakMemory) / gib * 10).rounded() / 10
    }
}
