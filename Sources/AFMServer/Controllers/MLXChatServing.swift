import Foundation
import AFMKit
import AFMOpenAICompat
import MLXLMCommon

typealias ChatGenerationResult = AFMMLXChatGenerationResult
typealias ChatStreamingResult = AFMMLXChatStreamingResult

protocol MLXChatServing:
    AFMMLXAPIProfiling,
    AFMMLXRequestScheduling,
    AFMMLXBatchControlling,
    AFMMLXServingConfigurationProviding
{
    var defaultGuidedJsonSchema: ResponseFormat? { get }

    /// Resolve effective response format: per-request format wins, falls back to server default.
    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat?

    func generate(
        model: String,
        messages: [AFMOpenAICompat.Message],
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
    ) async throws -> ChatGenerationResult

    func generateStreaming(
        model: String,
        messages: [AFMOpenAICompat.Message],
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
    ) async throws -> ChatStreamingResult
}

extension MLXChatServing {
    var defaultGuidedJsonSchema: ResponseFormat? { nil }
    var toolCallParser: String? { servingConfiguration.toolCallParser }
    var supportsStrictToolGrammar: Bool { servingConfiguration.supportsStrictToolGrammar }
    var thinkStartTag: String? { servingConfiguration.thinkStartTag }
    var thinkEndTag: String? { servingConfiguration.thinkEndTag }
    var harmonyChannels: Bool { servingConfiguration.harmonyChannels }
    /// Structural wrapper tokens to strip from extracted output (e.g. Cohere
    /// `<|START_TEXT|>`/`<|END_TEXT|>`). Empty for most models. (#148)
    var structuralStripTags: [String] { servingConfiguration.structuralStripTags }
    var fixToolArgs: Bool { servingConfiguration.fixToolArguments }
    var enableGrammarConstraints: Bool { servingConfiguration.grammarConstraintsEnabled }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        requestFormat ?? defaultGuidedJsonSchema
    }

    func shouldDowngradeGrammarConstraints(
        responseFormat: ResponseFormat?,
        tools: [RequestTool]?
    ) -> Bool {
        AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
            responseFormat: responseFormat,
            tools: tools,
            supportsStrictToolGrammar: supportsStrictToolGrammar,
            enableGrammarConstraints: enableGrammarConstraints
        )
    }

    func isToolCallParserDisabled(_ parser: String?) -> Bool {
        AFMMLXToolCallPolicy.isToolCallParserDisabled(parser)
    }

    func normalizeToolCalls(
        _ toolCalls: [ToolCall],
        startIndex: Int = 0,
        paramNameMapping: [String: String] = [:],
        tools: [RequestTool]? = nil
    ) -> [ResponseToolCall] {
        AFMMLXToolCallPolicy.normalizeToolCalls(
            toolCalls,
            startIndex: startIndex,
            paramNameMapping: paramNameMapping,
            tools: tools,
            fixToolArgs: fixToolArgs
        )
    }

    func coerceToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        AFMMLXToolCallPolicy.coerceArgumentTypes(toolCall, tools: tools)
    }

    func remapArgumentKeys(
        _ arguments: [String: any Sendable],
        toolName: String,
        tools: [RequestTool]?
    ) -> [String: any Sendable] {
        guard fixToolArgs else { return arguments }
        return AFMMLXToolCallPolicy.remapArgumentKeys(arguments, toolName: toolName, tools: tools)
    }

    func remapToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        guard fixToolArgs else { return toolCall }
        return AFMMLXToolCallPolicy.remapResponseToolCallArguments(toolCall, tools: tools)
    }

    /// Convenience overload without requestId for batch/internal callers.
    func generateStreaming(
        model: String, messages: [AFMOpenAICompat.Message], temperature: Double?, maxTokens: Int?,
        topP: Double?, repetitionPenalty: Double?, topK: Int?, minP: Double?,
        presencePenalty: Double?, seed: Int?, logprobs: Bool?, topLogprobs: Int?,
        tools: [RequestTool]?, parallelToolCalls: Bool?, stop: [String]?, responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatStreamingResult {
        try await generateStreaming(
            model: model, messages: messages, temperature: temperature, maxTokens: maxTokens,
            topP: topP, repetitionPenalty: repetitionPenalty, topK: topK, minP: minP,
            presencePenalty: presencePenalty, seed: seed, logprobs: logprobs, topLogprobs: topLogprobs,
            tools: tools, parallelToolCalls: parallelToolCalls, stop: stop, responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs, preserveStructuralTags: false, requestId: nil
        )
    }
}

extension MLXModelService: MLXChatServing {}
