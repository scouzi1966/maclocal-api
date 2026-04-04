import Foundation

typealias ChatGenerationResult = (
    modelID: String,
    content: String,
    promptTokens: Int,
    completionTokens: Int,
    tokenLogprobs: [ResolvedLogprob]?,
    toolCalls: [ResponseToolCall]?,
    cachedTokens: Int,
    promptTime: Double,
    generateTime: Double,
    stoppedBySequence: Bool
)

typealias ChatStreamingResult = (
    modelID: String,
    stream: AsyncThrowingStream<StreamChunk, Error>,
    promptTokens: Int,
    toolCallStartTag: String?,
    toolCallEndTag: String?,
    thinkStartTag: String?,
    thinkEndTag: String?
)

protocol MLXChatServing {
    var maxConcurrent: Int { get }
    var toolCallParser: String? { get }
    var thinkStartTag: String? { get }
    var thinkEndTag: String? { get }
    var fixToolArgs: Bool { get }
    var enableGrammarConstraints: Bool { get }

    func normalizeModel(_ raw: String) -> String
    func tryReserveSlot() -> Bool
    func waitForSlot(timeout: TimeInterval) async -> Bool
    func releaseSlot()

    func ensureBatchMode(concurrency: Int) async throws
    func releaseBatchReference()
    func cancelBatchSlots(ids: Set<UUID>) async

    func startAPIProfile()
    func stopAPIProfile(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfile
    func stopAPIProfileExtended(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfileExtended

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
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatGenerationResult

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
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        requestId: String?
    ) async throws -> ChatStreamingResult
}

extension MLXChatServing {
    /// Convenience overload without requestId for batch/internal callers.
    func generateStreaming(
        model: String, messages: [Message], temperature: Double?, maxTokens: Int?,
        topP: Double?, repetitionPenalty: Double?, topK: Int?, minP: Double?,
        presencePenalty: Double?, seed: Int?, logprobs: Bool?, topLogprobs: Int?,
        tools: [RequestTool]?, stop: [String]?, responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatStreamingResult {
        try await generateStreaming(
            model: model, messages: messages, temperature: temperature, maxTokens: maxTokens,
            topP: topP, repetitionPenalty: repetitionPenalty, topK: topK, minP: minP,
            presencePenalty: presencePenalty, seed: seed, logprobs: logprobs, topLogprobs: topLogprobs,
            tools: tools, stop: stop, responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs, requestId: nil
        )
    }
}

extension MLXModelService: MLXChatServing {}
