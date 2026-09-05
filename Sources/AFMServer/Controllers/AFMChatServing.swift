import Foundation
import AFMKitCore
import AFMOpenAICompat

/// Server-owned response channel modes used while rendering OpenAI responses.
/// Provider packages emit typed AFMKit events and do not expose their parser
/// or scheduler implementation through this HTTP boundary.
enum AFMResponseChannelFormat: String, Sendable, Equatable {
    case none
    case harmony
    case muse
}

struct AFMChatServingConfiguration: Sendable, Equatable {
    var toolCallParser: String?
    var supportsStrictToolGrammar: Bool
    var thinkStartTag: String?
    var thinkEndTag: String?
    var harmonyChannels: Bool
    var responseChannelFormat: AFMResponseChannelFormat
    var structuralStripTags: [String]
    var fixToolArguments: Bool
    var grammarConstraintsEnabled: Bool

    init(
        toolCallParser: String? = nil,
        supportsStrictToolGrammar: Bool = false,
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        harmonyChannels: Bool = false,
        responseChannelFormat: AFMResponseChannelFormat = .none,
        structuralStripTags: [String] = [],
        fixToolArguments: Bool = false,
        grammarConstraintsEnabled: Bool = false
    ) {
        self.toolCallParser = toolCallParser
        self.supportsStrictToolGrammar = supportsStrictToolGrammar
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.harmonyChannels = harmonyChannels
        self.responseChannelFormat = harmonyChannels ? .harmony : responseChannelFormat
        self.structuralStripTags = structuralStripTags
        self.fixToolArguments = fixToolArguments
        self.grammarConstraintsEnabled = grammarConstraintsEnabled
    }
}

/// Server-local logprob representation used to render OpenAI responses without
/// exposing provider-specific token/logprob structs through the HTTP layer.
struct AFMServerResolvedLogprob: Sendable {
    let token: String
    let tokenId: Int
    let logprob: Float
    let topTokens: [(token: String, tokenId: Int, logprob: Float)]

    init(
        token: String,
        tokenId: Int,
        logprob: Float,
        topTokens: [(token: String, tokenId: Int, logprob: Float)] = []
    ) {
        self.token = token
        self.tokenId = tokenId
        self.logprob = logprob
        self.topTokens = topTokens
    }
}

/// Streaming chunk consumed by AFMServer's OpenAI-compatible controllers.
/// AFMKit providers are adapted into this shape at the server boundary.
struct AFMServerStreamChunk: Sendable {
    let text: String
    let logprobs: [AFMServerResolvedLogprob]?
    /// Completed per-call snapshots; consumers accumulate across chunks and
    /// replace repeated indices. Argument fragments live in toolCallDeltas.
    let toolCalls: [ResponseToolCall]?
    let toolCallDeltas: [StreamDeltaToolCall]?
    let promptTokens: Int?
    let completionTokens: Int?
    let cachedTokens: Int?
    let promptTime: Double?
    let generateTime: Double?
    let stoppedBySequence: Bool?

    init(
        text: String,
        logprobs: [AFMServerResolvedLogprob]? = nil,
        toolCalls: [ResponseToolCall]? = nil,
        toolCallDeltas: [StreamDeltaToolCall]? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        cachedTokens: Int? = nil,
        promptTime: Double? = nil,
        generateTime: Double? = nil,
        stoppedBySequence: Bool? = nil
    ) {
        self.text = text
        self.logprobs = logprobs
        self.toolCalls = toolCalls
        self.toolCallDeltas = toolCallDeltas
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.promptTime = promptTime
        self.generateTime = generateTime
        self.stoppedBySequence = stoppedBySequence
    }
}

typealias AFMChatGenerationResult = (
    modelID: String,
    content: String,
    promptTokens: Int,
    completionTokens: Int,
    tokenLogprobs: [AFMServerResolvedLogprob]?,
    toolCalls: [ResponseToolCall]?,
    cachedTokens: Int,
    promptTime: Double,
    generateTime: Double,
    stoppedBySequence: Bool
)

typealias AFMChatStreamingResult = (
    modelID: String,
    stream: AsyncThrowingStream<AFMServerStreamChunk, Error>,
    promptTokens: Int,
    toolCallStartTag: String?,
    toolCallEndTag: String?,
    thinkStartTag: String?,
    thinkEndTag: String?
)

/// OpenAI transport contract owned by AFMServer.
///
/// AFMKit providers are adapted to this protocol through `AnyAFMModel` and
/// typed `AFMGenerationEvent` values. Engine-specific serving protocols are
/// intentionally not part of this API.
protocol AFMChatServing: Sendable {
    var maxConcurrent: Int { get }
    /// Whether a successfully returned streaming response owns and releases
    /// the slot reserved by the caller. Controllers use this explicit contract
    /// instead of inferring ownership from the configured capacity.
    var generatedStreamOwnsSlotReservation: Bool { get }
    var servingConfiguration: AFMChatServingConfiguration { get }
    var defaultGuidedJsonSchema: ResponseFormat? { get }

    func normalizeModel(_ raw: String) -> String
    func loadedModelDescriptor(model: String) -> AFMModelDescriptor?
    func resolvedToolCallParser(logBypass: Bool) -> String?

    func tryReserveSlot() -> Bool
    func waitForSlot(timeout: TimeInterval) async -> Bool
    func releaseSlot()

    func ensureBatchMode(concurrency: Int) async throws
    func releaseBatchReference()
    func cancelBatchSlots(ids: Set<UUID>) async

    func startAPIProfile()
    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile
    func stopAPIProfileExtended(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfileExtended

    func resetRequestPeakMemory()
    func currentRequestPeakMemoryGib() -> Double?
    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat?

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
    ) async throws -> AFMChatGenerationResult

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
    ) async throws -> AFMChatStreamingResult

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
    ) async throws -> AFMChatGenerationResult

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
    ) async throws -> AFMChatStreamingResult
}

extension AFMChatServing {
    var toolCallParser: String? { servingConfiguration.toolCallParser }
    var supportsStrictToolGrammar: Bool { servingConfiguration.supportsStrictToolGrammar }
    var thinkStartTag: String? { servingConfiguration.thinkStartTag }
    var thinkEndTag: String? { servingConfiguration.thinkEndTag }
    var harmonyChannels: Bool { servingConfiguration.harmonyChannels }
    var responseChannelFormat: AFMResponseChannelFormat { servingConfiguration.responseChannelFormat }
    var structuralStripTags: [String] { servingConfiguration.structuralStripTags }
    var fixToolArgs: Bool { servingConfiguration.fixToolArguments }
    var enableGrammarConstraints: Bool { servingConfiguration.grammarConstraintsEnabled }
    var defaultGuidedJsonSchema: ResponseFormat? { nil }

    func loadedModelDescriptor(model: String) -> AFMModelDescriptor? { nil }

    func resetRequestPeakMemory() {}
    func currentRequestPeakMemoryGib() -> Double? { nil }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        requestFormat ?? defaultGuidedJsonSchema
    }

    func waitForSlot(timeout: TimeInterval) async -> Bool {
        if Task.isCancelled { return false }
        if timeout <= 0 { return tryReserveSlot() }
        if tryReserveSlot() { return true }

        let deadline = ContinuousClock.now + .seconds(timeout)
        var delay: UInt64 = 10_000_000
        while ContinuousClock.now < deadline {
            if Task.isCancelled { return false }
            try? await Task.sleep(nanoseconds: delay)
            if tryReserveSlot() { return true }
            delay = min(delay * 2, 500_000_000)
        }
        return false
    }

    func shouldDowngradeGrammarConstraints(
        responseFormat: ResponseFormat?,
        tools: [RequestTool]?
    ) -> Bool {
        let strictSchema = responseFormat?.type == "json_schema"
            && responseFormat?.jsonSchema?.strict == true
        let strictTools = supportsStrictToolGrammar
            && (tools?.contains { $0.function.strict == true } ?? false)
        return (strictSchema || strictTools) && !enableGrammarConstraints
    }

    func isToolCallParserDisabled(_ parser: String?) -> Bool {
        parser?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "none"
    }

    func coerceToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        toolCall
    }

    func remapArgumentKeys(
        _ arguments: [String: any Sendable],
        toolName: String,
        tools: [RequestTool]?
    ) -> [String: any Sendable] {
        arguments
    }

    func remapToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        toolCall
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
    ) async throws -> AFMChatGenerationResult {
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
    ) async throws -> AFMChatStreamingResult {
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
}
