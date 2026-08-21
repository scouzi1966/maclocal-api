import AFMOpenAICompat

public typealias AFMMLXChatGenerationResult = (
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

/// Opt-in result wrapper that adds telemetry without changing the legacy tuple API.
public struct AFMMLXChatGenerationResultWithTelemetry: Sendable {
    public let result: AFMMLXChatGenerationResult
    public let speculativeTelemetry: AFMMLXSpeculativeTelemetry?

    public init(
        result: AFMMLXChatGenerationResult,
        speculativeTelemetry: AFMMLXSpeculativeTelemetry? = nil
    ) {
        self.result = result
        self.speculativeTelemetry = speculativeTelemetry
    }

    public init(
        modelID: String,
        content: String,
        promptTokens: Int,
        completionTokens: Int,
        tokenLogprobs: [ResolvedLogprob]?,
        toolCalls: [ResponseToolCall]?,
        cachedTokens: Int,
        promptTime: Double,
        generateTime: Double,
        stoppedBySequence: Bool,
        speculativeTelemetry: AFMMLXSpeculativeTelemetry? = nil
    ) {
        self.init(
            result: (
                modelID: modelID,
                content: content,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                tokenLogprobs: tokenLogprobs,
                toolCalls: toolCalls,
                cachedTokens: cachedTokens,
                promptTime: promptTime,
                generateTime: generateTime,
                stoppedBySequence: stoppedBySequence),
            speculativeTelemetry: speculativeTelemetry)
    }

    public var modelID: String { result.modelID }
    public var content: String { result.content }
    public var promptTokens: Int { result.promptTokens }
    public var completionTokens: Int { result.completionTokens }
    public var tokenLogprobs: [ResolvedLogprob]? { result.tokenLogprobs }
    public var toolCalls: [ResponseToolCall]? { result.toolCalls }
    public var cachedTokens: Int { result.cachedTokens }
    public var promptTime: Double { result.promptTime }
    public var generateTime: Double { result.generateTime }
    public var stoppedBySequence: Bool { result.stoppedBySequence }
}

public typealias AFMMLXChatStreamingResult = (
    modelID: String,
    stream: AsyncThrowingStream<StreamChunk, Error>,
    promptTokens: Int,
    toolCallStartTag: String?,
    toolCallEndTag: String?,
    thinkStartTag: String?,
    thinkEndTag: String?
)
