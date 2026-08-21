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
    stoppedBySequence: Bool,
    speculativeTelemetry: AFMMLXSpeculativeTelemetry?
)

public typealias AFMMLXChatStreamingResult = (
    modelID: String,
    stream: AsyncThrowingStream<StreamChunk, Error>,
    promptTokens: Int,
    toolCallStartTag: String?,
    toolCallEndTag: String?,
    thinkStartTag: String?,
    thinkEndTag: String?
)
