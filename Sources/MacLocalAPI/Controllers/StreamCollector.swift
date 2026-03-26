import Foundation

/// Result of collecting a `ChatStreamingResult` stream with post-processing applied.
struct CollectedResult: Sendable {
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ResponseToolCall]?
    let logprobs: [ResolvedLogprob]
    let promptTokens: Int
    let completionTokens: Int
    let cachedTokens: Int
    let promptTime: Double
    let generateTime: Double
    let stoppedBySequence: Bool
    let finishReason: String
}

/// Utility that collects a ChatStreamingResult stream into a finalized result
/// with think extraction, logprobs, and tool call handling applied.
///
/// Used by BatchAPIController and BatchCompletionsController to avoid
/// duplicating the post-processing logic from MLXChatCompletionsController.
enum StreamCollector {

    /// Collect all chunks from a streaming result into a single finalized result.
    ///
    /// - Parameters:
    ///   - streamResult: The streaming result from `generateStreaming()`
    ///   - extractThinking: Whether to extract `<think>` tags into `reasoningContent`
    ///   - thinkStartTag: The start tag for think blocks (default: `<think>`)
    ///   - thinkEndTag: The end tag for think blocks (default: `</think>`)
    ///   - maxTokens: Maximum tokens for finish_reason determination
    /// - Returns: A `CollectedResult` with all post-processing applied
    static func collect(
        from streamResult: ChatStreamingResult,
        extractThinking: Bool,
        thinkStartTag: String = "<think>",
        thinkEndTag: String = "</think>",
        maxTokens: Int = Int.max
    ) async throws -> CollectedResult {
        var fullText = ""
        var allLogprobs: [ResolvedLogprob] = []
        var toolCalls: [ResponseToolCall]? = nil
        var promptTokens = streamResult.promptTokens
        var completionTokens = 0
        var cachedTokens = 0
        var promptTime: Double = 0
        var generateTime: Double = 0
        var stoppedBySequence = false

        for try await chunk in streamResult.stream {
            fullText += chunk.text
            if let lp = chunk.logprobs { allLogprobs.append(contentsOf: lp) }
            if let tc = chunk.toolCalls { toolCalls = tc }
            if let pt = chunk.promptTokens { promptTokens = pt }
            if let ct = chunk.completionTokens { completionTokens = ct }
            if let cached = chunk.cachedTokens { cachedTokens = cached }
            if let pt = chunk.promptTime { promptTime = pt }
            if let gt = chunk.generateTime { generateTime = gt }
            if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
        }

        // Determine finish reason
        let finishReason: String
        if let tc = toolCalls, !tc.isEmpty {
            finishReason = "tool_calls"
        } else if stoppedBySequence {
            finishReason = "stop"
        } else if completionTokens >= maxTokens {
            finishReason = "length"
        } else {
            finishReason = "stop"
        }

        // Extract think tags if applicable
        let content: String?
        let reasoningContent: String?

        if let tc = toolCalls, !tc.isEmpty {
            // Tool call response: no content, no reasoning extraction needed
            content = nil
            reasoningContent = nil
        } else if extractThinking {
            let (extracted, reasoning) = MLXChatCompletionsController.extractThinkContent(
                from: fullText,
                startTag: thinkStartTag,
                endTag: thinkEndTag
            )
            content = extracted.isEmpty ? nil : extracted
            reasoningContent = reasoning
        } else {
            content = fullText.isEmpty ? nil : fullText
            reasoningContent = nil
        }

        return CollectedResult(
            content: content,
            reasoningContent: reasoningContent,
            toolCalls: toolCalls,
            logprobs: allLogprobs,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            cachedTokens: cachedTokens,
            promptTime: promptTime,
            generateTime: generateTime,
            stoppedBySequence: stoppedBySequence,
            finishReason: finishReason
        )
    }
}
