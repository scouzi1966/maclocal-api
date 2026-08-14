import Foundation
import AFMKit
import AFMKitCore
import AFMKitMLX

/// Result of collecting a `AFMMLXChatStreamingResult` stream with post-processing applied.
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

/// Utility that collects an AFMMLXChatStreamingResult stream into a finalized result
/// with think extraction, logprobs, and tool call handling applied.
///
/// Used by BatchAPIController and BatchCompletionsController to avoid
/// duplicating the post-processing logic from MLXChatCompletionsController.
enum StreamCollector {
    static func buildChoiceLogprobs(_ resolved: [ResolvedLogprob]) -> ChoiceLogprobs? {
        guard !resolved.isEmpty else { return nil }
        let content = resolved.map { entry in
            let topEntries = entry.topTokens.map { top in
                TopLogprobEntry(
                    token: top.token,
                    logprob: Double(top.logprob),
                    bytes: Array(top.token.utf8).map { Int($0) }
                )
            }
            return TokenLogprobContent(
                token: entry.token,
                logprob: Double(entry.logprob),
                bytes: Array(entry.token.utf8).map { Int($0) },
                topLogprobs: topEntries
            )
        }
        return ChoiceLogprobs(content: content)
    }

    static func extractThinkContent(
        from text: String,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (content: String, reasoning: String?) {
        AFMReasoningOutputExtractor.extractThinkContent(
            from: text,
            startTag: startTag,
            endTag: endTag
        )
    }

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
        from streamResult: AFMMLXChatStreamingResult,
        extractThinking: Bool,
        thinkStartTag: String = "<think>",
        thinkEndTag: String = "</think>",
        maxTokens: Int = Int.max,
        responseChannelFormat: AFMMLXResponseChannelFormat = .none
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
        } else {
            let parsed = extractResponseChannels(
                from: fullText,
                responseChannelFormat: responseChannelFormat,
                extractThinking: extractThinking,
                thinkStartTag: thinkStartTag,
                thinkEndTag: thinkEndTag
            )
            content = parsed.content?.isEmpty == true ? nil : parsed.content
            reasoningContent = parsed.reasoning
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

    static func extractResponseChannels(
        from fullText: String,
        responseChannelFormat: AFMMLXResponseChannelFormat,
        extractThinking: Bool,
        thinkStartTag: String = "<think>",
        thinkEndTag: String = "</think>"
    ) -> (content: String?, reasoning: String?) {
        switch responseChannelFormat {
        case .harmony:
            let extracted = MLXChatCompletionsController.extractHarmonyContent(from: fullText)
            return (
                content: extracted.content.isEmpty ? nil : extracted.content,
                reasoning: extracted.reasoning
            )
        case .muse:
            let extracted = MLXChatCompletionsController.extractMuseResponseContent(from: fullText)
            return (
                content: extracted.content.isEmpty ? nil : extracted.content,
                reasoning: extracted.reasoning
            )
        case .none:
            break
        }

        if extractThinking {
            let (extracted, reasoning) = extractThinkContent(
                from: fullText,
                startTag: thinkStartTag,
                endTag: thinkEndTag
            )
            return (
                content: extracted.isEmpty ? nil : extracted,
                reasoning: reasoning
            )
        }
        return (content: fullText.isEmpty ? nil : fullText, reasoning: nil)
    }
}
