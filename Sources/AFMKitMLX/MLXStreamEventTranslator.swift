import AFMKitCore
import AFMOpenAICompat
import Foundation

public struct MLXStreamEventTranslator {
    private struct ToolState {
        var id: String
        var name: String
        var arguments = ""
        var completed = false

        var call: AFMToolCall {
            AFMToolCall(id: id, name: name, arguments: arguments)
        }
    }

    private let thinkStartTag: String?
    private let thinkEndTag: String?
    private let maximumResponseTokens: Int?
    private let requestTools: [RequestTool]?
    private var textBuffer = ""
    private var bufferedTokenCount = 0
    private var insideReasoning = false
    private var tools: [Int: ToolState] = [:]
    private var cachedInputTokens = 0
    private var completionTokens = 0
    private var stoppedBySequence = false

    public init(
        thinkStartTag: String?,
        thinkEndTag: String?,
        maximumResponseTokens: Int?,
        tools: [RequestTool]? = nil
    ) {
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.maximumResponseTokens = maximumResponseTokens
        self.requestTools = tools
    }

    public mutating func consume(_ chunk: StreamChunk) -> [AFMGenerationEvent] {
        var events = textEvents(from: chunk)
        if let logprobs = chunk.logprobs, !logprobs.isEmpty {
            events.append(
                .tokenLogprobs(
                    logprobs.map {
                        AFMTokenLogProbability(
                            token: $0.token,
                            tokenID: $0.tokenId,
                            logprob: $0.logprob,
                            topTokens: $0.topTokens.map {
                                AFMTopLogProbability(
                                    token: $0.token,
                                    tokenID: $0.tokenId,
                                    logprob: $0.logprob
                                )
                            }
                        )
                    }
                )
            )
        }
        events.append(contentsOf: toolEvents(from: chunk.toolCallDeltas ?? []))
        events.append(contentsOf: completedToolEvents(from: chunk.toolCalls ?? []))

        if let cachedTokens = chunk.cachedTokens {
            cachedInputTokens = cachedTokens
        }
        if let promptTokens = chunk.promptTokens,
           let completionTokens = chunk.completionTokens {
            self.completionTokens = completionTokens
            events.append(
                .usage(
                    AFMUsage(
                        inputTokens: promptTokens,
                        cachedInputTokens: cachedInputTokens,
                        outputTokens: completionTokens
                    )
                )
            )
        }
        if chunk.stoppedBySequence == true {
            stoppedBySequence = true
        }
        var metadata: [String: AFMJSONValue] = [:]
        if let promptTime = chunk.promptTime {
            metadata["promptTime"] = .number(promptTime)
        }
        if let generateTime = chunk.generateTime {
            metadata["generateTime"] = .number(generateTime)
        }
        if let stoppedBySequence = chunk.stoppedBySequence {
            metadata["stoppedBySequence"] = .bool(stoppedBySequence)
        }
        if let telemetry = chunk.speculativeTelemetry {
            metadata[AFMMLXSpeculativeTelemetry.metadataKey] = telemetry.metadataValue
        }
        if !metadata.isEmpty {
            events.append(.metadata(metadata))
        }
        return events
    }

    public mutating func finish() -> [AFMGenerationEvent] {
        var events = flushTextBuffer()
        let reason: AFMFinishReason
        if tools.values.contains(where: \.completed) {
            reason = .toolCalls
        } else if stoppedBySequence {
            reason = .stop
        } else if let maximumResponseTokens,
                  completionTokens >= maximumResponseTokens {
            reason = .length
        } else {
            reason = .stop
        }
        events.append(.completed(reason))
        return events
    }

    private mutating func textEvents(from chunk: StreamChunk) -> [AFMGenerationEvent] {
        guard !chunk.text.isEmpty else { return [] }
        let tokenCount = max(1, chunk.logprobs?.count ?? 1)
        guard let thinkStartTag, let thinkEndTag else {
            return [
                .responseText(action: .append, text: chunk.text, tokenCount: tokenCount)
            ]
        }

        textBuffer += chunk.text
        bufferedTokenCount += tokenCount
        var events: [AFMGenerationEvent] = []
        while !textBuffer.isEmpty {
            let boundary = insideReasoning ? thinkEndTag : thinkStartTag
            if let range = textBuffer.range(of: boundary) {
                let text = String(textBuffer[..<range.lowerBound])
                append(text, to: &events)
                textBuffer = String(textBuffer[range.upperBound...])
                insideReasoning.toggle()
                continue
            }

            let retainedCount = partialSuffixLength(in: textBuffer, matching: boundary)
            let emitCount = textBuffer.count - retainedCount
            guard emitCount > 0 else { break }
            let end = textBuffer.index(textBuffer.startIndex, offsetBy: emitCount)
            append(String(textBuffer[..<end]), to: &events)
            textBuffer = String(textBuffer[end...])
            break
        }
        return events
    }

    private mutating func flushTextBuffer() -> [AFMGenerationEvent] {
        guard !textBuffer.isEmpty else { return [] }
        var events: [AFMGenerationEvent] = []
        append(textBuffer, to: &events)
        textBuffer = ""
        return events
    }

    private mutating func append(
        _ text: String,
        to events: inout [AFMGenerationEvent]
    ) {
        guard !text.isEmpty else { return }
        let tokenCount = bufferedTokenCount
        bufferedTokenCount = 0
        if insideReasoning {
            events.append(
                .reasoningText(action: .append, text: text, tokenCount: tokenCount)
            )
        } else {
            events.append(
                .responseText(action: .append, text: text, tokenCount: tokenCount)
            )
        }
    }

    private func partialSuffixLength(in text: String, matching boundary: String) -> Int {
        let maximum = min(text.count, max(0, boundary.count - 1))
        guard maximum > 0 else { return 0 }
        for length in stride(from: maximum, through: 1, by: -1) {
            if text.suffix(length) == boundary.prefix(length) {
                return length
            }
        }
        return 0
    }

    private mutating func toolEvents(
        from deltas: [StreamDeltaToolCall]
    ) -> [AFMGenerationEvent] {
        var events: [AFMGenerationEvent] = []
        for delta in deltas {
            let existing = tools[delta.index]
            var state = existing ?? ToolState(
                id: delta.id ?? "call_\(delta.index)",
                name: delta.function?.name ?? ""
            )
            if let id = delta.id {
                state.id = id
            }
            if let name = delta.function?.name, !name.isEmpty {
                state.name = name
            }
            if existing == nil {
                events.append(.toolCall(call: state.call, stage: .started))
            }
            if let arguments = delta.function?.arguments, !arguments.isEmpty {
                let combinedArguments = state.arguments + arguments
                let normalizedArguments = normalizedCompleteArguments(
                    combinedArguments,
                    toolName: state.name,
                    index: delta.index,
                    id: state.id
                )
                let emittedArguments: String
                if state.arguments.isEmpty, let normalizedArguments {
                    state.arguments = normalizedArguments
                    emittedArguments = normalizedArguments
                } else {
                    state.arguments = combinedArguments
                    emittedArguments = arguments
                }
                events.append(
                    .toolCall(call: state.call, stage: .argumentsDelta(emittedArguments))
                )
            }
            tools[delta.index] = state
        }
        return events
    }

    private mutating func completedToolEvents(
        from completedCalls: [ResponseToolCall]
    ) -> [AFMGenerationEvent] {
        var events: [AFMGenerationEvent] = []
        for (fallbackIndex, rawCompletedCall) in completedCalls.enumerated() {
            let completedCall = MLXModelService.coerceArgumentTypes(
                rawCompletedCall,
                tools: requestTools
            )
            let index = completedCall.index ?? fallbackIndex
            let existing = tools[index]
            var state = existing ?? ToolState(
                id: completedCall.id,
                name: completedCall.function.name
            )
            // Keep the identity established by the incremental stream. Some
            // runtimes synthesize a fresh ID for their final aggregate call;
            // replacing the streamed ID makes downstream adapters treat the
            // completion as a second call and resend the full arguments.
            if existing == nil {
                state.id = completedCall.id
            }
            state.name = completedCall.function.name
            if existing == nil {
                events.append(.toolCall(call: state.call, stage: .started))
            }

            let rawFinalArguments = completedCall.function.arguments
            let finalArguments = rawFinalArguments
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "{}"
                : rawFinalArguments
            if finalArguments.hasPrefix(state.arguments) {
                let suffix = String(finalArguments.dropFirst(state.arguments.count))
                if !suffix.isEmpty {
                    state.arguments += suffix
                    events.append(
                        .toolCall(call: state.call, stage: .argumentsDelta(suffix))
                    )
                }
            } else if state.arguments != finalArguments {
                events.append(.toolCall(call: state.call, stage: .retracted))
                state.arguments = finalArguments
                events.append(.toolCall(call: state.call, stage: .started))
                if !finalArguments.isEmpty {
                    events.append(
                        .toolCall(
                            call: state.call,
                            stage: .argumentsDelta(finalArguments)
                        )
                    )
                }
            }

            state.completed = true
            events.append(.toolCall(call: state.call, stage: .completed))
            tools[index] = state
        }
        return events
    }

    private func normalizedCompleteArguments(
        _ arguments: String,
        toolName: String,
        index: Int,
        id: String
    ) -> String? {
        guard !toolName.isEmpty,
              let data = arguments.data(using: .utf8),
              (try? JSONSerialization.jsonObject(with: data)) is [String: Any] else {
            return nil
        }
        let call = ResponseToolCall(
            index: index,
            id: id,
            type: "function",
            function: ResponseToolCallFunction(
                name: toolName,
                arguments: arguments
            )
        )
        return MLXModelService.coerceArgumentTypes(call, tools: requestTools)
            .function.arguments
    }
}
