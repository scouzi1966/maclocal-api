import AFMKitCore
import AFMKitMLX
import AFMOpenAICompat

struct MLXStreamEventTranslator {
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
    private var textBuffer = ""
    private var bufferedTokenCount = 0
    private var insideReasoning = false
    private var tools: [Int: ToolState] = [:]
    private var completionTokens = 0
    private var stoppedBySequence = false

    init(
        thinkStartTag: String?,
        thinkEndTag: String?,
        maximumResponseTokens: Int?
    ) {
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.maximumResponseTokens = maximumResponseTokens
    }

    mutating func consume(_ chunk: StreamChunk) -> [AFMStreamEvent] {
        var events = textEvents(from: chunk)
        events.append(contentsOf: toolEvents(from: chunk.toolCallDeltas ?? []))
        events.append(contentsOf: completedToolEvents(from: chunk.toolCalls ?? []))

        if let promptTokens = chunk.promptTokens,
           let completionTokens = chunk.completionTokens {
            self.completionTokens = completionTokens
            events.append(
                .usage(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    cachedTokens: chunk.cachedTokens ?? 0
                )
            )
        }
        if chunk.stoppedBySequence == true {
            stoppedBySequence = true
        }
        return events
    }

    mutating func finish() -> [AFMStreamEvent] {
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

    private mutating func textEvents(from chunk: StreamChunk) -> [AFMStreamEvent] {
        guard !chunk.text.isEmpty else { return [] }
        let tokenCount = max(1, chunk.logprobs?.count ?? 1)
        guard let thinkStartTag, let thinkEndTag else {
            return [.text(chunk.text, tokenCount: tokenCount)]
        }

        textBuffer += chunk.text
        bufferedTokenCount += tokenCount
        var events: [AFMStreamEvent] = []
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

    private mutating func flushTextBuffer() -> [AFMStreamEvent] {
        guard !textBuffer.isEmpty else { return [] }
        var events: [AFMStreamEvent] = []
        append(textBuffer, to: &events)
        textBuffer = ""
        return events
    }

    private mutating func append(
        _ text: String,
        to events: inout [AFMStreamEvent]
    ) {
        guard !text.isEmpty else { return }
        let tokenCount = bufferedTokenCount
        bufferedTokenCount = 0
        if insideReasoning {
            events.append(.reasoning(text, tokenCount: tokenCount))
        } else {
            events.append(.text(text, tokenCount: tokenCount))
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
    ) -> [AFMStreamEvent] {
        var events: [AFMStreamEvent] = []
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
                events.append(.toolCall(state.call, stage: .started))
            }
            if let arguments = delta.function?.arguments, !arguments.isEmpty {
                state.arguments += arguments
                events.append(.toolCall(state.call, stage: .argumentsDelta(arguments)))
            }
            tools[delta.index] = state
        }
        return events
    }

    private mutating func completedToolEvents(
        from completedCalls: [ResponseToolCall]
    ) -> [AFMStreamEvent] {
        var events: [AFMStreamEvent] = []
        for (fallbackIndex, completedCall) in completedCalls.enumerated() {
            let index = completedCall.index ?? fallbackIndex
            let existing = tools[index]
            var state = existing ?? ToolState(
                id: completedCall.id,
                name: completedCall.function.name
            )
            state.id = completedCall.id
            state.name = completedCall.function.name
            if existing == nil {
                events.append(.toolCall(state.call, stage: .started))
            }

            let finalArguments = completedCall.function.arguments
            if finalArguments.hasPrefix(state.arguments) {
                let suffix = String(finalArguments.dropFirst(state.arguments.count))
                if !suffix.isEmpty {
                    state.arguments += suffix
                    events.append(.toolCall(state.call, stage: .argumentsDelta(suffix)))
                }
            } else if state.arguments != finalArguments {
                events.append(.toolCall(state.call, stage: .retracted))
                state.arguments = finalArguments
                events.append(.toolCall(state.call, stage: .started))
                if !finalArguments.isEmpty {
                    events.append(
                        .toolCall(state.call, stage: .argumentsDelta(finalArguments))
                    )
                }
            }

            state.completed = true
            events.append(.toolCall(state.call, stage: .completed))
            tools[index] = state
        }
        return events
    }
}
