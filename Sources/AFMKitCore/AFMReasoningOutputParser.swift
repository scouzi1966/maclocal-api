import Foundation

public struct AFMReasoningMarkerFormat: Sendable, Equatable {
    public let name: String
    public let startMarkers: [String]
    public let endMarker: String
    public let stripMarkers: [String]

    public init(
        name: String,
        startMarkers: [String],
        endMarker: String,
        stripMarkers: [String]
    ) {
        self.name = name
        self.startMarkers = startMarkers
        self.endMarker = endMarker
        self.stripMarkers = stripMarkers
    }

    public static let knownFormats: [AFMReasoningMarkerFormat] = [
        AFMReasoningMarkerFormat(
            name: "gptOSS",
            startMarkers: ["<|channel|>", "<|start|>"],
            endMarker: "<|channel|>final<|message|>",
            stripMarkers: [
                "analysis<|message|>",
                "<|channel|>",
                "<|message|>",
                "<|start|>",
                "<|end|>",
                "<|return|>"
            ]
        ),
        AFMReasoningMarkerFormat(
            name: "thinkTags",
            startMarkers: ["<think>"],
            endMarker: "</think>",
            stripMarkers: ["<think>", "</think>"]
        ),
        AFMReasoningMarkerFormat(
            name: "thinkTagsImplicitStart",
            startMarkers: [],
            endMarker: "</think>",
            stripMarkers: ["</think>"]
        )
    ]
}

public struct AFMReasoningParsedResponse: Sendable, Equatable {
    public let reasoning: String?
    public let finalContent: String
    public let formatName: String?
    public let reasoningTokenCount: Int?

    public init(
        reasoning: String?,
        finalContent: String,
        formatName: String?,
        reasoningTokenCount: Int?
    ) {
        self.reasoning = reasoning
        self.finalContent = finalContent
        self.formatName = formatName
        self.reasoningTokenCount = reasoningTokenCount
    }
}

public struct AFMReasoningChunkProcessResult: Sendable, Equatable {
    public let reasoningChunk: String?
    public let finalChunk: String?
    public let isInReasoningPhase: Bool
    public let reasoningDuration: TimeInterval?

    public init(
        reasoningChunk: String?,
        finalChunk: String?,
        isInReasoningPhase: Bool,
        reasoningDuration: TimeInterval?
    ) {
        self.reasoningChunk = reasoningChunk
        self.finalChunk = finalChunk
        self.isInReasoningPhase = isInReasoningPhase
        self.reasoningDuration = reasoningDuration
    }
}

public enum AFMReasoningOutputExtractor {
    public static func extractThinkContent(
        from text: String,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (content: String, reasoning: String?) {
        guard text.contains(startTag) else { return (text, nil) }
        var buffer = text
        var inside = false
        var allReasoning = ""
        var allContent = ""

        while !buffer.isEmpty {
            let extracted = extractThinkTags(
                buffer: &buffer,
                insideThinkBlock: &inside,
                startTag: startTag,
                endTag: endTag
            )
            if let reasoning = extracted.reasoning { allReasoning += reasoning }
            if let content = extracted.content { allContent += content }
            if extracted.reasoning == nil && extracted.content == nil { break }
        }

        if !buffer.isEmpty {
            if inside {
                allReasoning += buffer
            } else {
                allContent += buffer
            }
        }

        let reasoning = allReasoning.isEmpty ? nil : allReasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let content = allContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return (content, reasoning)
    }

    private static func extractThinkTags(
        buffer: inout String,
        insideThinkBlock: inout Bool,
        startTag: String,
        endTag: String
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""
        let startTagLen = startTag.count
        let endTagLen = endTag.count

        while !buffer.isEmpty {
            if insideThinkBlock {
                if let endRange = buffer.range(of: endTag) {
                    reasoning += String(buffer[buffer.startIndex..<endRange.lowerBound])
                    buffer = String(buffer[endRange.upperBound...])
                    insideThinkBlock = false
                } else if buffer.count > endTagLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -endTagLen)
                    reasoning += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break
                } else {
                    break
                }
            } else {
                if let startRange = buffer.range(of: startTag) {
                    content += String(buffer[buffer.startIndex..<startRange.lowerBound])
                    buffer = String(buffer[startRange.upperBound...])
                    insideThinkBlock = true
                } else if buffer.count > startTagLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -startTagLen)
                    content += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break
                } else {
                    break
                }
            }
        }

        return (
            reasoning: reasoning.isEmpty ? nil : reasoning,
            content: content.isEmpty ? nil : content
        )
    }
}

private struct AFMReasoningOutputState {
    var buffer = ""
    var detectedFormat: AFMReasoningMarkerFormat?
    var reasoningComplete = false
    var reasoningContent = ""
    var finalContent = ""
    var reasoningStartTime: Date?
    var reasoningEndTime: Date?
    var implicitReasoningMode = false
    var allowImplicitReasoning: Bool
    var reasoningStreamStarted = false
    var tailBuffer = ""

    init(allowImplicitReasoning: Bool) {
        self.allowImplicitReasoning = allowImplicitReasoning
    }

    mutating func process(chunk: String) -> (reasoning: String?, final: String?) {
        if !reasoningStreamStarted {
            buffer += chunk
        }

        if detectedFormat == nil {
            detectFormat()
        }

        if implicitReasoningMode {
            return processImplicitReasoning(chunk: chunk)
        }

        guard let format = detectedFormat else {
            if allowImplicitReasoning {
                implicitReasoningMode = true
                reasoningStreamStarted = true
                reasoningStartTime = Date()
                detectedFormat = AFMReasoningMarkerFormat.knownFormats.first { $0.name == "thinkTagsImplicitStart" }
                return processImplicitReasoning(chunk: chunk)
            }
            finalContent += chunk
            return (nil, chunk)
        }

        return processWithFormat(format, chunk: chunk)
    }

    mutating func processWithState(chunk: String) -> AFMReasoningChunkProcessResult {
        let (reasoningChunk, finalChunk) = process(chunk: chunk)
        return AFMReasoningChunkProcessResult(
            reasoningChunk: reasoningChunk,
            finalChunk: finalChunk,
            isInReasoningPhase: detectedFormat != nil && !reasoningComplete,
            reasoningDuration: reasoningDuration
        )
    }

    var result: AFMReasoningParsedResponse {
        let reasoningTokens = reasoningContent.isEmpty ? nil : reasoningContent.count / 4
        return AFMReasoningParsedResponse(
            reasoning: reasoningContent.isEmpty ? nil : reasoningContent,
            finalContent: finalContent,
            formatName: detectedFormat?.name,
            reasoningTokenCount: reasoningTokens
        )
    }

    var reasoningDuration: TimeInterval? {
        guard let start = reasoningStartTime else { return nil }
        let end = reasoningEndTime ?? Date()
        return end.timeIntervalSince(start)
    }

    var isInReasoningPhase: Bool {
        detectedFormat != nil && !reasoningComplete
    }

    mutating func reset() {
        let keepImplicitReasoning = allowImplicitReasoning
        self = AFMReasoningOutputState(allowImplicitReasoning: keepImplicitReasoning)
    }

    private mutating func processImplicitReasoning(chunk: String) -> (reasoning: String?, final: String?) {
        if reasoningComplete {
            finalContent += chunk
            return (nil, chunk)
        }

        tailBuffer += chunk

        if let endRange = tailBuffer.range(of: "</think>") {
            reasoningComplete = true
            reasoningEndTime = Date()

            let beforeEnd = String(tailBuffer[..<endRange.lowerBound])
            if !beforeEnd.isEmpty {
                reasoningContent += beforeEnd
            }

            let afterEnd = String(tailBuffer[endRange.upperBound...])
            finalContent = afterEnd
            let cleanedFinal = afterEnd.trimmingCharacters(in: .whitespacesAndNewlines)
            tailBuffer = ""

            return (
                beforeEnd.isEmpty ? nil : beforeEnd,
                cleanedFinal.isEmpty ? nil : cleanedFinal
            )
        }

        let keepLength = "</think>".count + 32
        if tailBuffer.count > keepLength * 2 {
            let emitCount = tailBuffer.count - keepLength
            let emitEnd = tailBuffer.index(tailBuffer.startIndex, offsetBy: emitCount)
            let emitted = String(tailBuffer[..<emitEnd])
            tailBuffer = String(tailBuffer[emitEnd...])

            reasoningContent += emitted
            return (emitted.isEmpty ? nil : emitted, nil)
        }

        return (nil, nil)
    }

    private mutating func detectFormat() {
        for format in AFMReasoningMarkerFormat.knownFormats {
            for startMarker in format.startMarkers where buffer.contains(startMarker) {
                detectedFormat = format
                reasoningStartTime = Date()
                return
            }
        }
    }

    private mutating func processWithFormat(
        _ format: AFMReasoningMarkerFormat,
        chunk: String
    ) -> (reasoning: String?, final: String?) {
        if !reasoningComplete {
            if reasoningStreamStarted {
                tailBuffer += chunk

                if let endRange = tailBuffer.range(of: format.endMarker) {
                    reasoningComplete = true
                    reasoningEndTime = Date()

                    let beforeEnd = String(tailBuffer[..<endRange.lowerBound])
                    let cleanedBefore = stripMarkers(from: beforeEnd, markers: format.stripMarkers)
                    if !cleanedBefore.isEmpty {
                        reasoningContent += cleanedBefore
                    }

                    let afterEnd = String(tailBuffer[endRange.upperBound...])
                    let cleanedFinal = stripMarkers(from: afterEnd, markers: format.stripMarkers)
                    finalContent = cleanedFinal
                    tailBuffer = ""

                    return (
                        cleanedBefore.isEmpty ? nil : cleanedBefore,
                        cleanedFinal.isEmpty ? nil : cleanedFinal
                    )
                }

                let keepLength = format.endMarker.count + 32
                if tailBuffer.count > keepLength * 2 {
                    let emitCount = tailBuffer.count - keepLength
                    let emitEnd = tailBuffer.index(tailBuffer.startIndex, offsetBy: emitCount)
                    let emitted = String(tailBuffer[..<emitEnd])
                    tailBuffer = String(tailBuffer[emitEnd...])

                    let cleaned = stripMarkers(from: emitted, markers: format.stripMarkers)
                    reasoningContent += cleaned
                    return (cleaned.isEmpty ? nil : cleaned, nil)
                }

                return (nil, nil)
            }

            if let endRange = buffer.range(of: format.endMarker) {
                reasoningComplete = true
                reasoningEndTime = Date()

                let beforeEnd = String(buffer[..<endRange.lowerBound])
                reasoningContent = stripMarkers(from: beforeEnd, markers: format.stripMarkers)
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                let afterEnd = String(buffer[endRange.upperBound...])
                let cleanedFinal = stripMarkers(from: afterEnd, markers: format.stripMarkers)
                finalContent = cleanedFinal

                return (nil, cleanedFinal.isEmpty ? nil : cleanedFinal)
            }

            for startMarker in format.startMarkers {
                if let startRange = buffer.range(of: startMarker) {
                    let afterStart = String(buffer[startRange.upperBound...])
                    let cleanedReasoning = stripMarkers(from: afterStart, markers: format.stripMarkers)
                    reasoningContent = cleanedReasoning
                    reasoningStreamStarted = true
                    tailBuffer = ""
                    return (cleanedReasoning.isEmpty ? nil : cleanedReasoning, nil)
                }
            }

            return (nil, nil)
        }

        let cleanedChunk = stripMarkers(from: chunk, markers: format.stripMarkers)
        finalContent += cleanedChunk
        return (nil, cleanedChunk.isEmpty ? nil : cleanedChunk)
    }

    private func stripMarkers(from text: String, markers: [String]) -> String {
        var result = text
        for marker in markers {
            result = result.replacingOccurrences(of: marker, with: "")
        }
        return result
    }
}

public final class AFMReasoningOutputParser: @unchecked Sendable {
    private nonisolated(unsafe) var state: AFMReasoningOutputState

    public nonisolated init(allowImplicitReasoning: Bool = false) {
        self.state = AFMReasoningOutputState(allowImplicitReasoning: allowImplicitReasoning)
    }

    public nonisolated func process(chunk: String) -> (reasoning: String?, final: String?) {
        state.process(chunk: chunk)
    }

    public nonisolated func processWithState(chunk: String) -> AFMReasoningChunkProcessResult {
        state.processWithState(chunk: chunk)
    }

    public nonisolated func getResult() -> AFMReasoningParsedResponse {
        state.result
    }

    public nonisolated func getReasoningDuration() -> TimeInterval? {
        state.reasoningDuration
    }

    public nonisolated func isInReasoningPhase() -> Bool {
        state.isInReasoningPhase
    }

    public nonisolated func reset() {
        state.reset()
    }

    public var hasReasoningFormat: Bool {
        get async { state.detectedFormat != nil }
    }

    public var formatName: String? {
        get async { state.detectedFormat?.name }
    }
}

public final class AFMReasoningOutputFilterSync: @unchecked Sendable {
    private let lock = NSLock()
    private nonisolated(unsafe) var state: AFMReasoningOutputState

    public nonisolated init(allowImplicitReasoning: Bool = false) {
        self.state = AFMReasoningOutputState(allowImplicitReasoning: allowImplicitReasoning)
    }

    public nonisolated func process(chunk: String) -> (reasoning: String?, final: String?) {
        lock.lock()
        defer { lock.unlock() }
        return state.process(chunk: chunk)
    }

    public nonisolated func getResult() -> AFMReasoningParsedResponse {
        lock.lock()
        defer { lock.unlock() }
        return state.result
    }

    public nonisolated func getReasoningDuration() -> TimeInterval? {
        lock.lock()
        defer { lock.unlock() }
        return state.reasoningDuration
    }

    public nonisolated func isInReasoningPhase() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return state.isInReasoningPhase
    }

    public nonisolated func reset() {
        lock.lock()
        defer { lock.unlock() }
        state.reset()
    }
}
