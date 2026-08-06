import Foundation

public struct AFMReasoningPendingUpdate: Equatable, Sendable {
    public let content: String?
    public let isReasoning: Bool
    public let duration: TimeInterval?

    public init(
        content: String?,
        isReasoning: Bool,
        duration: TimeInterval?
    ) {
        self.content = content
        self.isReasoning = isReasoning
        self.duration = duration
    }
}

public struct AFMReasoningUpdateBatcher: Equatable, Sendable {
    public private(set) var pendingContent: String?
    public private(set) var pendingIsReasoning: Bool
    public private(set) var pendingDuration: TimeInterval?
    public private(set) var tokensSinceFlush: Int

    public init(
        pendingContent: String? = nil,
        pendingIsReasoning: Bool = false,
        pendingDuration: TimeInterval? = nil,
        tokensSinceFlush: Int = 0
    ) {
        self.pendingContent = pendingContent
        self.pendingIsReasoning = pendingIsReasoning
        self.pendingDuration = pendingDuration
        self.tokensSinceFlush = tokensSinceFlush
    }

    public mutating func ingest(
        reasoningChunk: String?,
        isReasoning: Bool,
        duration: TimeInterval?,
        interval: Int = AFMReasoningStreamPolicy.defaultReasoningUpdateInterval
    ) -> AFMReasoningPendingUpdate? {
        if let reasoningChunk {
            pendingContent = (pendingContent ?? "") + reasoningChunk
        }
        pendingIsReasoning = isReasoning
        pendingDuration = duration
        tokensSinceFlush += 1

        guard AFMReasoningStreamPolicy.shouldFlushReasoningUpdate(
            tokensSinceLastFlush: tokensSinceFlush,
            interval: interval
        ) else {
            return nil
        }

        return drain()
    }

    public mutating func drainIfNeeded() -> AFMReasoningPendingUpdate? {
        guard pendingContent != nil || tokensSinceFlush > 0 else {
            return nil
        }
        return drain()
    }

    private mutating func drain() -> AFMReasoningPendingUpdate {
        let update = AFMReasoningPendingUpdate(
            content: pendingContent,
            isReasoning: pendingIsReasoning,
            duration: pendingDuration
        )
        pendingContent = nil
        tokensSinceFlush = 0
        return update
    }
}

public struct AFMReasoningStreamFinalization: Equatable, Sendable {
    public let hasReasoning: Bool
    public let finalContent: String
    public let historyText: String

    public init(
        hasReasoning: Bool,
        finalContent: String,
        historyText: String
    ) {
        self.hasReasoning = hasReasoning
        self.finalContent = finalContent
        self.historyText = historyText
    }
}

public enum AFMReasoningStreamPolicy {
    public static let defaultDetectionWindow = 4
    public static let defaultReasoningUpdateInterval = 4

    public static func shouldEnterFastPath(
        tokenCount: Int,
        detectionWindow: Int = defaultDetectionWindow,
        isInReasoningPhase: Bool,
        reasoningChunk: String?
    ) -> Bool {
        tokenCount == detectionWindow && !isInReasoningPhase && reasoningChunk == nil
    }

    public static func shouldFlushReasoningUpdate(
        tokensSinceLastFlush: Int,
        interval: Int = defaultReasoningUpdateInterval
    ) -> Bool {
        tokensSinceLastFlush >= interval
    }

    public static func finalize(
        skipReasoningParser: Bool,
        accumulatedText: String,
        parsedFinalContent: String,
        parsedReasoning: String?,
        parsedFormatName: String?
    ) -> AFMReasoningStreamFinalization {
        let finalContent = skipReasoningParser
            ? accumulatedText
            : (parsedFinalContent.isEmpty ? accumulatedText : parsedFinalContent)
        let historyText = !finalContent.isEmpty ? finalContent : (parsedReasoning ?? "")

        return AFMReasoningStreamFinalization(
            hasReasoning: parsedFormatName != nil,
            finalContent: finalContent,
            historyText: historyText
        )
    }
}

public struct AFMReasoningStreamChunkUpdate: Sendable {
    public let outputChunk: String?
    public let reasoningUpdate: AFMReasoningPendingUpdate?

    public init(
        outputChunk: String?,
        reasoningUpdate: AFMReasoningPendingUpdate?
    ) {
        self.outputChunk = outputChunk
        self.reasoningUpdate = reasoningUpdate
    }
}

public struct AFMReasoningStreamFinalState: Sendable {
    public let parsedResult: AFMReasoningParsedResponse
    public let finalization: AFMReasoningStreamFinalization

    public init(
        parsedResult: AFMReasoningParsedResponse,
        finalization: AFMReasoningStreamFinalization
    ) {
        self.parsedResult = parsedResult
        self.finalization = finalization
    }
}

public struct AFMReasoningStreamReducer: Sendable {
    private let parser: AFMReasoningOutputParser
    private let detectionWindow: Int
    private var reasoningUpdateBatcher: AFMReasoningUpdateBatcher
    public private(set) var tokenCount: Int
    public private(set) var accumulatedText: String
    public private(set) var skipReasoningParser: Bool

    public init(
        hasReasoningOutput: Bool,
        detectionWindow: Int = AFMReasoningStreamPolicy.defaultDetectionWindow,
        reasoningUpdateBatcher: AFMReasoningUpdateBatcher = AFMReasoningUpdateBatcher()
    ) {
        parser = AFMReasoningOutputParser(allowImplicitReasoning: hasReasoningOutput)
        self.detectionWindow = detectionWindow
        self.reasoningUpdateBatcher = reasoningUpdateBatcher
        tokenCount = 0
        accumulatedText = ""
        skipReasoningParser = false
    }

    public mutating func processChunk(_ string: String) -> AFMReasoningStreamChunkUpdate {
        tokenCount += 1

        if skipReasoningParser {
            accumulatedText += string
            return AFMReasoningStreamChunkUpdate(
                outputChunk: string,
                reasoningUpdate: nil
            )
        }

        let result = parser.processWithState(chunk: string)
        if AFMReasoningStreamPolicy.shouldEnterFastPath(
            tokenCount: tokenCount,
            detectionWindow: detectionWindow,
            isInReasoningPhase: result.isInReasoningPhase,
            reasoningChunk: result.reasoningChunk
        ) {
            skipReasoningParser = true
        }

        let reasoningUpdate = reasoningUpdateBatcher.ingest(
            reasoningChunk: result.reasoningChunk,
            isReasoning: result.isInReasoningPhase,
            duration: result.reasoningDuration
        )

        let outputChunk: String?
        if let final = result.finalChunk {
            accumulatedText += final
            outputChunk = final
        } else if result.isInReasoningPhase {
            outputChunk = ""
        } else {
            outputChunk = nil
        }

        return AFMReasoningStreamChunkUpdate(
            outputChunk: outputChunk,
            reasoningUpdate: reasoningUpdate
        )
    }

    public mutating func drainReasoningUpdate() -> AFMReasoningPendingUpdate? {
        reasoningUpdateBatcher.drainIfNeeded()
    }

    public func finalState() -> AFMReasoningStreamFinalState {
        let parsedResult = parser.getResult()
        return AFMReasoningStreamFinalState(
            parsedResult: parsedResult,
            finalization: AFMReasoningStreamPolicy.finalize(
                skipReasoningParser: skipReasoningParser,
                accumulatedText: accumulatedText,
                parsedFinalContent: parsedResult.finalContent,
                parsedReasoning: parsedResult.reasoning,
                parsedFormatName: parsedResult.formatName
            )
        )
    }
}
