#if canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationTranscriptWindowPlan: Equatable {
    public let entries: [Transcript.Entry]
    public let removedPromptTurns: Int
    public let originalPromptCount: Int
    public let finalTokenCount: Int

    public init(
        entries: [Transcript.Entry],
        removedPromptTurns: Int,
        originalPromptCount: Int,
        finalTokenCount: Int
    ) {
        self.entries = entries
        self.removedPromptTurns = removedPromptTurns
        self.originalPromptCount = originalPromptCount
        self.finalTokenCount = finalTokenCount
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationTranscriptWindowPlannerError: Error, Equatable {
    case currentTurnExceedsWindow(requiredTokens: Int, maxTokens: Int)
}

@available(macOS 27.0, *)
public enum AFMFoundationTranscriptWindowPlanner {
    @MainActor
    public static func trimmingOldestPromptTurns(
        _ transcript: [Transcript.Entry],
        maxTokenCount: Int,
        tokenCount: ([Transcript.Entry]) async throws -> Int
    ) async throws -> AFMFoundationTranscriptWindowPlan {
        var entries = transcript
        var total = try await tokenCount(entries)
        let originalPromptCount = promptCount(in: entries)
        var removedTurns = 0

        while total > maxTokenCount {
            let promptIndices = entries.indices.filter { isPrompt(entries[$0]) }
            guard promptIndices.count > 1 else {
                throw AFMFoundationTranscriptWindowPlannerError.currentTurnExceedsWindow(
                    requiredTokens: total,
                    maxTokens: maxTokenCount
                )
            }

            let nextTurnStart = promptIndices[1]
            entries = leadingInstructions(in: entries) + Array(entries[nextTurnStart...])
            removedTurns += 1
            total = try await tokenCount(entries)
        }

        return AFMFoundationTranscriptWindowPlan(
            entries: entries,
            removedPromptTurns: removedTurns,
            originalPromptCount: originalPromptCount,
            finalTokenCount: total
        )
    }

    private static func promptCount(in entries: [Transcript.Entry]) -> Int {
        entries.reduce(into: 0) { count, entry in
            if isPrompt(entry) { count += 1 }
        }
    }

    private static func leadingInstructions(in entries: [Transcript.Entry]) -> [Transcript.Entry] {
        Array(entries.prefix {
            if case .instructions = $0 { return true }
            return false
        })
    }

    private static func isPrompt(_ entry: Transcript.Entry) -> Bool {
        if case .prompt = entry { return true }
        return false
    }
}
#endif
