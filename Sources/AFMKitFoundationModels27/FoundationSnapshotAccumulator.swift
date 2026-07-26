#if canImport(FoundationModels)
import Foundation

@available(macOS 27.0, *)
public struct AFMFoundationSnapshotUpdate<ProgressState: Equatable>: Equatable {
    public let responseDelta: String?
    public let shouldYieldProgressUpdate: Bool
    public let firstChunkStarted: Bool
    public let streamChunkCount: Int
    public let reasoningContent: String
    public let isInReasoningPhase: Bool
    public let progressState: ProgressState
}

/// Converts Foundation Models response snapshots into incremental UI-friendly
/// deltas while tracking reasoning-only progress and tool progress refreshes.
@available(macOS 27.0, *)
public struct AFMFoundationSnapshotAccumulator<ProgressState: Equatable> {
    private var previousResponseSnapshot = ""
    private var previousReasoningSnapshot = ""
    private(set) public var progressState: ProgressState
    private(set) public var reasoningContent = ""
    private(set) public var isInReasoningPhase = false
    private(set) public var streamChunkCount = 0

    public init(initialProgressState: ProgressState) {
        self.progressState = initialProgressState
    }

    public mutating func consume(
        content: String,
        progressState newProgressState: ProgressState,
        reasoningContent newReasoningContent: String = ""
    ) -> AFMFoundationSnapshotUpdate<ProgressState> {
        let progressChanged = newProgressState != progressState
        progressState = newProgressState

        var responseDelta: String?
        var firstChunkStarted = false

        let reasoningChanged = newReasoningContent != previousReasoningSnapshot
        if reasoningChanged {
            previousReasoningSnapshot = newReasoningContent
            reasoningContent = newReasoningContent
            streamChunkCount += 1
            firstChunkStarted = true
            isInReasoningPhase = !newReasoningContent.isEmpty
        }

        if content.count >= previousResponseSnapshot.count {
            let delta = String(content.dropFirst(previousResponseSnapshot.count))
            if !delta.isEmpty {
                streamChunkCount += 1
                firstChunkStarted = true
                responseDelta = delta
                isInReasoningPhase = false
            }
            previousResponseSnapshot = content
        } else {
            streamChunkCount += 1
            firstChunkStarted = true
            responseDelta = content
            previousResponseSnapshot = content
            isInReasoningPhase = false
        }

        return AFMFoundationSnapshotUpdate(
            responseDelta: responseDelta,
            shouldYieldProgressUpdate: responseDelta == nil && (progressChanged || reasoningChanged),
            firstChunkStarted: firstChunkStarted,
            streamChunkCount: streamChunkCount,
            reasoningContent: reasoningContent,
            isInReasoningPhase: isInReasoningPhase,
            progressState: progressState
        )
    }
}
#endif
