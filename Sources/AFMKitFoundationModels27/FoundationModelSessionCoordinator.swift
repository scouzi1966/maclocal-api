#if canImport(FoundationModels)
import Foundation
import FoundationModels

/// Reusable session identity, reuse, prewarm, and history-view coordination for
/// apps that expose Foundation Models providers behind their own provider IDs.
@MainActor
@available(macOS 27.0, *)
public final class AFMFoundationModelSessionCoordinator<ProviderID: Hashable & Sendable> {
    private struct FoundationSessionSlot {
        let provider: ProviderID
        let signature: String
        let session: LanguageModelSession
        let profileState: AFMFoundationDynamicProfileState?
    }

    private var foundationSession: FoundationSessionSlot?
    private var preservedHistory: [Transcript.Entry] = []

    public init() {}

    public func dynamicProfileSession<Model: LanguageModel>(
        for provider: ProviderID,
        signature: String,
        model: Model,
        tools: [any FoundationModels.Tool],
        instructions: String
    ) -> LanguageModelSession {
        if let existing = foundationSession,
           existing.provider == provider,
           existing.signature == signature {
            return existing.session
        }

        if let existing = foundationSession {
            preservedHistory = Array(existing.session.transcript)
            foundationSession = nil
        }

        let profileState = AFMFoundationDynamicProfileState(
            model: model,
            tools: tools,
            instructions: instructions
        )
        let profile = AFMFoundationDynamicProfile(state: profileState)
        let session = LanguageModelSession(profile: profile, history: preservedHistory)
        foundationSession = FoundationSessionSlot(
            provider: provider,
            signature: signature,
            session: session,
            profileState: profileState
        )
        return session
    }

    /// Creates a Foundation Models session through the direct custom-model
    /// initializer. Use this for provider packages that own transcript
    /// translation inside their `LanguageModelExecutor`.
    public func simpleSession<Model: LanguageModel>(
        for provider: ProviderID,
        signature: String,
        model: Model,
        tools: [any FoundationModels.Tool],
        instructions: String
    ) -> LanguageModelSession {
        if let existing = foundationSession,
           existing.provider == provider,
           existing.signature == signature {
            return existing.session
        }

        foundationSession = nil
        preservedHistory.removeAll()

        let session = LanguageModelSession(
            model: model,
            tools: tools,
            instructions: instructions
        )
        foundationSession = FoundationSessionSlot(
            provider: provider,
            signature: signature,
            session: session,
            profileState: nil
        )
        return session
    }

    public func invalidate(for provider: ProviderID) {
        guard let current = foundationSession, current.provider == provider else { return }
        preservedHistory = Array(current.session.transcript)
        foundationSession = nil
    }

    public func setHistoryView(_ entries: [Transcript.Entry], for provider: ProviderID) {
        guard let current = foundationSession,
              current.provider == provider,
              let profileState = current.profileState else { return }
        profileState.setHistoryView(entries)
    }

    public func clearHistoryView(for provider: ProviderID) {
        guard let current = foundationSession,
              current.provider == provider,
              let profileState = current.profileState else { return }
        profileState.setHistoryView(nil)
    }

    @discardableResult
    public func prewarm(promptPrefix: String, for provider: ProviderID) -> Bool {
        guard let current = foundationSession,
              current.provider == provider,
              !current.session.isResponding else { return false }
        current.session.prewarm(promptPrefix: Prompt { promptPrefix })
        return true
    }

    public func reset() {
        foundationSession = nil
        preservedHistory.removeAll()
    }
}

@available(macOS 27.0, *)
public nonisolated final class AFMFoundationDynamicProfileState: @unchecked Sendable {
    public struct Configuration {
        public let model: any LanguageModel
        public let tools: [any FoundationModels.Tool]
        public let instructions: String
    }

    private let lock = NSLock()
    private var configuration: Configuration
    private var historyView: [Transcript.Entry]?

    public init<Model: LanguageModel>(
        model: Model,
        tools: [any FoundationModels.Tool],
        instructions: String
    ) {
        configuration = Configuration(
            model: model,
            tools: tools,
            instructions: instructions
        )
    }

    public func update<Model: LanguageModel>(
        model: Model,
        tools: [any FoundationModels.Tool],
        instructions: String
    ) {
        lock.withLock {
            configuration = Configuration(
                model: model,
                tools: tools,
                instructions: instructions
            )
            historyView = nil
        }
    }

    public func snapshot() -> Configuration {
        lock.withLock { configuration }
    }

    public func setHistoryView(_ entries: [Transcript.Entry]?) {
        lock.withLock {
            historyView = entries
        }
    }

    public func transformedHistory(_ entries: [Transcript.Entry]) -> [Transcript.Entry] {
        lock.withLock {
            AFMFoundationHistoryTransform.normalized(historyView ?? entries)
        }
    }
}

@available(macOS 27.0, *)
public nonisolated enum AFMFoundationHistoryTransform {
    public static func normalized(_ entries: [Transcript.Entry]) -> [Transcript.Entry] {
        let instructions = entries.prefix {
            if case .instructions = $0 { return true }
            return false
        }
        let conversation = entries.dropFirst(instructions.count)
        guard let firstPrompt = conversation.firstIndex(where: {
            if case .prompt = $0 { return true }
            return false
        }) else {
            return Array(instructions)
        }
        return Array(instructions) + Array(conversation[firstPrompt...])
    }
}

@available(macOS 27.0, *)
nonisolated private struct AFMFoundationDynamicProfile: LanguageModelSession.DynamicProfile {
    let state: AFMFoundationDynamicProfileState

    var body: some LanguageModelSession.DynamicProfile {
        let configuration = state.snapshot()
        LanguageModelSession.Profile {
            Instructions(configuration.instructions)
            configuration.tools
        }
        .model(configuration.model)
        .historyTransform { entries in
            state.transformedHistory(entries)
        }
    }
}
#endif
