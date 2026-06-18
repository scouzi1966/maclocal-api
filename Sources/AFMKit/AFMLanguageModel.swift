import Foundation

/// A provider-agnostic language-model abstraction, shaped to mirror Apple's
/// **WWDC 26 "Year Two" `LanguageModel` protocol** so afm's backends can slot into
/// (or bridge to) Apple's `LanguageModelSession` once macOS 27 ships.
///
/// In year two, Apple unifies on-device (`SystemLanguageModel`), Private Cloud Compute
/// (`PrivateCloudComputeLanguageModel`), and third-party providers (`ClaudeLanguageModel`,
/// `MLXLanguageModel`, …) behind a single `LanguageModel` protocol that backs any
/// `LanguageModelSession`. `AFMLanguageModel` is the afm-native analogue: it lets callers
/// treat the MLX backend, the Foundation Models backend, and (later) PCC / third-party
/// conformers interchangeably.
///
/// > Important: This protocol is intentionally **afm-native and source-stable on macOS 26**.
/// > It does *not* reference any year-two Apple symbol (`PrivateCloudComputeLanguageModel`,
/// > `LanguageModelSession.DynamicProfile`, `Attachment`, reasoning levels, …) — those are
/// > unavailable until the macOS 27 SDK. When that SDK lands, an `@available(macOS 27, *)`
/// > adapter will bridge a real Apple `LanguageModel` to this protocol (and vice-versa)
/// > without changing this surface. See `docs/wwdc26-migration.md`.
public protocol AFMLanguageModel: Sendable {
    /// Whether this model is usable in the current environment.
    var isAvailable: Bool { get }

    /// Produce a complete response for a chat transcript.
    func respond(to messages: [Message], options: GenerationConfig) async throws -> AFMResponse

    /// Stream a response as incremental text deltas.
    func streamResponse(to messages: [Message], options: GenerationConfig) -> AsyncThrowingStream<String, Error>
}

public extension AFMLanguageModel {
    /// Convenience: respond with default generation options.
    func respond(to messages: [Message]) async throws -> AFMResponse {
        try await respond(to: messages, options: GenerationConfig())
    }
}

// MARK: - AFMEngine conformance

extension AFMEngine: AFMLanguageModel {
    public nonisolated var isAvailable: Bool {
        switch backend {
        case .mlx:
            return true
        case .foundationModels:
            if #available(macOS 26.0, *) { return true } else { return false }
        }
    }

    public func respond(to messages: [Message], options: GenerationConfig) async throws -> AFMResponse {
        try await respond(to: messages, options)
    }

    public nonisolated func streamResponse(to messages: [Message], options: GenerationConfig) -> AsyncThrowingStream<String, Error> {
        streamRespond(to: messages, options)
    }
}
