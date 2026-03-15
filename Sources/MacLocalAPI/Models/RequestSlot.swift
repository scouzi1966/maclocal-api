import Foundation

/// Ephemeral per-request generation state for concurrent inference.
/// Allocated when a request starts, discarded when it finishes.
final class RequestSlot: @unchecked Sendable {
    let id: UUID
    let startTime: Date
    let promptTokens: Int

    private let lock = NSLock()
    private var _fullContent: String = ""

    init(promptTokens: Int) {
        self.id = UUID()
        self.startTime = Date()
        self.promptTokens = promptTokens
    }

    /// Append generated text (thread-safe).
    func appendContent(_ text: String) {
        lock.lock()
        defer { lock.unlock() }
        _fullContent += text
    }

    /// Accumulated content so far.
    var fullContent: String {
        lock.lock()
        defer { lock.unlock() }
        return _fullContent
    }

    /// Elapsed generation time in seconds.
    var elapsedTime: TimeInterval {
        Date().timeIntervalSince(startTime)
    }
}
