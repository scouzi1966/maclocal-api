import Foundation
import Vapor

/// In-memory registry of inflight chat completion requests, keyed by the
/// `request_id` minted (or echoed) by `RequestIDMiddleware`. Used by:
///
/// - **T1.5 — `/v1/chat/completions/{id}/cancel`** to cancel a turn the agent
///   has decided is no longer wanted.
/// - **T1.4 — client-disconnect cancellation** as a fallback when the streaming
///   closure detects a closed connection and wants to short-circuit the
///   generator.
///
/// The registry stores a `@Sendable` cancel closure rather than the `Task`
/// itself so callers can compose multiple cancellation effects (e.g. cancel
/// the asyncStream task plus the underlying model generator) under one id.
actor InflightRequestRegistry {
    private var inflight: [String: @Sendable () -> Void] = [:]

    /// Number of currently-tracked requests. Useful for tests / metrics.
    var count: Int { inflight.count }

    /// Register a cancel closure for `id`. Replaces any prior registration
    /// with the same id (e.g. when a controller re-registers after slot wait).
    /// No-op for empty ids — callers that don't run behind RequestIDMiddleware
    /// won't have one.
    func register(id: String, cancel: @escaping @Sendable () -> Void) {
        guard !id.isEmpty else { return }
        inflight[id] = cancel
    }

    /// Remove `id` from the registry without firing the cancel closure.
    /// Called from `defer` in the request handler when work completes normally.
    func release(id: String) {
        guard !id.isEmpty else { return }
        inflight.removeValue(forKey: id)
    }

    /// Cancel `id` and return whether it was found. Removes the entry as a
    /// side effect so subsequent cancels are no-ops.
    @discardableResult
    func cancel(id: String) -> Bool {
        guard let cancel = inflight.removeValue(forKey: id) else { return false }
        cancel()
        return true
    }
}

extension Application {
    private struct InflightRegistryKey: StorageKey {
        typealias Value = InflightRequestRegistry
    }

    /// Process-wide inflight registry. Lazily created on first access.
    var inflightRegistry: InflightRequestRegistry {
        if let registry = storage[InflightRegistryKey.self] {
            return registry
        }
        let registry = InflightRequestRegistry()
        storage[InflightRegistryKey.self] = registry
        return registry
    }
}

extension Request {
    /// Convenience accessor for the application-wide inflight registry.
    var inflightRegistry: InflightRequestRegistry {
        application.inflightRegistry
    }
}
