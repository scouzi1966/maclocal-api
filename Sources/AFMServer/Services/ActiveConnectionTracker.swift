import Foundation
import os

struct ActiveConnectionSnapshot: Sendable {
    let activeConnections: Int
    let activeConnectionsPeak: Int
}

/// Server-owned HTTP lifecycle gauge for `/metrics`.
///
/// Provider runtimes expose model execution observations through AFMKit. Active
/// HTTP connections belong to the Vapor server and stay in AFMServer so AFMKit
/// does not expose transport-specific state.
final class ActiveConnectionTracker: @unchecked Sendable {
    static let shared = ActiveConnectionTracker()

    private struct State {
        var activeConnections: Int = 0
        var activeConnectionsPeak: Int = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    init() {}

    func connectionStarted() {
        state.withLock { state in
            state.activeConnections += 1
            if state.activeConnections > state.activeConnectionsPeak {
                state.activeConnectionsPeak = state.activeConnections
            }
        }
    }

    func connectionEnded() {
        state.withLock { state in
            if state.activeConnections > 0 {
                state.activeConnections -= 1
            }
        }
    }

    func snapshot() -> ActiveConnectionSnapshot {
        state.withLock {
            ActiveConnectionSnapshot(
                activeConnections: $0.activeConnections,
                activeConnectionsPeak: $0.activeConnectionsPeak
            )
        }
    }
}
