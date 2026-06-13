// Sources/MacLocalAPI/Models/RequestScheduler.swift
import Foundation

// MARK: - Design Note: Continuous Batching Constraints
//
// The current mlx-swift-lm TokenIterator does NOT support batch dimension > 1.
// model.callAsFunction(input, cache) expects input shape [1, seq_len].
//
// True continuous batching requires:
// 1. Modifying vendor model code to accept batched inputs (batch dim > 1)
// 2. Per-sequence KV cache management (paged attention)
// 3. Batched attention with per-sequence masks
// 4. Upstream MLX Swift changes
//
// This scheduler provides a practical first step: a request queue with
// round-robin scheduling that allows fair request interleaving at the
// request level (one request runs to completion, then the next starts).
// It replaces nothing yet — SerialAccessContainer still handles mutual
// exclusion. This is scaffolding for when batch-level changes are possible.

/// Request slot in the scheduler queue.
struct ScheduledRequest: Sendable {
    let id: UUID
    let continuation: CheckedContinuation<Void, Error>
    let priority: Int  // lower = higher priority
}

/// Round-robin request scheduler.
/// Provides fair ordering for concurrent requests.
/// Currently does not batch forward passes — each request still
/// gets exclusive model access via SerialAccessContainer.
actor RequestScheduler {
    private var queue: [ScheduledRequest] = []
    private var activeRequestID: UUID?

    /// Enqueue a request. Suspends until it's this request's turn.
    func enqueue(id: UUID = UUID(), priority: Int = 0) async throws {
        if activeRequestID == nil {
            activeRequestID = id
            return
        }
        try await withCheckedThrowingContinuation { cont in
            queue.append(ScheduledRequest(id: id, continuation: cont, priority: priority))
            queue.sort { $0.priority < $1.priority }
        }
    }

    /// Signal that the current request is done.
    func dequeue(id: UUID) {
        guard activeRequestID == id else { return }
        if let next = queue.first {
            queue.removeFirst()
            activeRequestID = next.id
            next.continuation.resume()
        } else {
            activeRequestID = nil
        }
    }

    /// Number of requests waiting in the queue.
    var queueDepth: Int { queue.count }

    /// Whether any request is currently active.
    var hasActiveRequest: Bool { activeRequestID != nil }
}
