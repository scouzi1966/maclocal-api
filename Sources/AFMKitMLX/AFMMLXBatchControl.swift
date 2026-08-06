import Foundation

/// Controls admission to an MLX runtime's concurrent request slots.
///
/// Server and app consumers use this contract without depending on the
/// concrete `MLXModelService` scheduler implementation.
public protocol AFMMLXRequestScheduling: Sendable {
    var maxConcurrent: Int { get }

    func tryReserveSlot() -> Bool
    func waitForSlot(timeout: TimeInterval) async -> Bool
    func releaseSlot()
}

public extension AFMMLXRequestScheduling {
    func waitForSlot(timeout: TimeInterval) async -> Bool {
        if Task.isCancelled { return false }
        if timeout <= 0 {
            return tryReserveSlot()
        }

        if tryReserveSlot() { return true }

        let initialPollNanoseconds: UInt64 = 10_000_000
        let maximumPollNanoseconds: UInt64 = 500_000_000
        let deadline = ContinuousClock.now + .seconds(timeout)
        var delay = initialPollNanoseconds

        while ContinuousClock.now < deadline {
            if Task.isCancelled { return false }
            try? await Task.sleep(nanoseconds: delay)
            if Task.isCancelled { return false }
            if tryReserveSlot() { return true }
            delay = min(delay * 2, maximumPollNanoseconds)
        }
        return false
    }
}

/// Controls the temporary batch runtime lifecycle used by bulk generation.
///
/// A successful `ensureBatchMode` call owns one reference that must be paired
/// with `releaseBatchReference`, including when generation fails or is
/// cancelled.
public protocol AFMMLXBatchControlling: Sendable {
    func ensureBatchMode(concurrency: Int) async throws
    func releaseBatchReference()
    func cancelBatchSlots(ids: Set<UUID>) async
}

extension MLXModelService: AFMMLXRequestScheduling, AFMMLXBatchControlling {}
