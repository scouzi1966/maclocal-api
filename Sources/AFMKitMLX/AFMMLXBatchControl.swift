import Foundation
import AFMKitCore

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

/// Provider-owned admission capability used by qualified built-in runtimes.
public protocol AFMMLXGenerationAdmitting: AFMMLXRequestScheduling {
    var generationAdmitter: AnyAFMGenerationAdmitter { get }
}

@available(
    *,
    deprecated,
    message: "External legacy schedulers are not queue-telemetry qualified."
)
public final class LegacyAFMMLXAdmissionAdapter:
    AFMGenerationAdmitting,
    @unchecked Sendable
{
    private let scheduler: any AFMMLXRequestScheduling
    private let observer: any AFMInferenceTelemetryObserving

    public init(
        scheduler: any AFMMLXRequestScheduling,
        observer: any AFMInferenceTelemetryObserving = AFMNoopInferenceTelemetryObserver()
    ) {
        self.scheduler = scheduler
        self.observer = observer
    }

    public func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        let acceptedAt = ProcessInfo.processInfo.systemUptime
        let token = observer.requestAccepted(at: acceptedAt)
        let seconds = timeout.map(Self.seconds) ?? 30
        guard await scheduler.waitForSlot(timeout: seconds) else {
            let reason: AFMInferenceFailureReason = Task.isCancelled ? .cancelled : .inference
            _ = observer.requestFailed(
                token,
                reason: reason,
                at: ProcessInfo.processInfo.systemUptime
            )
            throw Task.isCancelled
                ? AFMGenerationAdmissionError.cancelled
                : AFMGenerationAdmissionError.timedOut
        }
        observer.requestStarted(token, at: ProcessInfo.processInfo.systemUptime)
        return AFMGenerationLease(telemetryToken: token) { [scheduler] in
            scheduler.releaseSlot()
        } onAbandon: { [observer] in
            _ = observer.requestFailed(
                token,
                reason: .internal,
                at: ProcessInfo.processInfo.systemUptime
            )
        }
    }

    private static func seconds(_ duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds)
            + TimeInterval(components.attoseconds) / 1_000_000_000_000_000_000
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

extension MLXModelService:
    AFMMLXGenerationAdmitting,
    AFMMLXBatchControlling
{}
