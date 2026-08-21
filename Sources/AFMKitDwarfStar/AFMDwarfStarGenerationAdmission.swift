import AFMKitCore
import Foundation
import os

final class AFMDwarfStarGenerationAdmission: AFMGenerationAdmitting, @unchecked Sendable {
    private struct State {
        var reservations: Set<UUID> = []
        var waitingRequests = 0
    }

    private let maximumConcurrentRequests: Int
    private let telemetryObserver: any AFMInferenceTelemetryObserving
    private let state = OSAllocatedUnfairLock(initialState: State())

    init(
        maximumConcurrentRequests: Int,
        telemetryObserver: any AFMInferenceTelemetryObserving
    ) {
        self.maximumConcurrentRequests = max(1, maximumConcurrentRequests)
        self.telemetryObserver = telemetryObserver
    }

    func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        let acceptedAt = ProcessInfo.processInfo.systemUptime
        let telemetryToken = telemetryObserver.requestAccepted(at: acceptedAt)
        let reservationID = UUID()
        state.withLock { $0.waitingRequests += 1 }
        publishProviderState()

        let timeoutSeconds = timeout.map(Self.timeInterval) ?? 30
        let deadline = ContinuousClock.now + .seconds(max(0, timeoutSeconds))
        var delay: UInt64 = 10_000_000

        while !reserve(reservationID) {
            if Task.isCancelled {
                failWaitingRequest(telemetryToken, reason: .cancelled)
                throw AFMGenerationAdmissionError.cancelled
            }
            guard timeoutSeconds > 0 else {
                failWaitingRequest(telemetryToken, reason: .inference)
                throw AFMGenerationAdmissionError.capacity
            }
            guard ContinuousClock.now < deadline else {
                failWaitingRequest(telemetryToken, reason: .inference)
                throw AFMGenerationAdmissionError.timedOut
            }
            try? await Task.sleep(nanoseconds: delay)
            delay = min(delay * 2, 500_000_000)
        }

        telemetryObserver.requestStarted(
            telemetryToken,
            at: ProcessInfo.processInfo.systemUptime
        )
        publishProviderState()
        return AFMGenerationLease(telemetryToken: telemetryToken) { [weak self] in
            self?.release(reservationID)
        } onAbandon: { [telemetryObserver] in
            _ = telemetryObserver.requestFailed(
                telemetryToken,
                reason: .internal,
                at: ProcessInfo.processInfo.systemUptime
            )
        }
    }

    private func reserve(_ reservationID: UUID) -> Bool {
        state.withLock { state in
            guard state.reservations.count < maximumConcurrentRequests else {
                return false
            }
            state.reservations.insert(reservationID)
            state.waitingRequests = max(0, state.waitingRequests - 1)
            return true
        }
    }

    private func release(_ reservationID: UUID) {
        let removed = state.withLock { $0.reservations.remove(reservationID) != nil }
        if removed { publishProviderState() }
    }

    private func failWaitingRequest(
        _ telemetryToken: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason
    ) {
        state.withLock { $0.waitingRequests = max(0, $0.waitingRequests - 1) }
        _ = telemetryObserver.requestFailed(
            telemetryToken,
            reason: reason,
            at: ProcessInfo.processInfo.systemUptime
        )
        publishProviderState()
    }

    private func publishProviderState() {
        let snapshot = state.withLock { state in
            AFMInferenceProviderState(
                runningRequests: state.reservations.count,
                waitingRequests: state.waitingRequests
            )
        }
        telemetryObserver.updateProviderState(snapshot)
    }

    private static func timeInterval(_ duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds)
            + TimeInterval(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
