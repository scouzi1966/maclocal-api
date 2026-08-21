import Foundation
import os

public enum AFMGenerationAdmissionError: Error, Hashable, Sendable {
    case capacity
    case timedOut
    case cancelled
    case internalFailure
}

/// Provider-owned permission to execute one admitted generation.
public final class AFMGenerationLease: @unchecked Sendable {
    public let telemetryToken: AFMInferenceRequestToken

    private struct State {
        var released = false
        var providerOwnsRelease = false
        var providerOwnsTelemetry = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let releaseOperation: @Sendable () -> Void
    private let abandonmentOperation: @Sendable () -> Void

    public init(
        telemetryToken: AFMInferenceRequestToken,
        release: @escaping @Sendable () -> Void,
        onAbandon: @escaping @Sendable () -> Void = {}
    ) {
        self.telemetryToken = telemetryToken
        self.releaseOperation = release
        self.abandonmentOperation = onAbandon
    }

    public func release() {
        let operations = state.withLock { state -> (release: Bool, abandon: Bool) in
            guard !state.released else { return (false, false) }
            state.released = true
            return (!state.providerOwnsRelease, !state.providerOwnsTelemetry)
        }
        if operations.release { releaseOperation() }
        if operations.abandon { abandonmentOperation() }
    }

    /// Marks the lease as released when the provider's own scheduler has taken
    /// responsibility for releasing the underlying capacity reservation.
    public func transferReleaseToProvider() {
        state.withLock { state in
            state.providerOwnsRelease = true
            state.providerOwnsTelemetry = true
        }
    }

    /// Keeps capacity release with the caller while the provider takes
    /// responsibility for recording the request's terminal telemetry state.
    public func transferTelemetryToProvider() {
        state.withLock { $0.providerOwnsTelemetry = true }
    }

    deinit {
        release()
    }
}

/// Request-scoped generation values propagated without changing established
/// provider method signatures.
public enum AFMGenerationContext {
    @TaskLocal public static var telemetryToken: AFMInferenceRequestToken?
    @TaskLocal public static var acceptedAt: Double?
    @TaskLocal public static var admissionLease: AFMGenerationLease?
    @TaskLocal public static var requestedMaximumOutputTokens: Int?
    @TaskLocal public static var ignoreEndOfSequence = false
}

public protocol AFMGenerationAdmitting: Sendable {
    func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease
}

public struct AnyAFMGenerationAdmitter: AFMGenerationAdmitting, Sendable {
    private let operation: @Sendable (Duration?) async throws -> AFMGenerationLease

    public init<Admitter: AFMGenerationAdmitting>(_ admitter: Admitter) {
        operation = { timeout in
            try await admitter.admitGeneration(timeout: timeout)
        }
    }

    public init(
        admitGeneration:
            @escaping @Sendable (Duration?) async throws -> AFMGenerationLease
    ) {
        operation = admitGeneration
    }

    public func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        try await operation(timeout)
    }
}
