import Foundation
import os

public enum AFMGenerationAdmissionError: Error, Hashable, Sendable {
    case capacity
    case timedOut
    case cancelled
}

/// Provider-owned permission to execute one admitted generation.
public final class AFMGenerationLease: @unchecked Sendable {
    public let telemetryToken: AFMInferenceRequestToken

    private let state = OSAllocatedUnfairLock(initialState: false)
    private let releaseOperation: @Sendable () -> Void

    public init(
        telemetryToken: AFMInferenceRequestToken,
        release: @escaping @Sendable () -> Void
    ) {
        self.telemetryToken = telemetryToken
        self.releaseOperation = release
    }

    public func release() {
        let shouldRelease = state.withLock { released in
            guard !released else { return false }
            released = true
            return true
        }
        if shouldRelease { releaseOperation() }
    }

    deinit {
        release()
    }
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
