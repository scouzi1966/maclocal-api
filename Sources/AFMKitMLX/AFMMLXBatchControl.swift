import Foundation

/// Identity-bearing capacity reserved from one concrete scheduler.
public final class AFMMLXSchedulerReservation: @unchecked Sendable, Hashable {
    package let schedulerID: UUID
    package let reservationID: UUID

    package init(schedulerID: UUID, reservationID: UUID = UUID()) {
        self.schedulerID = schedulerID
        self.reservationID = reservationID
    }

    public static func == (
        lhs: AFMMLXSchedulerReservation,
        rhs: AFMMLXSchedulerReservation
    ) -> Bool {
        lhs.schedulerID == rhs.schedulerID
            && lhs.reservationID == rhs.reservationID
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(schedulerID)
        hasher.combine(reservationID)
    }
}

/// Result of asking a runtime for request capacity.
public enum AFMMLXSchedulerAdmission: Equatable, Sendable {
    /// No scheduler is installed; the request must use the serial model path.
    case serial
    /// Capacity was reserved from a specific scheduler for this caller.
    case reserved(AFMMLXSchedulerReservation)
    /// Scheduler capacity is currently unavailable or admission is closed.
    case unavailable

    public var isAdmitted: Bool {
        switch self {
        case .serial, .reserved:
            return true
        case .unavailable:
            return false
        }
    }

    public var reservation: AFMMLXSchedulerReservation? {
        guard case .reserved(let reservation) = self else { return nil }
        return reservation
    }
}

/// Defines who admitted an attached MLX request.
public enum AFMMLXSchedulerAdmissionOwnership: Sendable {
    case model
    case caller(AFMMLXSchedulerAdmission)
}

public enum AFMMLXRequestMetadata {
    public static let preserveStructuralTags = "preserveStructuralTags"
}

/// Controls admission to an MLX runtime's concurrent request slots.
///
/// Server and app consumers use this contract without depending on the
/// concrete `MLXModelService` scheduler implementation.
public protocol AFMMLXRequestScheduling: Sendable {
    var maxConcurrent: Int { get }

    func tryReserveSlot() -> AFMMLXSchedulerAdmission
    func waitForSlot(timeout: TimeInterval) async -> AFMMLXSchedulerAdmission
    @discardableResult
    func releaseSlot(_ reservation: AFMMLXSchedulerReservation) -> Bool
}

public extension AFMMLXRequestScheduling {
    func waitForSlot(timeout: TimeInterval) async -> AFMMLXSchedulerAdmission {
        if Task.isCancelled { return .unavailable }
        if timeout <= 0 {
            return tryReserveSlot()
        }

        var admission = tryReserveSlot()
        if admission.isAdmitted { return admission }

        let initialPollNanoseconds: UInt64 = 10_000_000
        let maximumPollNanoseconds: UInt64 = 500_000_000
        let deadline = ContinuousClock.now + .seconds(timeout)
        var delay = initialPollNanoseconds

        while ContinuousClock.now < deadline {
            if Task.isCancelled { return .unavailable }
            try? await Task.sleep(nanoseconds: delay)
            if Task.isCancelled { return .unavailable }
            admission = tryReserveSlot()
            if admission.isAdmitted { return admission }
            delay = min(delay * 2, maximumPollNanoseconds)
        }
        return .unavailable
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
