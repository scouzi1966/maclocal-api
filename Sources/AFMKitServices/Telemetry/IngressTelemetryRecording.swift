import AFMKitCore
import Foundation

public enum AFMIngressRejectionReason: String, CaseIterable, Hashable, Sendable {
    case decode
    case authentication
    case validation
    case capacity
}

public struct AFMIngressConnectionToken: Hashable, Sendable {
    public let rawValue: UUID

    init(rawValue: UUID = UUID()) {
        self.rawValue = rawValue
    }
}

public protocol AFMIngressTelemetryRecording: Sendable {
    func recordRejection(_ reason: AFMIngressRejectionReason)
    func connectionOpened() -> AFMIngressConnectionToken
    func connectionClosed(_ token: AFMIngressConnectionToken)
}
