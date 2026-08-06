#if canImport(FoundationModels)
import AFMKit
import Foundation

public enum AFMFoundationNativeProviderKind: String, Hashable, Sendable {
    case appleOnDevice
    case privateCloudCompute
}

public struct AFMFoundationNativeProviderCapabilitySnapshot: Hashable, Sendable {
    public let kind: AFMFoundationNativeProviderKind
    public let modelIdentifier: String
    public let capabilities: AFMModelCapabilities
    public let contextWindow: Int
    public let privacyBoundary: AFMPrivacyBoundary
    public let requiresNetwork: Bool
    public let entitlement: String?
    public let acceleration: String
    public let supportedReasoningLevels: Set<AFMFoundationReasoningLevel>

    public init(
        kind: AFMFoundationNativeProviderKind,
        modelIdentifier: String,
        capabilities: AFMModelCapabilities,
        contextWindow: Int,
        privacyBoundary: AFMPrivacyBoundary,
        requiresNetwork: Bool,
        entitlement: String?,
        acceleration: String,
        supportedReasoningLevels: Set<AFMFoundationReasoningLevel>
    ) {
        self.kind = kind
        self.modelIdentifier = modelIdentifier
        self.capabilities = capabilities
        self.contextWindow = contextWindow
        self.privacyBoundary = privacyBoundary
        self.requiresNetwork = requiresNetwork
        self.entitlement = entitlement
        self.acceleration = acceleration
        self.supportedReasoningLevels = supportedReasoningLevels
    }
}

public enum AFMFoundationNativeProviderCapabilities {
    public static let privateCloudComputeEntitlement = "com.apple.developer.private-cloud-compute"

    public static func appleOnDevice(
        systemContextWindow: Int,
        minimumContextWindow: Int = 8_192
    ) -> AFMFoundationNativeProviderCapabilitySnapshot {
        AFMFoundationNativeProviderCapabilitySnapshot(
            kind: .appleOnDevice,
            modelIdentifier: "apple.system.default",
            capabilities: [.text, .vision, .toolCalling, .structuredOutput],
            contextWindow: max(minimumContextWindow, systemContextWindow),
            privacyBoundary: .device,
            requiresNetwork: false,
            entitlement: nil,
            acceleration: "Apple Intelligence on-device",
            supportedReasoningLevels: []
        )
    }

    public static func privateCloudCompute(
        contextWindow: Int = 32_768,
        entitlement: String = privateCloudComputeEntitlement
    ) -> AFMFoundationNativeProviderCapabilitySnapshot {
        AFMFoundationNativeProviderCapabilitySnapshot(
            kind: .privateCloudCompute,
            modelIdentifier: "apple.private-cloud-compute",
            capabilities: [.text, .vision, .toolCalling, .structuredOutput],
            contextWindow: contextWindow,
            privacyBoundary: .privateCloud,
            requiresNetwork: true,
            entitlement: entitlement,
            acceleration: "Apple Private Cloud Compute",
            supportedReasoningLevels: [.light, .moderate, .deep]
        )
    }
}
#endif
