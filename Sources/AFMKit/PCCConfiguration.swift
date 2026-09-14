import Foundation
import AFMKitCore
import AFMKitFoundationModels

/// Application configuration for AFMKit's existing Apple PCC provider.
public struct PCCConfiguration: Sendable {
    public enum RuntimeError: Error, LocalizedError {
        case unsupported

        public var errorDescription: String? {
            "Private Cloud Compute requires macOS 27 or later. Use afm without the pcc subcommand for on-device Foundation Models."
        }
    }

    public static func requireSupportedRuntime(
        version: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
    ) throws {
        guard version.majorVersion >= 27 else { throw RuntimeError.unsupported }
    }

    public enum Reasoning: String, CaseIterable, Sendable {
        case automatic, light, moderate, deep
    }

    public let instructions: String
    public let reasoning: Reasoning

    public init(instructions: String = "You are a helpful assistant", reasoning: Reasoning = .automatic) {
        self.instructions = instructions
        self.reasoning = reasoning
    }

    @available(macOS 27.0, *)
    public var providerConfiguration: AFMProviderConfiguration {
        .init(values: [
            AFMFoundationProviderConfigurationKeys.systemPrompt: .string(instructions),
            AFMFoundationProviderConfigurationKeys.reasoningLevel: .string(reasoning.rawValue),
        ])
    }

    @available(macOS 27.0, *)
    public func makeModel() throws -> AnyAFMModel {
        try AFMFoundationProviderFactory().makeModel(
            id: AFMFoundationProviderFactory.privateCloudComputeModelID,
            configuration: providerConfiguration)
    }

    @available(macOS 27.0, *)
    public func makeEngine() throws -> AFMEngine {
        let registry = AFMProviderRegistry()
        try registry.register(AFMFoundationProviderFactory())
        return try AFMEngine(
            providerID: AFMFoundationProviderFactory.providerID,
            modelID: AFMFoundationProviderFactory.privateCloudComputeModelID,
            configuration: providerConfiguration,
            registry: registry)
    }
}

/// Serializable diagnostics. Entitlement presence alone does not prove that
/// macOS accepted the provisioning profile or that the next request will succeed.
public struct PCCStatus: Codable, Equatable, Sendable {
    public let model: String
    public let available: Bool
    public let hasEntitlement: Bool
    public let reason: String?
    public let detail: String?
    public let quotaStatus: String
    public let quotaLimitReached: Bool

    @available(macOS 27.0, *)
    public init(snapshot: AFMFoundationPrivateCloudComputeSnapshot) {
        model = AFMFoundationProviderFactory.privateCloudComputeModelID.rawValue
        hasEntitlement = snapshot.hasEntitlement
        quotaStatus = snapshot.quotaStatus
        quotaLimitReached = snapshot.quotaIsLimitReached
        switch snapshot.availability {
        case .unavailable(let reason, let detail):
            available = false
            self.reason = reason
            self.detail = detail
        case .available:
            available = snapshot.hasEntitlement && snapshot.localeSupported && !snapshot.quotaIsLimitReached
            if !snapshot.hasEntitlement {
                reason = "missingEntitlement"
                detail = "Sign the AFM app bundle with a PCC-enabled provisioning profile."
            } else if !snapshot.localeSupported {
                reason = "unsupportedLocale"
                detail = "PCC does not support locale \(snapshot.localeIdentifier)."
            } else if snapshot.quotaIsLimitReached {
                reason = "quotaLimitReached"
                detail = snapshot.quotaLimitDetail
            } else {
                reason = nil
                detail = nil
            }
        }
    }

    @available(macOS 27.0, *)
    public static func current() -> PCCStatus {
        let hasEntitlement = AFMFoundationManagedCapabilities.currentProcessHasPrivateCloudComputeEntitlement()
        return PCCStatus(snapshot: AFMFoundationNativeProviderProbe().privateCloudComputeSnapshot(
            hasEntitlement: hasEntitlement))
    }
}
