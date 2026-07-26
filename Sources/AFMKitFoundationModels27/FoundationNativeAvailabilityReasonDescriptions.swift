#if canImport(FoundationModels)
import Foundation
import FoundationModels

public struct AFMFoundationPrivateCloudComputeQuotaLimitSnapshot: Equatable, Sendable {
    public let resetDate: Date?
    public let hasLimitIncreaseSuggestion: Bool

    public init(
        resetDate: Date?,
        hasLimitIncreaseSuggestion: Bool
    ) {
        self.resetDate = resetDate
        self.hasLimitIncreaseSuggestion = hasLimitIncreaseSuggestion
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationNativeAvailabilityReasonDescriptions {
    public static func systemLanguageModel(
        _ reason: SystemLanguageModel.Availability.UnavailableReason
    ) -> String {
        switch reason {
        case .deviceNotEligible:
            return "device does not support Apple Intelligence"
        case .appleIntelligenceNotEnabled:
            return "Apple Intelligence is not enabled for this user or locale"
        case .modelNotReady:
            return "model assets are not ready yet"
        @unknown default:
            return "unknown Apple Intelligence availability reason"
        }
    }

    public static func privateCloudCompute(
        _ reason: PrivateCloudComputeLanguageModel.Availability.UnavailableReason
    ) -> String {
        switch reason {
        case .deviceNotEligible:
            return "device does not support Apple Intelligence"
        case .systemNotReady:
            return "system is not yet ready to serve PCC requests"
        @unknown default:
            return "unknown PCC availability reason"
        }
    }

    public static func privateCloudComputeQuotaLimit(
        _ snapshot: AFMFoundationPrivateCloudComputeQuotaLimitSnapshot
    ) -> String {
        var detail = "PCC quota limit reached"
        if let resetDate = snapshot.resetDate {
            detail += "; resets \(resetDate.formatted(date: .abbreviated, time: .shortened))"
        }
        if snapshot.hasLimitIncreaseSuggestion {
            detail += "; limit increase available"
        }
        return detail
    }

    public static func privateCloudComputeQuotaLimit(
        _ quota: PrivateCloudComputeLanguageModel.QuotaUsage
    ) -> String {
        privateCloudComputeQuotaLimit(
            AFMFoundationPrivateCloudComputeQuotaLimitSnapshot(
                resetDate: quota.resetDate,
                hasLimitIncreaseSuggestion: quota.limitIncreaseSuggestion != nil
            )
        )
    }
}
#endif
