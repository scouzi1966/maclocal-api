#if canImport(FoundationModels)
import FoundationModels

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
}
#endif
