#if canImport(FoundationModels)
import FoundationModels

public enum AFMFoundationToolAvailabilityPolicy {
    public static func includesVisionTools(
        includesImageInput: Bool,
        supportsAppleVisionTools: Bool
    ) -> Bool {
        includesImageInput && supportsAppleVisionTools
    }
}
#endif
