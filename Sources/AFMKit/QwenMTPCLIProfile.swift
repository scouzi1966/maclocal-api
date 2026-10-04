import Foundation

/// CLI selection only. AFMKit's MLX provider owns the profile's tuning values.
public enum QwenMTPCLIProfile: String, CaseIterable, Sendable {
    case off
    case throughputV1 = "throughput-v1"
    case throughputV2 = "throughput-v2"

    public static let environmentKey = "AFM_QWEN_MTP_PROFILE"

    public enum SelectionError: LocalizedError {
        case unknown(String)

        public var errorDescription: String? {
            switch self {
            case .unknown(let value):
                return "Unknown Qwen MTP profile '\(value)'; use throughput-v1, throughput-v2 or off."
            }
        }
    }

    /// Resolve before model loading, without mutating the caller's environment.
    /// Individual provider tuning overrides remain the provider's responsibility.
    public static func resolve(
        option: Self?, environment: [String: String]
    ) throws -> Self? {
        if let option { return option }
        let value = (environment[environmentKey] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !value.isEmpty else { return nil }
        guard let profile = Self(rawValue: value) else {
            throw SelectionError.unknown(value)
        }
        return profile
    }
}
