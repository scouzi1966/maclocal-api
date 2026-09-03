// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

import Foundation

public struct BuildInfo {
    public static let version: String? = "v0.9.18"
    static let commit: String? = nil

    /// True only when the executable was compiled with the toolchain required
    /// to include the real Apple Foundation Models provider.
    public static var foundationModelsCompiled: Bool {
#if compiler(>=6.4) && canImport(AFMKitFoundationModels)
        true
#else
        false
#endif
    }

    public static var fullVersion: String {
        resolvedVersion(
            override: ProcessInfo.processInfo.environment["AFM_BUILD_VERSION"]
        )
    }

    public static func resolvedVersion(override: String?) -> String {
        if let override = override?.trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty {
            return override.hasPrefix("v") ? override : "v\(override)"
        }
        let base = version ?? "dev-build"
        if let commit = commit { return "\(base)-\(commit)" }
        return base
    }
}
