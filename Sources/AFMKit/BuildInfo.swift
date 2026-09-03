// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

import Foundation

public struct BuildInfo {
    public static let version: String? = "v0.9.18.1"
    static let commit: String? = nil

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
