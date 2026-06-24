// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

public struct BuildInfo {
    public static let version: String? = "v0.9.13"
    static let commit: String? = nil

    public static var fullVersion: String {
        let base = version ?? "dev-build"
        if let commit = commit { return "\(base)-\(commit)" }
        return base
    }
}
