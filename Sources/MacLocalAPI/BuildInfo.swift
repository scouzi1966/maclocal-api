// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

struct BuildInfo {
    static let version: String? = "v0.9.5"
    static let commit: String? = nil

    static var fullVersion: String {
        let base = version ?? "dev-build"
        if let commit = commit { return "\(base)-\(commit)" }
        return base
    }
}
