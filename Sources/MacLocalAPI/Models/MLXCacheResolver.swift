import Foundation
struct MLXCacheResolver {
    let cacheRoot: URL?

    init() {
        if let raw = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            cacheRoot = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
        } else {
            cacheRoot = nil
        }
    }

    func applyEnvironment() {
        // No-op: AFM cache is read-only for loading side-loaded models.
        // Downloads always go to HF hub (~/.cache/huggingface/hub).
    }

    /// Original shell CWD, captured before MLXMetalLibrary.ensureAvailable() changes the process CWD.
    /// Used to resolve relative model paths against the directory the user invoked afm from.
    private static let shellCWD: String = ProcessInfo.processInfo.environment["PWD"]
        ?? FileManager.default.currentDirectoryPath

    /// Resolve a relative path against the original shell CWD (not the process CWD,
    /// which may have been changed by MLXMetalLibrary for metallib discovery).
    private func resolveRelativePath(_ path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardized
        }
        return URL(fileURLWithPath: Self.shellCWD).appendingPathComponent(path).standardized
    }

    func normalizedModelID(_ input: String) -> String {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }
        // Absolute or relative filesystem path: resolve to absolute if it exists on disk
        if trimmed.hasPrefix("/") || trimmed.hasPrefix("./") || trimmed.hasPrefix("..") {
            let url = resolveRelativePath(trimmed)
            if FileManager.default.fileExists(atPath: url.path) {
                return url.path
            }
        }
        // Check if it's a relative path that exists on disk (e.g. "models/foo")
        if trimmed.contains("/") {
            let url = resolveRelativePath(trimmed)
            if FileManager.default.fileExists(atPath: url.path) {
                return url.path
            }
        }
        if trimmed.contains("/") { return trimmed }
        return "mlx-community/\(trimmed)"
    }

    func localModelDirectory(repoId: String) -> URL? {
        // Absolute path: check directly (no HF cache resolution)
        if repoId.hasPrefix("/") {
            let url = URL(fileURLWithPath: repoId)
            return resolvedIfComplete(url)
        }

        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        let org = parts.count > 1 ? parts[0] : "mlx-community"
        let model = parts.count > 1 ? parts[1] : repoId
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        let hfStyleName = "models--\(org)--\(model)"
        let flatName = "\(org)/\(model)"

        var candidates: [URL] = []

        // 1. MACAFM_MLX_MODEL_CACHE — side-loaded / curated models (flat layout)
        if let root = cacheRoot {
            candidates.append(root.appendingPathComponent(flatName))
            candidates.append(root.appendingPathComponent("models/\(flatName)"))
        }

        // 2. HF hub — download destination, shared with Python mlx_lm (HF-style layout)
        //    Uses same env-aware resolution as downloadModel() (HF_HUB_CACHE → HF_HOME → default)
        let hfHub = MLXModelService.resolveHFHubCache()
        candidates.append(hfHub.appendingPathComponent(hfStyleName))

        for candidate in candidates {
            if let resolved = resolvedIfComplete(candidate) {
                return resolved
            }
        }

        return nil
    }

    private func resolvedIfComplete(_ path: URL) -> URL? {
        let fm = FileManager.default
        if !fm.fileExists(atPath: path.path) { return nil }

        let snapshots = path.appendingPathComponent("snapshots")
        if fm.fileExists(atPath: snapshots.path),
           let names = try? fm.contentsOfDirectory(atPath: snapshots.path),
           let first = names.first {
            let snapshotDir = snapshots.appendingPathComponent(first)
            if hasRequiredFiles(snapshotDir) {
                return snapshotDir
            }
        }

        return hasRequiredFiles(path) ? path : nil
    }

    private func hasRequiredFiles(_ dir: URL) -> Bool {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return false }
        let hasConfig = files.contains("config.json")
        let hasWeights = files.contains(where: { $0.hasSuffix(".safetensors") || $0 == "model.safetensors.index.json" })
        return hasConfig && hasWeights
    }
}
