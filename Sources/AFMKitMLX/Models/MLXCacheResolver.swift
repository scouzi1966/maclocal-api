import Foundation
public struct MLXCacheResolver: Sendable {
    public let cacheRoot: URL?

    public init() {
        if let raw = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            cacheRoot = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
        } else {
            cacheRoot = nil
        }
    }

    public init(cacheRoot: URL?) {
        self.cacheRoot = cacheRoot
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

    /// Resolve a user-provided filesystem path against the shell working
    /// directory captured before MLX changes the process working directory.
    /// Returns nil when the input does not currently identify a local file or
    /// directory, allowing the caller to treat it as a Hub repository ID.
    func localFilesystemURLIfExists(_ path: String) -> URL? {
        let expanded = NSString(string: path).expandingTildeInPath
        let resolved = resolveRelativePath(expanded)
        guard FileManager.default.fileExists(atPath: resolved.path) else { return nil }
        return resolved
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

    public func localModelDirectory(repoId: String) -> URL? {
        // Absolute path: check directly (no HF cache resolution)
        if repoId.hasPrefix("/") {
            let url = URL(fileURLWithPath: repoId)
            return resolvedIfComplete(url)
        }

        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        let org = parts.count > 1 ? parts[0] : "mlx-community"
        let model = parts.count > 1 ? parts[1] : repoId
        let hfStyleName = "models--\(org)--\(model)"
        let flatName = "\(org)/\(model)"

        var candidates: [URL] = []
        var seenCandidates = Set<String>()

        func appendCandidate(_ url: URL) {
            let path = url.standardizedFileURL.path
            guard seenCandidates.insert(path).inserted else { return }
            candidates.append(url)
        }

        // 1. MACAFM_MLX_MODEL_CACHE — side-loaded / curated models (flat layout)
        if let root = cacheRoot {
            appendCandidate(root.appendingPathComponent(flatName))
            appendCandidate(root.appendingPathComponent("models/\(flatName)"))
        }

        let fileManager = FileManager.default

        // 2. Swift Hub / Vesta-style flat model roots.
        if let documents = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first {
            appendCandidate(
                documents.appendingPathComponent("huggingface/models/\(flatName)")
            )
        }
        if let library = fileManager.urls(
            for: .libraryDirectory,
            in: .userDomainMask
        ).first {
            appendCandidate(library.appendingPathComponent("Caches/models/\(flatName)"))
        }
        appendCandidate(
            fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/lm-studio/models/\(flatName)")
        )

        // 3. HF hub — download destination, shared with Python mlx_lm (HF-style layout)
        //    Uses same env-aware resolution as downloadModel() (HF_HUB_CACHE → HF_HOME → default)
        let hfHub = MLXModelService.resolveHFHubCache()
        appendCandidate(hfHub.appendingPathComponent(hfStyleName))
        if let library = fileManager.urls(
            for: .libraryDirectory,
            in: .userDomainMask
        ).first {
            appendCandidate(
                library.appendingPathComponent("Caches/huggingface/hub/\(hfStyleName)")
            )
        }

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

        if let snapshotDir = AFMMLXModelStore.newestCompleteSnapshotDirectory(in: path) {
            return snapshotDir
        }

        return hasRequiredFiles(path) ? path : nil
    }

    func hasRequiredFiles(_ dir: URL) -> Bool {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return false }
        let hasConfig = files.contains("config.json")
        let standaloneWeights = files.filter {
            $0.hasSuffix(".safetensors") && !$0.hasPrefix("model-")
        }
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        let hasWeights: Bool
        if files.contains("model.safetensors.index.json") {
            guard let data = try? Data(contentsOf: indexURL),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = object["weight_map"] as? [String: Any] else {
                return false
            }
            let shardNames = Set(weightMap.values.compactMap { $0 as? String })
            hasWeights = !shardNames.isEmpty && shardNames.allSatisfy {
                FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
        } else {
            hasWeights = !standaloneWeights.isEmpty
        }
        return hasConfig && hasWeights
    }
}
