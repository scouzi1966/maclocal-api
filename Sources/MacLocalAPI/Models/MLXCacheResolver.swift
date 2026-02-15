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
        guard let root = cacheRoot else { return }
        let rootPath = root.path
        let hubPath = root.appendingPathComponent("huggingface/hub").path
        setenv("HF_HOME", rootPath, 1)
        setenv("HUGGINGFACE_HUB_CACHE", hubPath, 1)
    }

    func normalizedModelID(_ input: String) -> String {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }
        if trimmed.contains("/") { return trimmed }
        return "mlx-community/\(trimmed)"
    }

    func localModelDirectory(repoId: String) -> URL? {
        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        let org = parts.count > 1 ? parts[0] : "mlx-community"
        let model = parts.count > 1 ? parts[1] : repoId
        let fm = FileManager.default

        var candidates: [URL] = []
        if let root = cacheRoot {
            // Vesta-style layout: <root>/<org>/<model>
            candidates.append(root.appendingPathComponent("\(org)/\(model)"))
            // Alternate explicit models dir: <root>/models/<org>/<model>
            candidates.append(root.appendingPathComponent("models/\(org)/\(model)"))
            candidates.append(root.appendingPathComponent("huggingface/hub/models--\(org)--\(model)"))
        }
        if let library = fm.urls(for: .libraryDirectory, in: .userDomainMask).first {
            candidates.append(library.appendingPathComponent("Caches/models/\(org)/\(model)"))
            candidates.append(library.appendingPathComponent("Caches/huggingface/hub/models--\(org)--\(model)"))
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
