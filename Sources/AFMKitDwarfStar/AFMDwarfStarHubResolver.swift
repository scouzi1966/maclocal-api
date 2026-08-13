import Foundation
import CryptoKit
import AFMKitCore
import HuggingFace

public struct AFMDwarfStarHubArtifact: Equatable, Sendable {
    public enum Role: Equatable, Sendable {
        case model
        case speculativeSupport
    }

    public let path: String
    public let size: Int64?
    public let role: Role

    public init(path: String, size: Int64?) {
        self.path = path
        self.size = size
        self.role = Self.classify(path: path)
    }

    private static func classify(path: String) -> Role {
        let name = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return name.contains("dspark") || name.contains("speculator") || name.contains("draft")
            ? .speculativeSupport
            : .model
    }
}

public enum AFMDwarfStarHubSelectionError: LocalizedError, Equatable {
    case invalidRepositoryID(String)
    case noModelGGUF(String)
    case requestedFileNotFound(String)
    case modelExceedsMemoryBudget(path: String, size: Int64, budget: Int64)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            return "Invalid Hugging Face repository ID: \(id)"
        case .noModelGGUF(let id):
            return "Repository \(id) does not contain a DwarfStar model GGUF."
        case .requestedFileNotFound(let path):
            return "Requested GGUF file was not found in the repository: \(path)"
        case .modelExceedsMemoryBudget(let path, let size, let budget):
            return "GGUF \(path) is \(Self.bytes(size)), exceeding the \(Self.bytes(budget)) memory budget. Use --gguf-file to select it explicitly."
        }
    }

    private static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .binary)
    }
}

public enum AFMDwarfStarHubSelector {
    public static func selectModel(
        from artifacts: [AFMDwarfStarHubArtifact],
        repositoryID: String,
        requestedPath: String? = nil,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory,
        memoryFraction: Double = 0.80
    ) throws -> AFMDwarfStarHubArtifact {
        let candidates = artifacts.filter {
            $0.role == .model && $0.path.lowercased().hasSuffix(".gguf")
        }
        guard !candidates.isEmpty else {
            throw AFMDwarfStarHubSelectionError.noModelGGUF(repositoryID)
        }

        if let requestedPath {
            guard let exact = candidates.first(where: { $0.path == requestedPath }) else {
                throw AFMDwarfStarHubSelectionError.requestedFileNotFound(requestedPath)
            }
            return exact
        }

        let fraction = min(max(memoryFraction, 0.1), 1.0)
        let budget = Int64(min(Double(Int64.max), Double(physicalMemory) * fraction))
        let fitting = candidates.filter { artifact in
            guard let size = artifact.size else { return true }
            return size <= budget
        }
        guard !fitting.isEmpty else {
            let smallest = candidates.min { ($0.size ?? .max) < ($1.size ?? .max) }!
            throw AFMDwarfStarHubSelectionError.modelExceedsMemoryBudget(
                path: smallest.path,
                size: smallest.size ?? .max,
                budget: budget)
        }

        // Prefer the highest-fidelity artifact that fits. Unknown sizes sort
        // last so a repository with complete metadata remains deterministic.
        return fitting.sorted {
            switch ($0.size, $1.size) {
            case let (lhs?, rhs?) where lhs != rhs: return lhs > rhs
            case (_?, nil): return true
            case (nil, _?): return false
            default: return $0.path.localizedStandardCompare($1.path) == .orderedAscending
            }
        }.first!
    }
}

public struct AFMDwarfStarHubResolver: Sendable {
    private let cacheDirectory: URL

    public init(cacheDirectory: URL = Self.defaultCacheDirectory()) {
        self.cacheDirectory = cacheDirectory
    }

    public func resolve(
        repositoryID: String,
        requestedPath: String? = nil,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        guard let repo = HuggingFace.Repo.ID(rawValue: repositoryID) else {
            throw AFMDwarfStarHubSelectionError.invalidRepositoryID(repositoryID)
        }
        let cache = HubCache(cacheDirectory: cacheDirectory)
        let client = HuggingFace.HubClient(cache: cache)
        let revision = try await client.getModel(repo).sha ?? "main"
        let entries = try await client.listFiles(in: repo, revision: revision).filter {
            $0.type == .file && $0.path.lowercased().hasSuffix(".gguf")
        }
        let artifact = try AFMDwarfStarHubSelector.selectModel(
            from: entries.map {
                AFMDwarfStarHubArtifact(path: $0.path, size: $0.size.map(Int64.init))
            },
            repositoryID: repositoryID,
            requestedPath: requestedPath)
        guard let entry = entries.first(where: { $0.path == artifact.path }) else {
            throw AFMDwarfStarHubSelectionError.requestedFileNotFound(artifact.path)
        }

        let expectedBytes = max(Int64(entry.size ?? 1), 1)
        let blobKey = Self.hubBlobKey(repo: repo, revision: revision, entry: entry)
        let destination = try cache.blobPath(repo: repo, kind: .model, etag: blobKey)
        let aggregate = Progress(totalUnitCount: expectedBytes)
        let file = AFMDownloadProgressUserInfo.File(
            path: entry.path,
            expectedBytes: expectedBytes,
            destination: destination,
            progress: Progress(totalUnitCount: expectedBytes),
            transport: "xet")
        let monitor = Task {
            while !Task.isCancelled {
                AFMDownloadProgressUserInfo.enrich(aggregate, files: [file])
                progress?(aggregate)
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        defer { monitor.cancel() }

        print("Download destination: \(cacheDirectory.path)")
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        do {
            print("Hugging Face transport selected: xet file=\(entry.path)")
            _ = try await client.downloadFile(
                entry,
                from: repo,
                to: destination,
                revision: revision,
                progress: file.progress,
                transport: .xet)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            guard !Task.isCancelled else { throw CancellationError() }
            file.setTransport("xet-fallback-lfs")
            print("Hugging Face transport fallback: xet failed for \(entry.path): \(error.localizedDescription); retrying with lfs")
            try? FileManager.default.removeItem(at: destination)
            _ = try await client.downloadFile(
                entry,
                from: repo,
                to: destination,
                revision: revision,
                progress: file.progress,
                transport: .lfs)
        }
        try await cache.storeFile(
            at: destination,
            repo: repo,
            kind: .model,
            revision: revision,
            filename: entry.path,
            etag: blobKey,
            ref: "main")
        file.progress.completedUnitCount = expectedBytes
        AFMDownloadProgressUserInfo.enrich(aggregate, files: [file])
        progress?(aggregate)
        let snapshot = try cache.snapshotPath(repo: repo, kind: .model, commitHash: revision)
        return snapshot.appendingPathComponent(entry.path)
    }

    private static func hubBlobKey(
        repo: HuggingFace.Repo.ID,
        revision: String,
        entry: Git.TreeEntry
    ) -> String {
        let identity = "\(repo.description)|\(revision)|\(entry.path)|\(entry.oid ?? "")"
        return SHA256.hash(data: Data(identity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    public static func defaultCacheDirectory() -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["HF_HUB_CACHE"] ?? environment["HUGGINGFACE_HUB_CACHE"],
           !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: value).expandingTildeInPath)
        }
        if let value = environment["HF_HOME"],
           !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: value).expandingTildeInPath)
                .appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }
}
