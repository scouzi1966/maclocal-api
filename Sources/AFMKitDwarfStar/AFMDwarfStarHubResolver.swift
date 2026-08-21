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
    case unsupportedArchitecture(path: String, architecture: String)
    case unreadableArchitecture(String)

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
        case .unsupportedArchitecture(let path, let architecture):
            return "GGUF \(path) declares unsupported architecture \(architecture); DwarfStar requires general.architecture=deepseek4."
        case .unreadableArchitecture(let path):
            return "GGUF \(path) does not expose readable general.architecture metadata in its header."
        }
    }

    private static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .binary)
    }
}

public enum AFMDwarfStarHubSelector {
    public static func validateArchitecture(_ architecture: String?, path: String) throws {
        guard let architecture else {
            throw AFMDwarfStarHubSelectionError.unreadableArchitecture(path)
        }
        guard architecture == "deepseek4" else {
            throw AFMDwarfStarHubSelectionError.unsupportedArchitecture(
                path: path,
                architecture: architecture
            )
        }
    }

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
        let snapshot = try cache.snapshotPath(repo: repo, kind: .model, commitHash: revision)
        let cachedArtifact = snapshot.appendingPathComponent(entry.path)
        let blobKey = Self.hubBlobKey(repo: repo, revision: revision, entry: entry)
        let destination = try cache.blobPath(repo: repo, kind: .model, etag: blobKey)
        let partial = try cache.incompleteBlobPath(repo: repo, kind: .model, etag: blobKey)
        let segment = partial.appendingPathExtension("xet-segment")
        return try await AFMDwarfStarHubCacheCoordinator.withArtifactLock(
            cacheDirectory: cacheDirectory,
            artifact: destination
        ) {
            let cachedTarget = cachedArtifact.resolvingSymlinksInPath()
            if AFMDwarfStarResumableDownload.fileSize(cachedTarget) == expectedBytes {
                try AFMDwarfStarHubSelector.validateArchitecture(
                    AFMDwarfStarCheckpointCatalog.ggufArchitecture(at: cachedTarget),
                    path: entry.path
                )
                print("Using cached DwarfStar model: \(cachedArtifact.path)")
                return cachedArtifact
            }

            print("Download destination: \(cacheDirectory.path)")
            try FileManager.default.createDirectory(at: partial.deletingLastPathComponent(), withIntermediateDirectories: true)
            let token = try await TokenProvider.environment.getToken()
            let remoteArchitecture = try await AFMDwarfStarResumableDownload.fetchGGUFArchitecture(
                repositoryID: repositoryID,
                revision: revision,
                path: entry.path,
                token: token
            )
            try AFMDwarfStarHubSelector.validateArchitecture(
                remoteArchitecture,
                path: entry.path
            )
            let listedSHA256 = entry.oid.flatMap(Self.normalizedSHA256)
            let xetMetadata: AFMDwarfStarXetMetadata
            do {
                xetMetadata = try await AFMDwarfStarResumableDownload.fetchXetMetadata(
                    repositoryID: repositoryID,
                    revision: revision,
                    path: entry.path,
                    expectedBytes: expectedBytes,
                    token: token)
            } catch {
                print("Hugging Face Xet metadata unavailable: \(AFMDwarfStarResumableDownload.detailedError(error)); using cache-local LFS")
                xetMetadata = AFMDwarfStarXetMetadata(
                    fileID: nil,
                    expectedBytes: expectedBytes,
                    expectedSHA256: listedSHA256)
            }
        let aggregate = Progress(totalUnitCount: xetMetadata.expectedBytes)
        let file = AFMDownloadProgressUserInfo.File(
            path: entry.path,
            expectedBytes: xetMetadata.expectedBytes,
            destination: nil,
            progress: Progress(totalUnitCount: xetMetadata.expectedBytes),
            transport: "xet")
        let monitor = Task {
            while !Task.isCancelled {
                let persisted = AFMDwarfStarResumableDownload.fileSize(partial)
                let activeSegment = AFMDwarfStarResumableDownload.fileSize(segment)
                file.progress.completedUnitCount = min(
                    persisted + activeSegment,
                    xetMetadata.expectedBytes)
                AFMDownloadProgressUserInfo.enrich(aggregate, files: [file])
                progress?(aggregate)
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        defer { monitor.cancel() }
        try AFMDwarfStarResumableDownload.adoptSegment(
            segment,
            as: partial,
            expectedBytes: xetMetadata.expectedBytes)
        var offset = min(
            AFMDwarfStarResumableDownload.fileSize(partial),
            xetMetadata.expectedBytes)
        file.progress.completedUnitCount = offset
        if offset > 0 {
            print("Resuming cached partial: \(ByteCountFormatter.string(fromByteCount: offset, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: xetMetadata.expectedBytes, countStyle: .file))")
        }

        let maximumXetAttempts = 4
        var lastXetError: Error?
        if xetMetadata.fileID != nil {
            for attempt in 1 ... maximumXetAttempts where offset < xetMetadata.expectedBytes {
                try? FileManager.default.removeItem(at: segment)
                file.setTransport(attempt == 1 ? "xet" : "xet-retry-\(attempt)")
                print("Hugging Face transport selected: xet file=\(entry.path) offset=\(offset) attempt=\(attempt)/\(maximumXetAttempts)")
                do {
                    try await AFMDwarfStarResumableDownload.downloadXetRange(
                        metadata: xetMetadata,
                        repositoryID: repositoryID,
                        revision: revision,
                        offset: offset,
                        segmentURL: segment,
                        token: token)
                    try AFMDwarfStarResumableDownload.adoptSegment(
                        segment,
                        as: partial,
                        expectedBytes: xetMetadata.expectedBytes)
                    offset = AFMDwarfStarResumableDownload.fileSize(partial)
                    file.progress.completedUnitCount = offset
                    lastXetError = nil
                } catch is CancellationError {
                    try? AFMDwarfStarResumableDownload.adoptSegment(
                        segment,
                        as: partial,
                        expectedBytes: xetMetadata.expectedBytes)
                    throw CancellationError()
                } catch {
                    guard !Task.isCancelled else { throw CancellationError() }
                    try? AFMDwarfStarResumableDownload.adoptSegment(
                        segment,
                        as: partial,
                        expectedBytes: xetMetadata.expectedBytes)
                    offset = AFMDwarfStarResumableDownload.fileSize(partial)
                    file.progress.completedUnitCount = offset
                    lastXetError = error
                    let detail = AFMDwarfStarResumableDownload.detailedError(error)
                    if attempt < maximumXetAttempts {
                        let delay = min(1 << (attempt - 1), 8)
                        print("Xet interrupted at \(offset)/\(xetMetadata.expectedBytes) bytes: \(detail); retrying in \(delay)s")
                        try await Task.sleep(for: .seconds(delay))
                    }
                }
            }
        } else {
            lastXetError = AFMDwarfStarDownloadError.missingXetMetadata(entry.path)
        }

        if offset < xetMetadata.expectedBytes {
            file.setTransport("xet-fallback-lfs")
            let detail = lastXetError.map(AFMDwarfStarResumableDownload.detailedError) ?? "unknown Xet failure"
            print("Hugging Face transport fallback: Xet exhausted at \(offset)/\(xetMetadata.expectedBytes) bytes: \(detail); continuing with cache-local LFS")
            try await AFMDwarfStarResumableDownload.downloadLFSRange(
                repositoryID: repositoryID,
                revision: revision,
                path: entry.path,
                destination: partial,
                offset: offset,
                expectedBytes: xetMetadata.expectedBytes,
                token: token,
                progress: { completed in file.progress.completedUnitCount = completed })
        }

        file.setTransport("verifying-sha256")
        print("Verifying downloaded file size and SHA-256 before publication")
        try AFMDwarfStarResumableDownload.publish(
            partial: partial,
            blob: destination,
            expectedBytes: xetMetadata.expectedBytes,
            expectedSHA256: xetMetadata.expectedSHA256)
        try await cache.storeFile(
            at: destination,
            repo: repo,
            kind: .model,
            revision: revision,
            filename: entry.path,
            etag: blobKey,
            ref: "main")
        file.progress.completedUnitCount = xetMetadata.expectedBytes
        AFMDownloadProgressUserInfo.enrich(aggregate, files: [file])
        progress?(aggregate)
            return snapshot.appendingPathComponent(entry.path)
        }
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

    private static func normalizedSHA256(_ value: String) -> String? {
        let value = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard value.count == 64, value.allSatisfy(\.isHexDigit) else { return nil }
        return value
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

enum AFMDwarfStarHubCacheCoordinator {
    static func withArtifactLock<T>(
        cacheDirectory: URL,
        artifact: URL,
        operation: () async throws -> T
    ) async throws -> T {
        let cache = HubCache(cacheDirectory: cacheDirectory)
        let lock = FileLock(
            path: cache.lockPath(for: artifact),
            maxRetries: nil,
            retryDelay: 0.05
        )
        return try await lock.withLock(operation)
    }

    static func lockFileURL(cacheDirectory: URL, artifact: URL) -> URL {
        HubCache(cacheDirectory: cacheDirectory)
            .lockPath(for: artifact)
            .appendingPathExtension("lock")
    }
}
