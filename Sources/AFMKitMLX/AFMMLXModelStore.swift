import Foundation
import AFMKitCore
import Hub
import HuggingFace

public typealias AFMMLXModelDownloadSnapshot = @Sendable (
    _ modelID: String,
    _ matching: [String],
    _ progress: (@Sendable (Progress) -> Void)?
) async throws -> URL

public enum AFMMLXModelOrigin: String, Hashable, Sendable {
    case configuredCache
    case huggingFace
    case swiftHub
    case systemCache
    case lmStudio

    public var displayLabel: String {
        switch self {
        case .configuredCache:
            return "Configured cache"
        case .huggingFace:
            return "Hugging Face"
        case .swiftHub:
            return "Swift Hub"
        case .systemCache:
            return "System cache"
        case .lmStudio:
            return "LM Studio"
        }
    }
}

public struct AFMMLXDiscoveryLocation: Hashable, Sendable {
    public enum Layout: Hashable, Sendable {
        case flat
        case huggingFaceHub
    }

    public var directory: URL
    public var layout: Layout
    public var origin: AFMMLXModelOrigin

    public init(
        directory: URL,
        layout: Layout,
        origin: AFMMLXModelOrigin
    ) {
        self.directory = directory
        self.layout = layout
        self.origin = origin
    }
}

public struct AFMMLXDiscoveredModel: Hashable, Sendable {
    public var id: AFMModelID
    public var loadIdentifier: String
    public var localDirectory: URL
    public var packageDirectory: URL
    public var sizeBytes: Int64
    public var origin: AFMMLXModelOrigin
    public var descriptor: AFMModelDescriptor

    public init(
        id: AFMModelID,
        loadIdentifier: String,
        localDirectory: URL,
        packageDirectory: URL,
        sizeBytes: Int64,
        origin: AFMMLXModelOrigin,
        descriptor: AFMModelDescriptor
    ) {
        self.id = id
        self.loadIdentifier = loadIdentifier
        self.localDirectory = localDirectory
        self.packageDirectory = packageDirectory
        self.sizeBytes = sizeBytes
        self.origin = origin
        self.descriptor = descriptor
    }
}

public struct AFMMLXModelLoadReference: Hashable, Sendable {
    public var requestedID: String
    public var loadIdentifier: String
    public var localDirectory: URL
    public var descriptor: AFMModelDescriptor

    public init(
        requestedID: String,
        loadIdentifier: String,
        localDirectory: URL,
        descriptor: AFMModelDescriptor
    ) {
        self.requestedID = requestedID
        self.loadIdentifier = loadIdentifier
        self.localDirectory = localDirectory
        self.descriptor = descriptor
    }
}

public struct AFMMLXModelDeleteResult: Hashable, Sendable {
    public var requestedID: String
    public var removedDirectory: URL
    public var deleted: Bool

    public init(
        requestedID: String,
        removedDirectory: URL,
        deleted: Bool
    ) {
        self.requestedID = requestedID
        self.removedDirectory = removedDirectory
        self.deleted = deleted
    }
}

public struct AFMMLXModelDownloadResult: Hashable, Sendable {
    public var requestedID: String
    public var downloadedDirectory: URL
    public var loadReference: AFMMLXModelLoadReference

    public init(
        requestedID: String,
        downloadedDirectory: URL,
        loadReference: AFMMLXModelLoadReference
    ) {
        self.requestedID = requestedID
        self.downloadedDirectory = downloadedDirectory
        self.loadReference = loadReference
    }
}

public enum AFMMLXModelStoreError: Error, LocalizedError, Sendable {
    case modelNotFound(String)
    case invalidRepositoryID(String)
    case invalidModelConfiguration(String)
    case refusingToDeleteEmptyPath

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let modelID):
            return "No complete local MLX model was found for \(modelID)."
        case .invalidRepositoryID(let modelID):
            return "\(modelID) is not a valid Hugging Face repository ID."
        case .invalidModelConfiguration(let modelID):
            return "Could not decode \(modelID)'s config.json as a JSON object."
        case .refusingToDeleteEmptyPath:
            return "Refusing to delete an empty model path."
        }
    }
}

/// Shared MLX model discovery and validation for AFMKit consumers.
///
/// Hosts may supply model identifiers from their own curated lists or UI
/// registries, then use this store as the single source of truth for whether
/// those identifiers resolve to complete local model assets.
public struct AFMMLXModelStore: Sendable {
    public static let defaultDownloadPatterns = [
        "*.json",
        "*.jinja",
        "*.safetensors",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "tokenizer*",
        "*.bpe",
        "*.bin"
    ]

    private let resolver: MLXCacheResolver
    private let downloadSnapshot: AFMMLXModelDownloadSnapshot

    public init(
        resolver: MLXCacheResolver = .init(),
        downloadSnapshot: AFMMLXModelDownloadSnapshot? = nil
    ) {
        self.resolver = resolver
        self.downloadSnapshot = downloadSnapshot ?? Self.downloadSnapshot
    }

    /// Returns the complete local model directory for an identifier or path.
    public func localDirectory(for modelID: String) -> URL? {
        resolver.localModelDirectory(repoId: modelID)
    }

    /// Returns whether an identifier resolves to complete local model assets.
    public func isAvailableLocally(_ modelID: String) -> Bool {
        localDirectory(for: modelID) != nil
    }

    /// Returns whether a locally resolved model advertises vision capability
    /// through its model descriptor.
    public func isVisionModel(_ modelID: String) -> Bool {
        loadReference(for: modelID)?.descriptor.capabilities.contains(.vision) ?? false
    }

    /// Returns the identifier a host should pass to MLX loading for a complete
    /// local model. Repository IDs stay stable when they resolve through the
    /// configured cache; direct filesystem paths resolve to their complete
    /// snapshot directory.
    public func loadReference(for modelID: String) -> AFMMLXModelLoadReference? {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let localDirectory = localDirectory(for: trimmed) else {
            return nil
        }

        var descriptor = descriptor(for: localDirectory.path)
        descriptor.modelID = AFMModelID(rawValue: trimmed)
        descriptor.requiresNetwork = false

        return AFMMLXModelLoadReference(
            requestedID: trimmed,
            loadIdentifier: isPathLike(trimmed) ? localDirectory.path : trimmed,
            localDirectory: localDirectory,
            descriptor: descriptor
        )
    }

    /// Returns the directory that should be removed for a locally available
    /// model. Hugging Face cache snapshots resolve to their repository package
    /// directory (`models--org--repo`); flat cache models resolve to the model
    /// directory itself.
    public func removablePackageDirectory(for modelID: String) -> URL? {
        guard let reference = loadReference(for: modelID) else { return nil }
        return packageDirectory(containing: reference.localDirectory)
    }

    /// Deletes the local package for a model that resolves through the shared
    /// store. This intentionally operates on the package directory rather than
    /// the load directory so HF snapshot caches are removed as a unit.
    @discardableResult
    public func deleteLocalModelPackage(for modelID: String) throws -> AFMMLXModelDeleteResult {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw AFMMLXModelStoreError.refusingToDeleteEmptyPath
        }
        guard let directory = removablePackageDirectory(for: trimmed) else {
            throw AFMMLXModelStoreError.modelNotFound(trimmed)
        }

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
            return AFMMLXModelDeleteResult(
                requestedID: trimmed,
                removedDirectory: directory,
                deleted: true
            )
        }

        return AFMMLXModelDeleteResult(
            requestedID: trimmed,
            removedDirectory: directory,
            deleted: false
        )
    }

    /// Downloads a Hugging Face MLX model into the shared AFMKit/Hugging Face
    /// cache and returns the same load reference hosts should pass to MLX.
    /// If a complete local package already exists, this returns immediately
    /// without invoking the downloader.
    public func downloadModelPackage(
        for modelID: String,
        matching patterns: [String] = Self.defaultDownloadPatterns,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> AFMMLXModelDownloadResult {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw AFMMLXModelStoreError.invalidRepositoryID(modelID)
        }
        guard !isPathLike(trimmed),
              HuggingFace.Repo.ID(rawValue: trimmed) != nil else {
            throw AFMMLXModelStoreError.invalidRepositoryID(trimmed)
        }

        if let reference = loadReference(for: trimmed) {
            return AFMMLXModelDownloadResult(
                requestedID: trimmed,
                downloadedDirectory: reference.localDirectory,
                loadReference: reference
            )
        }

        let downloadedDirectory = try await downloadSnapshot(trimmed, patterns, progress)
        try? MLXModelRegistry().registerModel(trimmed)

        if let reference = loadReference(for: trimmed) ?? loadReference(for: downloadedDirectory.path) {
            return AFMMLXModelDownloadResult(
                requestedID: trimmed,
                downloadedDirectory: downloadedDirectory,
                loadReference: reference
            )
        }

        throw AFMMLXModelStoreError.modelNotFound(trimmed)
    }

    /// Describes a model using the same assets and capability inference as the
    /// AFMKit MLX provider.
    public func descriptor(for modelID: String) -> AFMModelDescriptor {
        AFMMLXModelDescriptor.describe(modelID: modelID, resolver: resolver)
    }

    /// Fetches a Hugging Face model's `config.json` and applies AFMKit's shared
    /// architecture/VLM preflight policy.
    public func preflightRemoteArchitecture(
        for modelID: String
    ) async throws -> AFMMLXModelArchitecturePreflight {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !isPathLike(trimmed),
              HuggingFace.Repo.ID(rawValue: trimmed) != nil else {
            throw AFMMLXModelStoreError.invalidRepositoryID(modelID)
        }

        let config = try await remoteModelConfiguration(for: trimmed)
        return try AFMMLXModelArchitecture.preflightConfiguration(config, modelID: trimmed)
    }

    /// Fetches a Hugging Face model's `config.json` without downloading model
    /// weights. Hosts should prefer `preflightRemoteArchitecture(for:)` when
    /// they only need loadability and modality decisions.
    public func remoteModelConfiguration(for modelID: String) async throws -> [String: Any] {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !isPathLike(trimmed),
              HuggingFace.Repo.ID(rawValue: trimmed) != nil else {
            throw AFMMLXModelStoreError.invalidRepositoryID(modelID)
        }

        guard let configURL = URL(string: "https://huggingface.co/\(trimmed)/raw/main/config.json") else {
            throw AFMMLXModelStoreError.invalidRepositoryID(trimmed)
        }

        let (data, _) = try await URLSession.shared.data(from: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AFMMLXModelStoreError.invalidModelConfiguration(trimmed)
        }
        return json
    }

    /// Returns locally available descriptors from a host-provided candidate
    /// list, preserving order while removing duplicate identifiers.
    public func localDescriptors<S: Sequence>(
        for modelIDs: S
    ) -> [AFMModelDescriptor] where S.Element == String {
        var seen = Set<String>()
        return modelIDs.compactMap { modelID in
            let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  seen.insert(trimmed).inserted,
                  isAvailableLocally(trimmed) else {
                return nil
            }
            return descriptor(for: trimmed)
        }
    }

    /// Returns whether a directory contains a complete local MLX model package.
    public static func isCompleteModelDirectory(_ directory: URL) -> Bool {
        MLXCacheResolver(cacheRoot: nil).hasRequiredFiles(directory)
    }

    /// Expands a user-visible model name or repository ID into the repository
    /// identifiers hosts should try when resolving a local MLX package.
    ///
    /// Curated catalog entries win first so app UI names map to their canonical
    /// repository IDs. Explicit repository IDs stay unchanged. Bare names then
    /// fall back to common MLX community namespaces.
    public static func identifierCandidates(
        forModelName modelName: String,
        curatedModels: [AFMMLXCuratedModel] = AFMMLXModelCatalog.availableModels,
        defaultOrganizations: [String] = ["mlx-community", "lmstudio-community"]
    ) -> [String] {
        let trimmed = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var candidates: [String] = []
        if let curated = curatedModels.first(where: {
            $0.displayName == trimmed || $0.repoID == trimmed
        }) {
            candidates.append(curated.repoID)
        }

        if trimmed.contains("/") {
            candidates.append(trimmed)
        } else {
            candidates.append(contentsOf: defaultOrganizations.map { "\($0)/\(trimmed)" })
        }

        var seen = Set<String>()
        return candidates.filter { seen.insert($0).inserted }
    }

    /// Returns whether a persisted model identifier looks like a repository or
    /// display identifier rather than an accidental filesystem path fragment.
    ///
    /// Hosts may persist user-selected model IDs over time. This helper keeps
    /// cleanup policy shared while preserving compatibility with existing
    /// app-side bare model names that are later expanded by
    /// ``identifierCandidates(forModelName:curatedModels:defaultOrganizations:)``.
    public static func isLikelyRepositoryIdentifier(_ modelID: String) -> Bool {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/") else { return false }

        let invalidAuthors = [
            "volumes",
            "users",
            "private",
            "tmp",
            "var",
            "etc",
            "library",
            "applications",
            "system",
        ]
        let parts = trimmed.lowercased().split(separator: "/")
        if let author = parts.first, invalidAuthors.contains(String(author)) {
            return false
        }

        return true
    }

    /// Returns whether a model identifier belongs to a non-chat MLX specialty
    /// family such as TTS, STT, speech, or vocoder models.
    ///
    /// AFMKit consumers can use this when presenting generic local-model
    /// discovery results so speech/audio assets are not mixed into ordinary LLM
    /// orphan-model cleanup flows.
    public static func isSpecialtyModelIdentifier(_ modelID: String) -> Bool {
        let lower = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        let knownSpecialtyRepos = [
            "prince-canuma/kokoro-82m",
            "mlx-community/orpheus-3b-0.1-ft-4bit",
            "marvis-ai/marvis-tts-100m-v0.2-mlx-6bit",
            "marvis-ai/marvis-tts-100m-v0.2-mlx-8bit",
            "marvis-ai/marvis-tts-250m-v0.2-mlx-6bit",
            "marvis-ai/marvis-tts-250m-v0.2-mlx-8bit",
        ]
        if knownSpecialtyRepos.contains(lower) { return true }

        let specialtyKeywords = [
            "tts",
            "stt",
            "whisper",
            "vocoder",
            "kokoro",
            "marvis",
            "orpheus",
            "outetts",
            "bark",
            "speecht5",
            "speech",
        ]
        if specialtyKeywords.contains(where: lower.contains) {
            return true
        }

        let specialtyOrgs = ["prince-canuma", "marvis-ai"]
        let organization = lower.split(separator: "/").first.map(String.init) ?? ""
        return specialtyOrgs.contains(organization)
    }

    /// Cleans host-persisted MLX model records without requiring AFMKit to own
    /// the host's storage type.
    ///
    /// Invalid filesystem-like identifiers and host-curated models are removed.
    /// Duplicate display names are collapsed, preferring canonical `org/model`
    /// identifiers over bare names to preserve stable Hugging Face resolution.
    public static func cleanedPersistedModelRecords<Record>(
        _ records: [Record],
        id: (Record) -> String,
        displayName: (Record) -> String,
        isCurated: (String) -> Bool = { _ in false }
    ) -> [Record] {
        var seenNames = Set<String>()
        var unique = [Record]()

        for record in records {
            let modelID = id(record).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !isCurated(modelID),
                  isLikelyRepositoryIdentifier(modelID),
                  modelID.contains("/"),
                  seenNames.insert(displayName(record)).inserted else {
                continue
            }
            unique.append(record)
        }

        for record in records {
            let modelID = id(record).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !isCurated(modelID),
                  isLikelyRepositoryIdentifier(modelID),
                  !modelID.contains("/"),
                  seenNames.insert(displayName(record)).inserted else {
                continue
            }
            unique.append(record)
        }

        return unique
    }

    /// Returns a complete snapshot directory for a specific revision inside a
    /// Hugging Face package root.
    public static func completeSnapshotDirectory(
        in repositoryDirectory: URL,
        revision: String
    ) -> URL? {
        let snapshotDirectory = repositoryDirectory
            .appendingPathComponent("snapshots")
            .appendingPathComponent(revision)
        return isCompleteModelDirectory(snapshotDirectory) ? snapshotDirectory : nil
    }

    /// Returns the newest complete snapshot directory inside a Hugging Face
    /// package root. Modification date wins; path name is a deterministic
    /// fallback for equal dates.
    public static func newestCompleteSnapshotDirectory(
        in repositoryDirectory: URL
    ) -> URL? {
        let snapshotsDirectory = repositoryDirectory.appendingPathComponent("snapshots")
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            atPath: snapshotsDirectory.path
        ) else {
            return nil
        }

        return snapshots
            .map { snapshotsDirectory.appendingPathComponent($0) }
            .filter { isCompleteModelDirectory($0) }
            .sorted { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                    ?? .distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                    ?? .distantPast
                if lhsDate == rhsDate {
                    return lhs.lastPathComponent > rhs.lastPathComponent
                }
                return lhsDate > rhsDate
            }
            .first
    }

    /// Discovers complete local models across AFM, Hugging Face, Swift Hub,
    /// system, and LM Studio caches.
    public func discoverLocalModels() -> [AFMMLXDiscoveredModel] {
        discoverLocalModels(in: defaultDiscoveryLocations())
    }

    /// Discovers complete local models in explicit locations. Supplying
    /// locations keeps tests and host-specific catalogs deterministic.
    public func discoverLocalModels(
        in locations: [AFMMLXDiscoveryLocation]
    ) -> [AFMMLXDiscoveredModel] {
        var models = [AFMMLXDiscoveredModel]()
        var seenIDs = Set<AFMModelID>()

        for location in deduplicated(locations) {
            let candidates: [(id: String, directory: URL)]
            switch location.layout {
            case .flat:
                candidates = scanFlat(location.directory)
            case .huggingFaceHub:
                candidates = scanHuggingFaceHub(location.directory)
            }

            for candidate in candidates {
                let id = AFMModelID(rawValue: candidate.id)
                guard seenIDs.insert(id).inserted,
                      let localDirectory = localDirectory(at: candidate.directory) else {
                    continue
                }
                let loadIdentifier = resolver.localModelDirectory(repoId: candidate.id) == nil
                    ? localDirectory.path
                    : candidate.id
                let packageDirectory = packageDirectory(containing: localDirectory)
                var descriptor = AFMMLXModelDescriptor.describe(
                    modelID: localDirectory.path,
                    resolver: resolver
                )
                descriptor.modelID = id
                descriptor.displayName =
                    candidate.id.split(separator: "/").last.map(String.init)
                    ?? candidate.id
                descriptor.requiresNetwork = false

                models.append(
                    AFMMLXDiscoveredModel(
                        id: id,
                        loadIdentifier: loadIdentifier,
                        localDirectory: localDirectory,
                        packageDirectory: packageDirectory,
                        sizeBytes: directorySize(at: packageDirectory),
                        origin: location.origin,
                        descriptor: descriptor
                    )
                )
            }
        }

        return models.sorted {
            $0.id.rawValue.localizedCaseInsensitiveCompare($1.id.rawValue)
                == .orderedAscending
        }
    }

    private func defaultDiscoveryLocations() -> [AFMMLXDiscoveryLocation] {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment
        var locations = [AFMMLXDiscoveryLocation]()

        if let root = resolver.cacheRoot {
            locations += [
                .init(directory: root, layout: .flat, origin: .configuredCache),
                .init(
                    directory: root.appendingPathComponent("models"),
                    layout: .flat,
                    origin: .configuredCache
                ),
                .init(
                    directory: root.appendingPathComponent("huggingface/hub"),
                    layout: .huggingFaceHub,
                    origin: .huggingFace
                )
            ]
        }

        if let documents = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first {
            locations.append(
                .init(
                    directory: documents.appendingPathComponent("huggingface/models"),
                    layout: .flat,
                    origin: .swiftHub
                )
            )
        }

        for key in ["HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"] {
            if let directory = expandedDirectory(environment[key]) {
                locations.append(
                    .init(
                        directory: directory,
                        layout: .huggingFaceHub,
                        origin: .huggingFace
                    )
                )
            }
        }
        if let home = expandedDirectory(environment["HF_HOME"]) {
            locations.append(
                .init(
                    directory: home.appendingPathComponent("hub"),
                    layout: .huggingFaceHub,
                    origin: .huggingFace
                )
            )
        }
        if let cacheHome = expandedDirectory(environment["XDG_CACHE_HOME"]) {
            locations.append(
                .init(
                    directory: cacheHome.appendingPathComponent("huggingface/hub"),
                    layout: .huggingFaceHub,
                    origin: .huggingFace
                )
            )
        }

        locations.append(
            .init(
                directory: fileManager.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cache/huggingface/hub"),
                layout: .huggingFaceHub,
                origin: .huggingFace
            )
        )

        if let library = fileManager.urls(
            for: .libraryDirectory,
            in: .userDomainMask
        ).first {
            locations += [
                .init(
                    directory: library.appendingPathComponent("Caches/models"),
                    layout: .flat,
                    origin: .systemCache
                ),
                .init(
                    directory: library.appendingPathComponent("Caches/huggingface/hub"),
                    layout: .huggingFaceHub,
                    origin: .systemCache
                )
            ]
        }
        locations.append(
            .init(
                directory: fileManager.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cache/lm-studio/models"),
                layout: .flat,
                origin: .lmStudio
            )
        )
        return locations
    }

    private func deduplicated(
        _ locations: [AFMMLXDiscoveryLocation]
    ) -> [AFMMLXDiscoveryLocation] {
        var seen = Set<String>()
        return locations.filter { location in
            let layout: String
            switch location.layout {
            case .flat:
                layout = "flat"
            case .huggingFaceHub:
                layout = "hugging-face"
            }
            return seen.insert(
                "\(location.directory.standardizedFileURL.path)|\(layout)"
            ).inserted
        }
    }

    private func scanFlat(_ root: URL) -> [(id: String, directory: URL)] {
        let fileManager = FileManager.default
        guard let organizations = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var candidates = [(id: String, directory: URL)]()
        for organization in organizations {
            guard isDirectory(organization),
                  let names = try? fileManager.contentsOfDirectory(
                      at: organization,
                      includingPropertiesForKeys: [.isDirectoryKey],
                      options: [.skipsHiddenFiles]
                  ) else {
                continue
            }
            for model in names where isDirectory(model) {
                candidates.append(
                    (
                        id: "\(organization.lastPathComponent)/\(model.lastPathComponent)",
                        directory: model
                    )
                )
            }
        }
        return candidates
    }

    private func scanHuggingFaceHub(
        _ root: URL
    ) -> [(id: String, directory: URL)] {
        let fileManager = FileManager.default
        guard let entries = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return entries.compactMap { directory in
            let name = directory.lastPathComponent
            guard isDirectory(directory), name.hasPrefix("models--") else {
                return nil
            }
            let components = name
                .dropFirst("models--".count)
                .components(separatedBy: "--")
            guard components.count >= 2 else { return nil }
            return (
                id: "\(components[0])/\(components.dropFirst().joined(separator: "--"))",
                directory: directory
            )
        }
    }

    private func localDirectory(at candidate: URL) -> URL? {
        resolver.localModelDirectory(repoId: candidate.standardizedFileURL.path)
    }

    private func packageDirectory(containing localDirectory: URL) -> URL {
        let parent = localDirectory.deletingLastPathComponent()
        if parent.lastPathComponent == "snapshots" {
            return parent.deletingLastPathComponent()
        }
        return localDirectory
    }

    private func directorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        return enumerator.reduce(into: Int64(0)) { total, item in
            guard let fileURL = item as? URL,
                  let values = try? fileURL.resourceValues(
                    forKeys: [.fileSizeKey, .isRegularFileKey]
                  ),
                  values.isRegularFile == true,
                  let fileSize = values.fileSize else {
                return
            }
            total += Int64(fileSize)
        }
    }

    private static func downloadSnapshot(
        modelID: String,
        matching patterns: [String],
        progress: (@Sendable (Progress) -> Void)?
    ) async throws -> URL {
        guard let repoID = HuggingFace.Repo.ID(rawValue: modelID) else {
            throw AFMMLXModelStoreError.invalidRepositoryID(modelID)
        }
        let cache = HubCache(cacheDirectory: MLXModelService.resolveHFHubCache())
        let client = HuggingFace.HubClient(cache: cache)
        return try await client.downloadSnapshot(
            of: repoID,
            matching: patterns,
            progressHandler: progress
        )
    }

    private func isPathLike(_ modelID: String) -> Bool {
        modelID.hasPrefix("/") || modelID.hasPrefix("./") || modelID.hasPrefix("..")
    }

    private func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private func expandedDirectory(_ rawValue: String?) -> URL? {
        guard let rawValue = rawValue?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !rawValue.isEmpty else {
            return nil
        }
        return URL(
            fileURLWithPath: NSString(string: rawValue).expandingTildeInPath
        )
    }
}
