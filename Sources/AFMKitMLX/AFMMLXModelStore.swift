import Foundation
import AFMKitCore

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
    public var origin: AFMMLXModelOrigin
    public var descriptor: AFMModelDescriptor

    public init(
        id: AFMModelID,
        loadIdentifier: String,
        localDirectory: URL,
        origin: AFMMLXModelOrigin,
        descriptor: AFMModelDescriptor
    ) {
        self.id = id
        self.loadIdentifier = loadIdentifier
        self.localDirectory = localDirectory
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

/// Shared MLX model discovery and validation for AFMKit consumers.
///
/// Hosts may supply model identifiers from their own curated lists or UI
/// registries, then use this store as the single source of truth for whether
/// those identifiers resolve to complete local model assets.
public struct AFMMLXModelStore: Sendable {
    private let resolver: MLXCacheResolver

    public init(resolver: MLXCacheResolver = .init()) {
        self.resolver = resolver
    }

    /// Returns the complete local model directory for an identifier or path.
    public func localDirectory(for modelID: String) -> URL? {
        resolver.localModelDirectory(repoId: modelID)
    }

    /// Returns whether an identifier resolves to complete local model assets.
    public func isAvailableLocally(_ modelID: String) -> Bool {
        localDirectory(for: modelID) != nil
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

    /// Describes a model using the same assets and capability inference as the
    /// AFMKit MLX provider.
    public func descriptor(for modelID: String) -> AFMModelDescriptor {
        AFMMLXModelDescriptor.describe(modelID: modelID, resolver: resolver)
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
