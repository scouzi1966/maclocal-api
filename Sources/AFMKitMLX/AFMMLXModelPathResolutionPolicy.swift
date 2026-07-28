import Foundation

public enum AFMMLXCurrentModelPathResolution: Equatable, Sendable {
    case noLoadedModel
    case resolved(path: String)
    case missing(modelName: String)
}

public struct AFMMLXDownloadedModelLookupCandidate: Hashable, Sendable {
    public let id: String
    public let name: String

    public init(id: String, name: String) {
        self.id = id
        self.name = name
    }
}

public enum AFMMLXLocalModelDirectoryLookup: Hashable, Sendable {
    case importedPath(String)
    case customRepositoryID(String)
    case repositoryID(String)
    case downloadedModel(repoID: String)
    case modelName(String)
}

public enum AFMMLXModelPathResolutionPolicy {
    public static func localModelDirectoryLookups(
        forSelection selection: String,
        customModelPath: String? = nil,
        downloadedCandidates: [AFMMLXDownloadedModelLookupCandidate] = []
    ) -> [AFMMLXLocalModelDirectoryLookup] {
        guard let trimmedSelection = normalized(selection) else {
            return []
        }

        if let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedSelection) {
            return [.importedPath(importedPath)]
        }

        var lookups: [AFMMLXLocalModelDirectoryLookup] = []

        if let trimmedCustomPath = normalized(customModelPath) {
            if let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedCustomPath) {
                lookups.append(.importedPath(importedPath))
            } else {
                lookups.append(.customRepositoryID(trimmedCustomPath))
            }
        }

        if let downloaded = downloadedCandidates.first(where: {
            $0.id == trimmedSelection || $0.name == trimmedSelection
        }) {
            lookups.append(.downloadedModel(repoID: downloaded.id))
        }

        lookups.append(.modelName(trimmedSelection))
        lookups.append(.repositoryID(trimmedSelection))

        return deduplicated(lookups)
    }

    public static func benchmarkLoadPath(
        forSelection selection: String,
        resolvedDirectory: URL?
    ) -> String? {
        let trimmedSelection = normalized(selection)
        guard trimmedSelection != nil else { return nil }
        return resolvedDirectory?.path
    }

    public static func hasLocalModel(
        forSelection selection: String,
        resolvedDirectory: URL?
    ) -> Bool {
        guard normalized(selection) != nil else { return false }
        return resolvedDirectory != nil
    }

    public static func currentModelPathResolution(
        loadedModelName: String?,
        resolvedDirectory: URL?
    ) -> AFMMLXCurrentModelPathResolution {
        guard let trimmedLoadedName = normalized(loadedModelName) else {
            return .noLoadedModel
        }

        guard let resolvedDirectory else {
            return .missing(modelName: trimmedLoadedName)
        }

        return .resolved(path: resolvedDirectory.path)
    }

    private static func normalized(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private static func deduplicated(
        _ lookups: [AFMMLXLocalModelDirectoryLookup]
    ) -> [AFMMLXLocalModelDirectoryLookup] {
        var seen: Set<AFMMLXLocalModelDirectoryLookup> = []
        var result: [AFMMLXLocalModelDirectoryLookup] = []
        for lookup in lookups where !seen.contains(lookup) {
            seen.insert(lookup)
            result.append(lookup)
        }
        return result
    }
}
