public enum AFMMLXQuickDeletePlan: Equatable, Sendable {
    case importedReference(rawPath: String)
    case userDownloaded(repoID: String)
    case cachedModel(name: String)
    case unavailable
}

public struct AFMMLXDownloadedModelDeletionPlan: Equatable, Sendable {
    public let repoID: String
    public let shouldUnloadCurrentModel: Bool

    public init(repoID: String, shouldUnloadCurrentModel: Bool) {
        self.repoID = repoID
        self.shouldUnloadCurrentModel = shouldUnloadCurrentModel
    }
}

public enum AFMMLXQuickDeletePolicy {
    public static func make(
        selectionID: String,
        downloadedIDs: [String]
    ) -> AFMMLXQuickDeletePlan {
        let trimmedSelection = selectionID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedSelection.isEmpty else { return .unavailable }

        if trimmedSelection.hasPrefix("imported:"),
           let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedSelection) {
            return .importedReference(rawPath: importedPath)
        }

        if downloadedIDs.contains(trimmedSelection) {
            return .userDownloaded(repoID: trimmedSelection)
        }

        let name = trimmedSelection.split(separator: "/").last.map(String.init) ?? trimmedSelection
        guard !name.isEmpty else { return .unavailable }
        return .cachedModel(name: name)
    }

    public static func downloadedModelDeletionPlan(
        repoID: String,
        isModelLoaded: Bool,
        loadedModelName: String?
    ) -> AFMMLXDownloadedModelDeletionPlan {
        let trimmedRepoID = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedLoadedName = loadedModelName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let repoModelName = trimmedRepoID
            .split(separator: "/")
            .last
            .map(String.init) ?? trimmedRepoID

        return AFMMLXDownloadedModelDeletionPlan(
            repoID: trimmedRepoID,
            shouldUnloadCurrentModel: isModelLoaded
                && !repoModelName.isEmpty
                && trimmedLoadedName == repoModelName
        )
    }
}
