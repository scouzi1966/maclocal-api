public enum AFMMLXQuickDeletePlan: Equatable, Sendable {
    case importedReference(rawPath: String)
    case userDownloaded(repoID: String)
    case cachedModel(name: String)
    case unavailable
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
}
