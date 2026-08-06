import Foundation

public struct AFMMLXQuickCuratedLoadCandidate: Equatable, Sendable {
    public let id: String
    public let name: String
    public let repoID: String

    public init(id: String, name: String, repoID: String) {
        self.id = id
        self.name = name
        self.repoID = repoID
    }
}

public enum AFMMLXQuickReloadPlan: Equatable, Sendable {
    case imported(rawPath: String)
    case curated(selectionID: String)
    case userDownloaded(repoID: String)
    case unavailable
}

public struct AFMMLXImportedFallbackAccess: Equatable, Sendable {
    public let rawPath: String
    public let name: String
    public let isVision: Bool

    public init(rawPath: String, name: String, isVision: Bool) {
        self.rawPath = rawPath
        self.name = name
        self.isVision = isVision
    }
}

public enum AFMMLXQuickReloadPolicy {
    public static func make(
        loadedModelRepoID: String?,
        loadedModelName: String?,
        curatedCandidates: [AFMMLXQuickCuratedLoadCandidate],
        downloadedIDs: [String]
    ) -> AFMMLXQuickReloadPlan {
        let trimmedRepoID = loadedModelRepoID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let trimmedName = loadedModelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        if let importedPath = importedPath(from: trimmedRepoID) {
            return .imported(rawPath: importedPath)
        }

        if let curated = curatedCandidates.first(where: { candidate in
            candidate.id == trimmedRepoID || candidate.name == trimmedName
        }) {
            return .curated(selectionID: curated.id)
        }

        if !trimmedRepoID.isEmpty, downloadedIDs.contains(trimmedRepoID) {
            return .userDownloaded(repoID: trimmedRepoID)
        }

        return .unavailable
    }

    public static func importedPath(from selection: String) -> String? {
        let trimmed = selection.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("imported:") {
            return String(trimmed.dropFirst("imported:".count))
        }
        return trimmed.hasPrefix("/") ? trimmed : nil
    }

    public static func fallbackImportedAccess(
        rawPath: String,
        isVision: Bool
    ) -> AFMMLXImportedFallbackAccess? {
        guard let importedPath = importedPath(from: rawPath) else {
            return nil
        }
        return AFMMLXImportedFallbackAccess(
            rawPath: importedPath,
            name: URL(fileURLWithPath: importedPath).lastPathComponent,
            isVision: isVision
        )
    }
}
