import Foundation

public enum AFMMLXQuickLoadPlan: Equatable, Sendable {
    case imported(rawPath: String)
    case curatedStandard(selectionID: String)
    case curatedDualMode(repoID: String, isVision: Bool, forceLLMOnly: Bool)
    case userDownloaded(repoID: String, isVisionOverride: Bool?)
    case fallback(path: String, name: String)
}

public struct AFMMLXSelectedLoadDownloadedCandidate: Equatable, Sendable {
    public let id: String
    public let name: String

    public init(id: String, name: String) {
        self.id = id
        self.name = name
    }
}

public enum AFMMLXSelectedLoadPlan: Equatable, Sendable {
    case imported(rawPath: String)
    case userDownloaded(repoID: String, isVisionOverride: Bool?)
    case curated
}

public enum AFMMLXNamedLoadPlan: Equatable, Sendable {
    case userDownloaded(repoID: String)
    case curated(modelName: String)
    case unavailable
}

public struct AFMMLXCuratedSelection: Equatable, Sendable {
    public let modelName: String
    public let afm27ModelID: String

    public init(modelName: String, afm27ModelID: String) {
        self.modelName = modelName
        self.afm27ModelID = afm27ModelID
    }
}

public struct AFMMLXSelectedNameChangePlan: Equatable, Sendable {
    public let shouldUnloadLoadedModel: Bool
    public let curatedModelName: String?

    public init(shouldUnloadLoadedModel: Bool, curatedModelName: String?) {
        self.shouldUnloadLoadedModel = shouldUnloadLoadedModel
        self.curatedModelName = curatedModelName
    }
}

public enum AFMMLXLoadSelectionPolicy {
    public static func quickLoadPlan(
        for selectionID: String,
        curatedCandidates: [AFMMLXQuickCuratedLoadCandidate],
        downloadedIDs: [String],
        isDualMode: Bool,
        loadAsVLM: Bool
    ) -> AFMMLXQuickLoadPlan {
        let trimmedSelection = selectionID.trimmingCharacters(in: .whitespacesAndNewlines)
        if let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedSelection),
           trimmedSelection.hasPrefix("imported:") {
            return .imported(rawPath: importedPath)
        }

        if let curated = curatedCandidates.first(where: { $0.id == trimmedSelection }) {
            if isDualMode {
                return .curatedDualMode(
                    repoID: curated.repoID,
                    isVision: loadAsVLM,
                    forceLLMOnly: !loadAsVLM
                )
            }
            return .curatedStandard(selectionID: curated.id)
        }

        if downloadedIDs.contains(trimmedSelection) {
            return .userDownloaded(
                repoID: trimmedSelection,
                isVisionOverride: isDualMode ? loadAsVLM : nil
            )
        }

        return .fallback(
            path: trimmedSelection,
            name: fallbackDisplayName(for: trimmedSelection)
        )
    }

    public static func selectedLoadPlan(
        modelName: String,
        customModelPath: String?,
        downloadedCandidates: [AFMMLXSelectedLoadDownloadedCandidate],
        isDualMode: Bool,
        textOnlyMode: Bool
    ) -> AFMMLXSelectedLoadPlan {
        let trimmedName = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedPath = customModelPath?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        if trimmedPath.hasPrefix("/") {
            return .imported(rawPath: trimmedPath)
        }

        if let downloaded = downloadedCandidates.first(where: { $0.name == trimmedName }) {
            return .userDownloaded(
                repoID: downloaded.id,
                isVisionOverride: isDualMode ? !textOnlyMode : nil
            )
        }

        return .curated
    }

    public static func namedLoadPlan(
        modelName: String,
        downloadedCandidates: [AFMMLXSelectedLoadDownloadedCandidate],
        curatedModelNames: [String]
    ) -> AFMMLXNamedLoadPlan {
        let trimmedName = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return .unavailable }

        if let downloaded = downloadedCandidates.first(where: { $0.name == trimmedName }) {
            return .userDownloaded(repoID: downloaded.id)
        }

        if curatedModelNames.contains(trimmedName) {
            return .curated(modelName: trimmedName)
        }

        return .unavailable
    }

    public static func curatedSelection(
        modelName: String,
        customModelPath: String?
    ) -> AFMMLXCuratedSelection {
        let trimmedName = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedPath = customModelPath?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let afm27ModelID = trimmedPath.isEmpty ? "mlx-community/\(trimmedName)" : trimmedPath
        return AFMMLXCuratedSelection(
            modelName: trimmedName,
            afm27ModelID: afm27ModelID
        )
    }

    public static func selectedNameChangePlan(
        oldModelName: String,
        newModelName: String,
        loadedModelName: String?,
        hasAppearedOnce: Bool,
        isModelLoaded: Bool,
        selectedModelCustomPath: String?,
        importedModelNames: [String],
        curatedModelNames: [String]
    ) -> AFMMLXSelectedNameChangePlan {
        let trimmedNewName = newModelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedLoadedName = loadedModelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let isSyncingToLoadedModel = !trimmedLoadedName.isEmpty && trimmedNewName == trimmedLoadedName
        let shouldUnloadLoadedModel = hasAppearedOnce
            && isModelLoaded
            && !isSyncingToLoadedModel
            && modelFamily(for: oldModelName) != modelFamily(for: trimmedNewName)

        let customPath = selectedModelCustomPath?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let isImportedSelection = importedModelNames.contains(trimmedNewName) || customPath.hasPrefix("/")
        let curatedModelName = isImportedSelection
            ? nil
            : curatedModelNames.first(where: { $0 == trimmedNewName })

        return AFMMLXSelectedNameChangePlan(
            shouldUnloadLoadedModel: shouldUnloadLoadedModel,
            curatedModelName: curatedModelName
        )
    }

    public static func modelFamily(for modelName: String) -> String {
        let components = modelName.split(separator: "-")
        if components.count >= 3 {
            return components[0...2].joined(separator: "-")
        }
        return modelName
    }

    public static func fallbackDisplayName(for selectionID: String) -> String {
        let trimmedSelection = selectionID.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedSelection.split(separator: "/").last.map(String.init) ?? trimmedSelection
    }
}
