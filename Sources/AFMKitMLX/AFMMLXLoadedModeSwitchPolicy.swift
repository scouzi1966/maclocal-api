public enum AFMMLXLoadedModeSwitchPlan: Equatable, Sendable {
    case imported(rawPath: String, targetVLM: Bool)
    case currentLoadedModel(targetVLM: Bool)

    public var targetVLM: Bool {
        switch self {
        case .imported(_, let targetVLM), .currentLoadedModel(let targetVLM):
            targetVLM
        }
    }
}

public enum AFMMLXLoadedModeSwitchPolicy {
    public static func make(
        loadedModelRepoID: String?,
        loadedModelType: String?,
        isLoadedModelVLM: Bool,
        loadedModelDirectoryIsVision: Bool
    ) -> AFMMLXLoadedModeSwitchPlan? {
        let trimmedModelType = loadedModelType?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmedModelType.isEmpty,
              AFMMLXModelArchitecture.isDualModeModelType(trimmedModelType),
              loadedModelDirectoryIsVision else {
            return nil
        }

        let targetVLM = !isLoadedModelVLM
        let trimmedRepoID = loadedModelRepoID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedRepoID) {
            return .imported(rawPath: importedPath, targetVLM: targetVLM)
        }
        return .currentLoadedModel(targetVLM: targetVLM)
    }
}
