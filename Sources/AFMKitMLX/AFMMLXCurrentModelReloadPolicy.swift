import Foundation

public enum AFMMLXCurrentModelReloadPlan: Equatable, Sendable {
    case unavailable
    case imported(name: String, path: String, isVision: Bool)
    case repository(repoID: String, isVision: Bool, forceLLMOnly: Bool)
}

public enum AFMMLXCurrentModelReloadPolicy {
    public static func make(
        loadedModelRepoID: String?,
        targetVLM: Bool
    ) -> AFMMLXCurrentModelReloadPlan {
        guard let loadedModelRepoID = normalized(loadedModelRepoID) else {
            return .unavailable
        }

        if loadedModelRepoID.hasPrefix("/") {
            return .imported(
                name: URL(fileURLWithPath: loadedModelRepoID).lastPathComponent,
                path: loadedModelRepoID,
                isVision: targetVLM
            )
        }

        return .repository(
            repoID: loadedModelRepoID,
            isVision: targetVLM,
            forceLLMOnly: !targetVLM
        )
    }

    private static func normalized(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}
