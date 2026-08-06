import Foundation

public enum AFMMLXHelperModelPathPolicy {
    public static func modelPath(
        repoID: String?,
        loadedModelName: String?,
        resolvedDirectory: URL? = nil,
        fallbackCacheRoot: URL,
        loadedModelDefaultOrganization: String = "mlx-community"
    ) -> String? {
        let trimmedRepoID = normalized(repoID)
        if let trimmedRepoID {
            return resolvedDirectory?.path
                ?? fallbackCacheRoot.appendingPathComponent(trimmedRepoID).path
        }

        guard let trimmedLoadedModelName = normalized(loadedModelName) else {
            return nil
        }
        if let resolvedDirectory {
            return resolvedDirectory.path
        }
        return fallbackCacheRoot
            .appendingPathComponent(loadedModelDefaultOrganization)
            .appendingPathComponent(trimmedLoadedModelName)
            .path
    }

    private static func normalized(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}
