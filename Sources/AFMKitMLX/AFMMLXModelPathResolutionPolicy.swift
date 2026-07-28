import Foundation

public enum AFMMLXCurrentModelPathResolution: Equatable, Sendable {
    case noLoadedModel
    case resolved(path: String)
    case missing(modelName: String)
}

public enum AFMMLXModelPathResolutionPolicy {
    public static func benchmarkLoadPath(
        forSelection selection: String,
        resolvedDirectory: URL?
    ) -> String? {
        let trimmedSelection = normalized(selection)
        guard trimmedSelection != nil else { return nil }
        return resolvedDirectory?.path
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
}
