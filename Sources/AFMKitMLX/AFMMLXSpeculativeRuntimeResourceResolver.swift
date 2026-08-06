import Foundation

public enum AFMMLXSpeculativeRuntimeResourceResolver {
    public static let mtpSidecarFilename = "mtp.safetensors"

    public static func currentLoadedModelDirectory(
        loadedModelRepoID: String?,
        repositoryDirectory: (String) -> URL?
    ) -> URL? {
        guard let loadedModelRepoID else {
            return nil
        }

        let trimmed = loadedModelRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }

        if trimmed.hasPrefix("/") {
            return URL(fileURLWithPath: trimmed)
        }

        return repositoryDirectory(trimmed)
    }

    public static func mtpSidecarPath(
        modelDirectory: URL?,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> String? {
        guard let modelDirectory else {
            return nil
        }

        let path = modelDirectory
            .appendingPathComponent(mtpSidecarFilename)
            .path

        return fileExists(path) ? path : nil
    }
}
