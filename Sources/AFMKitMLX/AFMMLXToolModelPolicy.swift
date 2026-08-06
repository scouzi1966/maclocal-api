import Foundation

public struct AFMMLXToolDownloadedModel: Equatable, Sendable {
    public let id: String
    public let name: String
    public let isVision: Bool

    public init(id: String, name: String, isVision: Bool) {
        self.id = id
        self.name = name
        self.isVision = isVision
    }
}

public struct AFMMLXToolImportedModel: Equatable, Sendable {
    public let id: String
    public let name: String
    public let path: String

    public init(id: String, name: String, path: String) {
        self.id = id
        self.name = name
        self.path = path
    }
}

public struct AFMMLXToolModelListEntry: Equatable, Sendable {
    public let id: String
    public let name: String
    public let onDisk: Bool
    public let isVision: Bool?
    public let source: String
    public let url: String?

    public init(
        id: String,
        name: String,
        onDisk: Bool,
        isVision: Bool?,
        source: String,
        url: String?
    ) {
        self.id = id
        self.name = name
        self.onDisk = onDisk
        self.isVision = isVision
        self.source = source
        self.url = url
    }
}

public enum AFMMLXToolModelResolution: Equatable, Sendable {
    case downloaded(id: String, displayName: String)
    case repositoryOnDisk(id: String, displayName: String, isVision: Bool)
    case imported(id: String, name: String, path: String)
    case missing(modelID: String)
}

public enum AFMMLXToolModelPolicy {
    public static func modelListEntries(
        downloadedModels: [AFMMLXToolDownloadedModel],
        importedModels: [AFMMLXToolImportedModel]
    ) -> [AFMMLXToolModelListEntry] {
        var entries = downloadedModels.map { model in
            AFMMLXToolModelListEntry(
                id: model.id,
                name: model.name,
                onDisk: true,
                isVision: model.isVision,
                source: AFMMLXModelCatalog.availableModels.contains { $0.repoID == model.id }
                    ? "curated"
                    : "downloaded",
                url: "https://huggingface.co/\(model.id)"
            )
        }

        for model in importedModels {
            entries.append(
                AFMMLXToolModelListEntry(
                    id: model.id,
                    name: model.name,
                    onDisk: true,
                    isVision: nil,
                    source: "imported",
                    url: nil
                )
            )
        }

        return entries
    }

    public static func resolve(
        modelID: String,
        downloadedModels: [AFMMLXToolDownloadedModel],
        importedModels: [AFMMLXToolImportedModel],
        isModelRepoOnDisk: (String) -> Bool,
        detectIsVisionFromDisk: (String) -> Bool
    ) -> AFMMLXToolModelResolution {
        if let downloaded = downloadedModels.first(where: { $0.id == modelID }) {
            return .downloaded(id: downloaded.id, displayName: downloaded.name)
        }

        if isModelRepoOnDisk(modelID) {
            return .repositoryOnDisk(
                id: modelID,
                displayName: displayName(forRepoID: modelID),
                isVision: detectIsVisionFromDisk(modelID)
            )
        }

        if let downloaded = downloadedModels.first(where: { $0.name == modelID }) {
            return .downloaded(id: downloaded.id, displayName: downloaded.name)
        }

        if let imported = importedModel(namedOrLocatedAt: modelID, importedModels: importedModels) {
            return .imported(id: imported.id, name: imported.name, path: imported.path)
        }

        return .missing(modelID: modelID)
    }

    public static func displayName(forRepoID repoID: String) -> String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }

    private static func importedModel(
        namedOrLocatedAt modelID: String,
        importedModels: [AFMMLXToolImportedModel]
    ) -> AFMMLXToolImportedModel? {
        importedModels.first { model in
            model.id == modelID ||
            model.name == modelID ||
            URL(fileURLWithPath: model.path).lastPathComponent == modelID ||
            model.path == modelID
        }
    }
}
